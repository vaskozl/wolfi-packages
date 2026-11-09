"""decide-export: export a Base Labs decide model (Qwen3.5 backbone, block-bidirectional
full attention, slot_lm readout) to an OpenVINO fork graph.

Decide model format 1
=====================

A model directory holds fork.xml/fork.bin, openvino_tokenizer.xml/.bin, decide.json and the
HF metadata (config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja,
readout.json, generation_config.json).

fork.xml. B = rows, T = chunk length, P = past length, K = slots; all dynamic.

Inputs (integers are i64, caches f32):
  input_ids   [B,T]  token ids, right padded
  lengths     [B]    valid tokens per row
  offset      [B]    absolute position of the chunk's first token (RoPE); past key j is
                     visible to row b iff j < offset[b]
  read_index  [B]    chunk index whose final-normed hidden state is read
  slot_ids    [K]    vocabulary ids of the answer slots
  past.L.key, past.L.value      [B,KV,P,D]  full-attention layer L
  past.L.conv                   [B,C,W]     linear-attention layer L, HF conv state layout
  past.L.recurrent              [B,H,Dk,Dv] linear-attention layer L

Outputs:
  slot_logits [B,K]  LM-head logits at read_index, gathered at slot_ids
  lse         [B]    logsumexp of the LM-head logits over the whole vocabulary at read_index
  present.*          one per past.*, same suffix; key/value grow to P+T, conv and
                     recurrent keep their shapes

One call is one attention block: in full-attention layers query i of row b sees every
visible past key and chunk keys 0..lengths[b]-1 (bidirectional); linear-attention
(GatedDeltaNet) layers are causal. present.* is exact only for unpadded rows. The past
batch must equal B.

Forking: call once with the shared prefix (B=1, P=0 caches), tile its present.* to B rows,
then call once with the B question suffixes, offset = prefix length and
read_index = lengths-1. The answer distribution is softmax(slot_logits[:, :n_options]).

decide.json: {"format": 1, "slot_ids": [...256 ids, slot label order...],
              "past": [{"name": "past.0.conv", "shape": [1, C, W]}, ...]}
with B=1 and -1 only for P.
"""
from __future__ import annotations

import argparse
import bisect
import gc
import importlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

__version__ = "1.0.0"
FORMAT = 1

LINEAR, FULL = "linear_attention", "full_attention"
METADATA = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "readout.json", "generation_config.json")

# Flide plugins/flide_onejev/prompt.py (onejev-labels-v2).
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ",
                  "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
SLOT_LABELS = tuple(
    list(LETTERS) + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)])

TOKENIZER_PROBE = "<|im_start|>user\nhi<|im_end|>"
TOKENIZER_PROBE_IDS = [248045, 846, 198, 5834, 248046]

CHECK_STATE = (
    "Ticket #4417 from a customer: \"The blender arrived yesterday with a cracked jar. "
    "I've attached a photo. I'd like my money back, please, I don't want a replacement.\"\n"
    "Order status: delivered 2026-10-30. Payment: card, captured."
)
CHECK_QUESTIONS = (
    "Is the customer asking for a refund?",
    "Is the customer satisfied with the product they received?",
)
# bf16 torch reference vs f32 OpenVINO: random weights at joey dims gave 3e-3.
REAL_TOLERANCE = 2e-2
# Selftest compares against an f32 torch reference of the same bf16 weights.
SELFTEST_TOLERANCE = 1e-3


def log(msg: str) -> None:
    print(f"decide-export: {msg}", file=sys.stderr, flush=True)


def log_peak(stage: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    msg = f"{stage}: peak RSS {rss:.2f} GiB"
    try:
        msg += f", cgroup memory.peak {int(Path('/sys/fs/cgroup/memory.peak').read_text()) / 2**30:.2f} GiB"
    except (OSError, ValueError):
        pass
    log(msg)


def fail(msg: str) -> None:
    raise SystemExit(f"decide-export: {msg}")


# --- model -------------------------------------------------------------------------------


def validate_source(src: Path) -> dict:
    config = json.loads((src / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        fail(f"model_type {config.get('model_type')!r} is not qwen3_5")
    attention = config.get("modeling_overrides", {}).get("attention")
    if attention != {FULL: "block_bidirectional", LINEAR: "causal"}:
        fail(f"modeling_overrides.attention {attention!r} is not block_bidirectional/causal")
    readout = json.loads((src / "readout.json").read_text())
    if readout.get("readout") != "slot_lm":
        fail(f"readout {readout.get('readout')!r} is not slot_lm")
    if (src / "calibration.json").exists():
        fail("calibration.json is not supported")
    return config


def load_text_model(src: Path, config: dict):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    tc = Qwen3_5TextConfig(**config["text_config"])
    with torch.device("meta"):
        lm = mq.Qwen3_5ForCausalLM(tc)
    # Vision tower (model.visual.*) and MTP (mtp.*) are not part of the text model.
    prefix = "model.language_model."
    state = {}
    shards = sorted(src.glob("model*.safetensors"))
    if not shards:
        fail(f"no safetensors in {src}")
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state["model." + key[len(prefix):]] = f.get_tensor(key).to(torch.bfloat16)
    missing, unexpected = lm.load_state_dict(state, strict=False, assign=True)
    del state
    if set(missing) - {"lm_head.weight"} or unexpected:
        fail(f"weights do not match the text config: missing {missing}, unexpected {unexpected}")
    return finish_model(lm)


def finish_model(lm):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if not lm.config.tie_word_embeddings:
        fail("untied LM head is not supported")
    lm.lm_head.weight = lm.model.embed_tokens.weight
    # Non-persistent buffers are still on the meta device after a meta-device load.
    lm.model.rotary_emb = mq.Qwen3_5TextRotaryEmbedding(lm.config)
    lm.config._attn_implementation = "sdpa"
    return lm.eval()


def past_specs(tc) -> list[tuple[str, list[int]]]:
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    conv_dim = 2 * tc.linear_key_head_dim * tc.linear_num_key_heads + tc.linear_value_head_dim * tc.linear_num_value_heads
    specs = []
    for i, kind in enumerate(tc.layer_types):
        if kind == LINEAR:
            specs.append((f"past.{i}.conv", [1, conv_dim, tc.linear_conv_kernel_dim]))
            specs.append((f"past.{i}.recurrent",
                          [1, tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim]))
        elif kind == FULL:
            specs.append((f"past.{i}.key", [1, tc.num_key_value_heads, -1, hd]))
            specs.append((f"past.{i}.value", [1, tc.num_key_value_heads, -1, hd]))
        else:
            fail(f"unknown layer type {kind!r}")
    return specs


def make_fork(lm):
    """The fork graph as a torch module over the model's own submodules and weights."""
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    from ov_gdn import RecurrentAttentionCell, ov_causal_conv1d, patched_recurrent_gated_delta_rule

    tc = lm.config
    for layer, kind in zip(lm.model.layers, tc.layer_types):
        if kind == LINEAR:
            layer.linear_attn.recurrent_attention_cell = RecurrentAttentionCell()

    def full_attention(sa, x, cos, sin, past_k, past_v, bias):
        b, t, _ = x.shape
        q, gate = torch.chunk(sa.q_proj(x).view(b, t, -1, sa.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(b, t, -1)
        q = sa.q_norm(q).transpose(1, 2)
        k = sa.k_norm(sa.k_proj(x).view(b, t, -1, sa.head_dim)).transpose(1, 2)
        v = sa.v_proj(x).view(b, t, -1, sa.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        o = F.scaled_dot_product_attention(q, repeat_kv(k, sa.num_key_value_groups),
                                           repeat_kv(v, sa.num_key_value_groups), attn_mask=bias, scale=sa.scaling)
        o = o.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
        return sa.o_proj(o), k, v

    # optimum-intel's qwen3_5_gated_delta_net_forward with the caches as plain tensors.
    def gated_delta_net(la, x, conv_state, recurrent_state):
        b, t, _ = x.shape
        mixed_qkv = la.in_proj_qkv(x).transpose(1, 2)
        z = la.in_proj_z(x).reshape(b, t, -1, la.head_v_dim)
        beta_in = la.in_proj_b(x)
        a = la.in_proj_a(x)
        mixed_qkv, conv_state = ov_causal_conv1d(conv_state, mixed_qkv, la.conv1d.weight, la.conv1d.bias)
        mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        query = query.reshape(query.shape[0], query.shape[1], -1, la.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, la.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, la.head_v_dim)
        beta = beta_in.sigmoid()
        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
        if la.num_v_heads // la.num_k_heads > 1:
            query = query.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
            key = key.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
        out, recurrent_state = patched_recurrent_gated_delta_rule(
            la, query, key, value, g=g, beta=beta, initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        out = la.norm(out.reshape(-1, la.head_v_dim), z.reshape(-1, la.head_v_dim)).reshape(b, t, -1)
        return la.out_proj(out), conv_state, recurrent_state

    class Fork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = lm

        def forward(self, input_ids, lengths, offset, read_index, slot_ids, *past):
            m = self.lm.model
            b, t = input_ids.shape
            h = m.embed_tokens(input_ids)
            chunk = torch.arange(t, device=input_ids.device)
            cos, sin = m.rotary_emb(h, offset[:, None] + chunk[None])
            first_key = past[tc.layer_types.index(FULL) * 2]
            past_pos = torch.arange(first_key.shape[2], device=input_ids.device)
            visible = torch.cat([past_pos[None] < offset[:, None], chunk[None] < lengths[:, None]], dim=1)
            bias = torch.where(visible, 0.0, torch.finfo(torch.float32).min)[:, None, None, :].to(h.dtype)
            present = []
            for i, (layer, kind) in enumerate(zip(m.layers, tc.layer_types)):
                x = layer.input_layernorm(h)
                if kind == LINEAR:
                    x, s0, s1 = gated_delta_net(layer.linear_attn, x, past[2 * i], past[2 * i + 1])
                else:
                    x, s0, s1 = full_attention(layer.self_attn, x, cos, sin, past[2 * i], past[2 * i + 1], bias)
                present += [s0, s1]
                h = h + x
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            h = m.norm(h)
            h = h.gather(1, read_index[:, None, None].expand(b, 1, h.shape[-1])).squeeze(1)
            logits = self.lm.lm_head(h).float()
            return (logits.index_select(1, slot_ids), torch.logsumexp(logits, dim=-1), *present)

    return Fork().eval()


def convert(lm, out: Path) -> None:
    import openvino as ov
    import torch
    from openvino.frontend.pytorch import ConversionExtension, ModuleExtension
    from openvino.frontend.pytorch import patch_model

    from ov_gdn import RecurrentAttentionCell, convert_recurrent_attention_cell

    tc = lm.config
    specs = past_specs(tc)
    fork = make_fork(lm)
    # Linear and Embedding keep their 16-bit weights as graph constants; activations trace in f32.
    getattr(patch_model, "__make_16bit_traceable")(fork)

    b, t, p = 2, 3, 2
    i64 = torch.int64
    example = [torch.randint(0, tc.vocab_size, (b, t), dtype=i64), torch.tensor([3, 2], dtype=i64),
               torch.tensor([p, p], dtype=i64), torch.tensor([2, 1], dtype=i64), torch.tensor([1, 2, 3], dtype=i64)]
    shapes = [[-1, -1], [-1], [-1], [-1], [-1]]
    names = ["input_ids", "lengths", "offset", "read_index", "slot_ids"]
    for name, shape in specs:
        example.append(torch.zeros([b] + [p if d == -1 else d for d in shape[1:]], dtype=torch.float32))
        shapes.append([-1] + shape[1:])
        names.append(name)
    types = [ov.Type.i64] * 5 + [ov.Type.f32] * len(specs)

    with torch.no_grad():
        model = ov.convert_model(
            fork,
            example_input=tuple(example),
            input=[(ov.PartialShape(s), ty) for s, ty in zip(shapes, types)],
            extension=[ModuleExtension(RecurrentAttentionCell, "RecurrentAttentionCellOp"),
                       ConversionExtension("RecurrentAttentionCellOp", convert_recurrent_attention_cell)],
        )
    if len(model.inputs) != len(names) or len(model.outputs) != 2 + len(specs):
        fail(f"converted graph has {len(model.inputs)} inputs and {len(model.outputs)} outputs")
    for port, name in zip(model.inputs, names):
        port.get_tensor().set_names({name})
    for port, name in zip(model.outputs, ["slot_logits", "lse"] + ["present" + n[len("past"):] for n, _ in specs]):
        port.get_tensor().set_names({name})

    lm_head = [op for op in model.get_ops() if op.get_type_name() == "Constant"
               and list(op.get_output_shape(0)) == [tc.vocab_size, tc.hidden_size]]
    if len(lm_head) != 1:
        fail(f"{len(lm_head)} [vocab, hidden] constants: the tied embedding / LM head was duplicated")
    if lm_head[0].get_element_type() != ov.Type.bf16:
        fail(f"embedding constant is {lm_head[0].get_element_type()}, not bf16")
    out.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, out / "fork.xml", compress_to_fp16=False)
    del model, fork
    log(f"fork.bin: {(out / 'fork.bin').stat().st_size / 2**30:.2f} GiB")


# --- reference ---------------------------------------------------------------------------


def reference(lm, prompts: list[list[int]], prefix_len: int, slot_ids: list[int], n_options: int) -> dict:
    """transformers' single forward of each whole prompt, prefix and suffix as blocks 0 and 1."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    original = mq.create_causal_mask
    blocks = {}

    def create_causal_mask(*args, **kwargs):
        kwargs["block_sequence_ids"] = blocks["ids"]
        return original(*args, **kwargs)

    captured = {}
    hook = lm.model.norm.register_forward_hook(lambda mod, args, out: captured.__setitem__("h", out))
    mq.create_causal_mask = create_causal_mask
    embed = lm.model.embed_tokens.weight
    rows = []
    try:
        with torch.no_grad():
            for ids in prompts:
                x = torch.tensor([ids])
                blocks["ids"] = torch.tensor([[0] * prefix_len + [1] * (len(ids) - prefix_len)])
                lm.model(input_ids=x.to(lm.device), use_cache=False)
                h = captured["h"][0, -1].float()
                slot = embed[slot_ids].float() @ h
                lse = torch.logsumexp(torch.cat([embed[i:i + 32768].float() @ h
                                                 for i in range(0, embed.shape[0], 32768)]), dim=0)
                rows.append({"slot_logits": slot.tolist(), "lse": float(lse)})
    finally:
        mq.create_causal_mask = original
        hook.remove()
    return {"prefix_len": prefix_len, "prompts": prompts, "slot_ids": slot_ids, "n_options": n_options,
            "rows": rows}


def render_check_prompts(tok) -> tuple[list[list[int]], int]:
    """Flide's onejev rendering of CHECK_QUESTIONS (noul) over CHECK_STATE."""
    mark = "\x00DECIDE-SUFFIX\x00"

    def suffix(instructions):
        return (f"Question: {instructions}\n\nOptions:\n"
                "A. yes: the statement is true / the answer is yes\n"
                "B. no: the statement is false / the answer is no\n\n"
                "Answer with one letter: A, B.")

    def render(text):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"<state>\n{CHECK_STATE}\n</state>\n\n{text}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    cut = render(mark).index(mark)
    prompts, prefix_len = [], None
    for q in CHECK_QUESTIONS:
        text = render(suffix(q))
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        n = bisect.bisect_right([e for _, e in enc["offset_mapping"]], cut)
        if prefix_len not in (None, n) or prompts and prompts[0][:n] != enc["input_ids"][:n]:
            fail("check prompts do not share their prefix block")
        prefix_len = n
        prompts.append(list(enc["input_ids"]))
    return prompts, prefix_len


def slot_token_ids(tok) -> list[int]:
    ids = []
    for label in SLOT_LABELS:
        enc = tok.encode(label, add_special_tokens=False)
        if len(enc) != 1 or tok.decode(enc) != label:
            fail(f"slot label {label!r} is not a single round-trip token: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        fail("slot tokens collide")
    return ids


# --- check (OpenVINO side, own process so its peak RSS is its own) -------------------------


def run_check(model_dir: Path, ref_path: Path, tolerance: float, tokenizer: bool) -> None:
    import numpy as np
    import openvino as ov

    ref = json.loads(ref_path.read_text())
    meta = json.loads((model_dir / "decide.json").read_text())
    core = ov.Core()
    if tokenizer:
        import openvino_tokenizers  # noqa: F401  registers the extension with the core

        tok = core.compile_model(model_dir / "openvino_tokenizer.xml", "CPU")
        ids = tok([TOKENIZER_PROBE])["input_ids"].reshape(-1).tolist()
        if ids != TOKENIZER_PROBE_IDS:
            fail(f"tokenizer IR maps the probe to {ids}, want {TOKENIZER_PROBE_IDS}")
        log(f"tokenizer IR: probe -> {ids}")

    model = core.read_model(model_dir / "fork.xml")
    compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    want = sum(1 for p in meta["past"] if p["name"].endswith(".recurrent"))
    fused = sum(1 for op in compiled.get_runtime_model().get_ordered_ops()
                if op.get_rt_info()["layerType"].astype(str) == "GatedDeltaNet")
    if fused != want:
        fail(f"{fused} GatedDeltaNet nodes in the compiled model, want {want} (one per linear-attention layer)")
    log(f"GatedDeltaNet fusion: {fused}/{want}")

    slot_ids = np.array(ref["slot_ids"], dtype=np.int64)
    p = ref["prefix_len"]
    prompts = ref["prompts"]
    req = compiled.create_infer_request()

    feed = {"input_ids": np.array([prompts[0][:p]], dtype=np.int64), "lengths": np.array([p], dtype=np.int64),
            "offset": np.zeros(1, dtype=np.int64), "read_index": np.array([p - 1], dtype=np.int64),
            "slot_ids": slot_ids}
    for spec in meta["past"]:
        feed[spec["name"]] = np.zeros([0 if d == -1 else d for d in spec["shape"]], dtype=np.float32)
    res = req.infer(feed)
    prefix_cache = {spec["name"]: res["present" + spec["name"][len("past"):]].copy() for spec in meta["past"]}

    suffixes = [ids[p:] for ids in prompts]
    b, t = len(suffixes), max(len(s) for s in suffixes)
    feed = {"input_ids": np.array([s + [0] * (t - len(s)) for s in suffixes], dtype=np.int64),
            "lengths": np.array([len(s) for s in suffixes], dtype=np.int64),
            "offset": np.full(b, p, dtype=np.int64),
            "read_index": np.array([len(s) - 1 for s in suffixes], dtype=np.int64),
            "slot_ids": slot_ids}
    for name, cache in prefix_cache.items():
        feed[name] = np.repeat(cache, b, axis=0)
    res = req.infer(feed)
    got_slots, got_lse = res["slot_logits"], res["lse"]

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    n = ref["n_options"]
    worst_dp, worst_logit, worst_lse = 0.0, 0.0, 0.0
    for i, row in enumerate(ref["rows"]):
        want_logits = np.array(row["slot_logits"])
        pw, pg = softmax(want_logits[:n]), softmax(got_slots[i, :n].astype(np.float64))
        log(f"question {i}: torch p={np.round(pw, 4).tolist()} openvino p={np.round(pg, 4).tolist()}")
        if pw.argmax() != pg.argmax():
            fail(f"question {i}: top slot {pg.argmax()} != reference {pw.argmax()}")
        worst_dp = max(worst_dp, float(np.abs(pw - pg).max()))
        worst_logit = max(worst_logit, float(np.abs(want_logits - got_slots[i]).max()))
        worst_lse = max(worst_lse, abs(row["lse"] - float(got_lse[i])))
    log(f"fork exactness: max |dp| {worst_dp:.3e} (tolerance {tolerance:.0e}), "
        f"max |dlogit| {worst_logit:.3e} over {len(slot_ids)} slots, max |dlse| {worst_lse:.3e}")
    if not np.isfinite(got_slots).all() or not np.isfinite(got_lse).all():
        fail("non-finite outputs")
    if worst_dp > tolerance:
        fail(f"max |dp| {worst_dp:.3e} exceeds {tolerance:.0e}")
    log_peak("self-check")


def spawn_check(model_dir: Path, ref: dict, tolerance: float, tokenizer: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "reference.json"
        ref_path.write_text(json.dumps(ref))
        cmd = [sys.executable, os.path.abspath(__file__), "check", "--model", str(model_dir), "--reference",
               str(ref_path), "--tolerance", str(tolerance)]
        if not tokenizer:
            cmd.append("--no-tokenizer")
        if subprocess.run(cmd).returncode != 0:
            fail("self-check failed")


# --- commands ----------------------------------------------------------------------------


def cmd_export(src: Path, out: Path) -> None:
    from transformers import AutoTokenizer

    config = validate_source(src)
    tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    slot_ids = slot_token_ids(tok)
    prompts, prefix_len = render_check_prompts(tok)
    log(f"check prompts: prefix {prefix_len} tokens, suffixes {[len(p) - prefix_len for p in prompts]}")

    out.mkdir(parents=True, exist_ok=True)
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer

    ov.save_model(convert_tokenizer(tok, with_detokenizer=False), out / "openvino_tokenizer.xml")
    del tok

    lm = load_text_model(src, config)
    ref = reference(lm, prompts, prefix_len, slot_ids, 2)
    convert(lm, out)
    write_decide_json(out, lm.config, slot_ids)
    del lm
    gc.collect()
    log_peak("export")

    for name in METADATA:
        shutil.move(src / name, out / name)
    spawn_check(out, ref, REAL_TOLERANCE, tokenizer=True)


def write_decide_json(out: Path, tc, slot_ids: list[int]) -> None:
    meta = {"format": FORMAT, "slot_ids": slot_ids,
            "past": [{"name": n, "shape": s} for n, s in past_specs(tc)]}
    (out / "decide.json").write_text(json.dumps(meta) + "\n")


def cmd_selftest() -> None:
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    # Two groups: with one, the only full-attention layer is the last and the read token
    # cannot tell a block-bidirectional mask from a causal one.
    torch.manual_seed(0)
    tc = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        layer_types=[LINEAR, LINEAR, LINEAR, FULL] * 2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, tie_word_embeddings=True, initializer_range=0.2,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    lm = mq.Qwen3_5ForCausalLM(tc).to(torch.bfloat16)
    with torch.no_grad():
        for name, param in lm.named_parameters():
            if name.endswith("norm.weight"):
                param.normal_(0, 0.3)
    lm = finish_model(lm)
    slot_ids = list(range(32, 32 + 26))
    gen = torch.Generator().manual_seed(1)
    prefix = torch.randint(0, tc.vocab_size, (37,), generator=gen).tolist()
    prompts = [prefix + torch.randint(0, tc.vocab_size, (n,), generator=gen).tolist() for n in (11, 6)]

    # f32 compute over the same bf16-rounded weights isolates graph errors from bf16 rounding.
    ref = reference(lm.float(), prompts, len(prefix), slot_ids, 4)
    lm = lm.to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        convert(lm, out)
        write_decide_json(out, lm.config, slot_ids)
        del lm
        gc.collect()
        spawn_check(out, ref, SELFTEST_TOLERANCE, tokenizer=False)
    log("selftest ok")


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(prog="decide-export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"decide-export {__version__} (format {FORMAT})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="export a HF checkpoint directory to a decide model directory")
    ex.add_argument("--src", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    sub.add_parser("selftest", help="export a tiny random model and check it against transformers")
    ck = sub.add_parser("check")
    ck.add_argument("--model", type=Path, required=True)
    ck.add_argument("--reference", type=Path, required=True)
    ck.add_argument("--tolerance", type=float, required=True)
    ck.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()
    if args.cmd == "export":
        cmd_export(args.src, args.out)
    elif args.cmd == "selftest":
        cmd_selftest()
    else:
        run_check(args.model, args.reference, args.tolerance, not args.no_tokenizer)


if __name__ == "__main__":
    main()
