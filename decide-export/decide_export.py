"""decide-export: export a Base Labs decide model (Qwen3.5 backbone, block-bidirectional
full attention, slot_lm readout) to an OpenVINO fork graph and vision encoder.

Decide model format 2
=====================

A model directory holds fork.xml/fork.bin, vision.xml/vision.bin, openvino_tokenizer.xml/.bin,
decide.json and the HF metadata (config.json, tokenizer.json, tokenizer_config.json,
chat_template.jinja, readout.json, generation_config.json, processor_config.json).

fork.xml. B = rows, T = chunk length, P = past length, K = slots, N = image tokens in the
chunk, H = hidden size; all dynamic but H.

Inputs (integers are i64, floats f32):
  input_ids    [B,T]    token ids, right padded
  lengths      [B]      valid tokens per row
  offset       [B]      cache index of the chunk's first token; past key j is visible to
                        row b iff j < offset[b]
  position_ids [3,B,T]  multimodal RoPE positions (temporal, height, width), see Positions
  image_embeds [N,H]    vision.xml rows that replace the embeddings of image tokens
  image_index  [N]      b*T+t of the chunk token each image_embeds row replaces
  read_index   [B]      chunk index whose final-normed hidden state is read
  slot_ids     [K]      vocabulary ids of the answer slots
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
batch must equal B. A text-only chunk passes N=0 image tensors.

vision.xml. L = patches, I = images; both dynamic.
  pixel_values   [L,C*Tp*Ps*Ps] f32  the patches of every image, image after image
  image_grid_thw [I,3]          i64  (t, gh, gw) patch grid of each image
  image_embeds   [L/M^2,H]      f32  one row per image token, image after image
Images attend only within themselves, so one call may hold any number of images.

Images (decide.json "vision", Qwen2-VL preprocessing). Ps = patch_size, Tp =
temporal_patch_size, M = merge_size. Resize an RGB image (resample filter) to height h
and width w, multiples of Ps*M, with min_pixels <= h*w <= max_pixels and the aspect
ratio kept as closely as possible (transformers' smart_resize). Scale by rescale_factor,
then normalize channel c as (x - image_mean[c]) / image_std[c]. The grid is
(t, gh, gw) = (1, h/Ps, w/Ps). Stack the [C,h,w] frame Tp times to [Tp,C,h,w], reshape to
[t, Tp, C, gh/M, M, Ps, gw/M, M, Ps], transpose to (t, gh/M, gw/M, M, M, C, Tp, Ps, Ps)
and flatten to [t*gh*gw, C*Tp*Ps*Ps].

Prompts. The chat template renders an image as vision_start_token_id, image_token_id,
vision_end_token_id; repeat image_token_id to t*gh*gw/M^2 tokens, one per image_embeds row.

Positions. Walk the whole prompt from p = 0. A text token gets (p, p, p) and p += 1. The
tokens of an image with merged grid (mh, mw) = (gh/M, gw/M) get (p, p+i, p+j) for row i,
column j, row-major, then p += max(mh, mw). A text-only prompt so has position_ids =
offset + chunk index, as in format 1; after an image the positions run behind the cache
index that offset counts.

Forking: call once with the shared prefix (B=1, P=0 caches), tile its present.* to B rows,
then call once with the B question suffixes, offset = prefix length, position_ids
continuing from the prefix's final p and read_index = lengths-1. The answer distribution
is softmax(slot_logits[:, :n_options]).

decide.json: {"format": 2, "slot_ids": [...256 ids, slot label order...],
              "past": [{"name": "past.0.conv", "shape": [1, C, W]}, ...],
              "hidden_size": H, "image_token_id": ..., "vision_start_token_id": ...,
              "vision_end_token_id": ...,
              "weights": "bf16" or "int8",
              "vision": {"patch_size": Ps, "temporal_patch_size": Tp, "merge_size": M,
                         "min_pixels": ..., "max_pixels": ..., "resample": "bicubic",
                         "rescale_factor": ..., "image_mean": [...], "image_std": [...]}}
with B=1 and -1 only for P in past.
weights is the storage type of the fork and vision weights. int8 is NNCF's INT8_ASYM
weight-only compression of a bf16 export (decide-export compress); activations keep the
runtime's inference precision.
"""
from __future__ import annotations

import argparse
import bisect
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

__version__ = "2.1.0"
FORMAT = 2

LINEAR, FULL = "linear_attention", "full_attention"
METADATA = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "readout.json", "generation_config.json", "processor_config.json")
# PIL's resampling filter enum, as processor_config.json stores it.
RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}

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
# 256x320 pixels: within joey's min_pixels, 80 image tokens.
CHECK_IMAGE = {"height": 256, "width": 320, "seed": 0}
# bf16 torch reference vs f32 OpenVINO: random weights at joey dims gave 3e-3.
REAL_TOLERANCE = 2e-2
# Selftest compares against an f32 torch reference of the same bf16 weights.
SELFTEST_TOLERANCE = 1e-3
# int8 is for CI only: its check requires the reference's top slot, not a |dp| bound.
INT8_TOLERANCE = float("inf")


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
    if config.get("vision_config", {}).get("deepstack_visual_indexes"):
        fail("deepstack vision features are not supported")
    return config


def vision_params(processor: dict) -> dict:
    ip = processor["image_processor"]
    if ip.get("image_processor_type") != "Qwen2VLImageProcessor":
        fail(f"image processor {ip.get('image_processor_type')!r} is not Qwen2VLImageProcessor")
    if not (ip["do_convert_rgb"] and ip["do_rescale"] and ip["do_normalize"] and ip["do_resize"]):
        fail("image processor skips a convert/rescale/normalize/resize step")
    return {"patch_size": ip["patch_size"], "temporal_patch_size": ip["temporal_patch_size"],
            "merge_size": ip["merge_size"], "min_pixels": ip["size"]["shortest_edge"],
            "max_pixels": ip["size"]["longest_edge"], "resample": RESAMPLE[ip["resample"]],
            "rescale_factor": ip["rescale_factor"], "image_mean": ip["image_mean"], "image_std": ip["image_std"]}


def load_model(src: Path):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    with torch.device("meta"):
        model = mq.Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_pretrained(src))
    state = {}
    shards = sorted(src.glob("model*.safetensors"))
    if not shards:
        fail(f"no safetensors in {src}")
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                # MTP (mtp.*) is not part of the model.
                if key.startswith("model."):
                    state[key] = f.get_tensor(key).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    if set(missing) - {"lm_head.weight"} or unexpected:
        fail(f"weights do not match the config: missing {missing}, unexpected {unexpected}")
    return finish_model(model)


def finish_model(model):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    tc = model.config.text_config
    if not tc.tie_word_embeddings:
        fail("untied LM head is not supported")
    text = model.model.language_model
    model.lm_head.weight = text.embed_tokens.weight
    # Non-persistent buffers are still on the meta device after a meta-device load, and a
    # dtype cast would round them.
    text.rotary_emb = mq.Qwen3_5TextRotaryEmbedding(tc)
    model.model.visual.rotary_pos_emb = mq.Qwen3_5VisionRotaryEmbedding(model.config.vision_config)
    model.config._attn_implementation = "sdpa"
    tc._attn_implementation = "sdpa"
    model.config.vision_config._attn_implementation = "sdpa"
    return model.eval()


def text_lm(model):
    """model's language model and LM head as a Qwen3_5ForCausalLM, sharing the weights."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    with torch.device("meta"):
        lm = mq.Qwen3_5ForCausalLM(model.config.text_config)
    lm.model = model.model.language_model
    lm.lm_head = model.lm_head
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

        def forward(self, input_ids, lengths, offset, position_ids, image_embeds, image_index, read_index, slot_ids,
                    *past):
            m = self.lm.model
            b, t = input_ids.shape
            h = m.embed_tokens(input_ids)
            h = h.reshape(b * t, -1).index_copy(0, image_index, image_embeds.to(h.dtype)).reshape(b, t, -1)
            chunk = torch.arange(t, device=input_ids.device)
            cos, sin = m.rotary_emb(h, position_ids)
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


def make_vision(visual):
    """Qwen3_5VisionModel.forward with the grid-derived tensors computed in-graph and the
    per-image attention as one masked attention, so image_grid_thw stays a runtime input."""
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    def attention(at, x, cos, sin, bias):
        n = x.shape[0]
        q, k, v = at.qkv(x).reshape(n, 3, at.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = (s.transpose(0, 1)[None] for s in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=at.scaling)
        return at.proj(o[0].transpose(0, 1).reshape(n, -1))

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = visual

        def forward(self, pixel_values, image_grid_thw):
            v = self.visual
            m = v.spatial_merge_size
            # Patch coordinates in spatial-merge-block order, as vision_utils derives them.
            counts = image_grid_thw.prod(-1)
            frame_len = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], counts)
            blocks_w = torch.repeat_interleave(image_grid_thw[:, 2], counts) // m
            starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
            index = torch.arange(counts.sum(), device=pixel_values.device)
            within = (index - starts) % frame_len
            row = within // (m * m * blocks_w) * m + (within // m) % m
            col = (within // (m * m)) % blocks_w * m + within % m
            frame = index - within
            bias = torch.where(frame[:, None] == frame[None, :], 0.0, torch.finfo(torch.float32).min)

            interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
                image_grid_thw, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
                align_corners=v.interpolation_align_corners, spatial_merge_size=m)
            proj = v.patch_embed.proj
            # The patch Conv3d has kernel == stride == the patch: a linear map of the flat patch.
            h = F.linear(pixel_values, proj.weight.reshape(proj.weight.shape[0], -1), proj.bias)
            h = h + (v.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
            cos, sin = v.rotary_pos_emb(h, torch.stack([row, col], dim=-1))
            for blk in v.blocks:
                h = h + attention(blk.attn, blk.norm1(h), cos, sin, bias)
                h = h + blk.mlp(blk.norm2(h))
            return v.merger(h)

    return Vision().eval()


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
    i64, f32 = torch.int64, torch.float32
    example = [torch.randint(0, tc.vocab_size, (b, t), dtype=i64), torch.tensor([3, 2], dtype=i64),
               torch.tensor([p, p], dtype=i64), torch.arange(p, p + t, dtype=i64).expand(3, b, t),
               torch.randn(2, tc.hidden_size, dtype=f32), torch.tensor([1, 4], dtype=i64),
               torch.tensor([2, 1], dtype=i64), torch.tensor([1, 2, 3], dtype=i64)]
    shapes = [[-1, -1], [-1], [-1], [3, -1, -1], [-1, tc.hidden_size], [-1], [-1], [-1]]
    types = [ov.Type.i64] * 4 + [ov.Type.f32] + [ov.Type.i64] * 3
    names = ["input_ids", "lengths", "offset", "position_ids", "image_embeds", "image_index", "read_index", "slot_ids"]
    for name, shape in specs:
        example.append(torch.zeros([b] + [p if d == -1 else d for d in shape[1:]], dtype=f32))
        shapes.append([-1] + shape[1:])
        types.append(ov.Type.f32)
        names.append(name)

    with torch.no_grad():
        model = ov.convert_model(
            fork,
            example_input=tuple(example),
            input=[(ov.PartialShape(s), ty) for s, ty in zip(shapes, types)],
            extension=[ModuleExtension(RecurrentAttentionCell, "RecurrentAttentionCellOp"),
                       ConversionExtension("RecurrentAttentionCellOp", convert_recurrent_attention_cell)],
        )
    name_ports(model, names, ["slot_logits", "lse"] + ["present" + n[len("past"):] for n, _ in specs])

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


def convert_vision(visual, out: Path) -> None:
    import openvino as ov
    import torch
    from openvino.frontend.pytorch import patch_model

    vc = visual.config
    vision = make_vision(visual)
    getattr(patch_model, "__make_16bit_traceable")(vision)
    m = vc.spatial_merge_size
    # Two images of different shapes, so nothing grid-dependent is traced as a constant.
    grid = torch.tensor([[1, 2 * m, 3 * m], [1, m, m]], dtype=torch.int64)
    patch = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
    example = (torch.randn(int(grid.prod(-1).sum()), patch), grid)
    with torch.no_grad():
        model = ov.convert_model(vision, example_input=example,
                                 input=[(ov.PartialShape([-1, patch]), ov.Type.f32),
                                        (ov.PartialShape([-1, 3]), ov.Type.i64)])
    name_ports(model, ["pixel_values", "image_grid_thw"], ["image_embeds"])
    ov.save_model(model, out / "vision.xml", compress_to_fp16=False)
    del model, vision
    log(f"vision.bin: {(out / 'vision.bin').stat().st_size / 2**30:.2f} GiB")


def name_ports(model, inputs: list[str], outputs: list[str]) -> None:
    if len(model.inputs) != len(inputs) or len(model.outputs) != len(outputs):
        fail(f"converted graph has {len(model.inputs)} inputs and {len(model.outputs)} outputs, "
             f"want {len(inputs)} and {len(outputs)}")
    for port, name in zip([*model.inputs, *model.outputs], inputs + outputs):
        port.get_tensor().set_names({name})
    # The OpenVINO C API rejects rank-0 tensors, so no port of the contract may be a scalar.
    scalars = [p.get_any_name() for p in [*model.inputs, *model.outputs] if p.get_partial_shape().rank.get_length() == 0]
    if scalars:
        fail(f"rank-0 ports, unusable from the C API: {scalars}")


# --- images ------------------------------------------------------------------------------


def check_image(spec: dict, vp: dict):
    """A deterministic synthetic RGB image, preprocessed as the module docstring describes.
    Its size is already valid, so there is no resize."""
    import numpy as np

    h, w = spec["height"], spec["width"]
    ps, tp, m = vp["patch_size"], vp["temporal_patch_size"], vp["merge_size"]
    if h % (ps * m) or w % (ps * m) or not vp["min_pixels"] <= h * w <= vp["max_pixels"]:
        fail(f"check image {h}x{w} needs a resize")
    y, x = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(spec["seed"])
    rgb = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x // 8 + y // 8) % 2) * 255], axis=-1)
    rgb = np.clip(rgb + rng.integers(-20, 21, rgb.shape), 0, 255).astype(np.uint8)

    frame = rgb.astype(np.float32).transpose(2, 0, 1) * np.float32(vp["rescale_factor"])
    frame = (frame - np.array(vp["image_mean"], np.float32)[:, None, None]) / np.array(vp["image_std"], np.float32)[:, None, None]
    c, gh, gw = frame.shape[0], h // ps, w // ps
    patches = np.repeat(frame[None], tp, axis=0).reshape(1, tp, c, gh // m, m, ps, gw // m, m, ps)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, c * tp * ps * ps)
    return np.ascontiguousarray(patches, dtype=np.float32), [1, gh, gw]


def expand_images(ids: list[int], image_token_id: int, n_tokens: list[int]) -> list[int]:
    """Repeat each image token to its image's token count, images in order."""
    counts = iter(n_tokens)
    out = []
    for i in ids:
        out += [i] * next(counts) if i == image_token_id else [i]
    if next(counts, None) is not None:
        fail("more images than image tokens")
    return out


def rope_positions(ids: list[int], grids: list[list[int]], image_token_id: int, merge: int) -> list[list[int]]:
    """position_ids [3, len(ids)] of a whole prompt, as the module docstring describes."""
    pos, p, i, images = [[], [], []], 0, 0, iter(grids)
    while i < len(ids):
        if ids[i] != image_token_id:
            for axis in pos:
                axis.append(p)
            p, i = p + 1, i + 1
            continue
        t, gh, gw = next(images)
        mh, mw = gh // merge, gw // merge
        for r in range(t * mh * mw):
            pos[0].append(p + r // (mh * mw))
            pos[1].append(p + r // mw % mh)
            pos[2].append(p + r % mw)
        if ids[i:i + t * mh * mw] != [image_token_id] * (t * mh * mw):
            fail("image token run does not match its grid")
        p, i = p + max(mh, mw), i + t * mh * mw
    return pos


# --- reference ---------------------------------------------------------------------------


def reference(model, case: dict, vp: dict) -> list[dict]:
    """transformers' single forward of each whole prompt, prefix and suffix as blocks 0 and 1."""
    import numpy as np
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    original = mq.create_causal_mask
    blocks = {}

    def create_causal_mask(*args, **kwargs):
        kwargs["block_sequence_ids"] = blocks["ids"]
        return original(*args, **kwargs)

    text = model.model.language_model
    captured = {}
    hook = text.norm.register_forward_hook(lambda mod, args, out: captured.__setitem__("h", out))
    mq.create_causal_mask = create_causal_mask
    embed = text.embed_tokens.weight
    image_token_id = model.config.image_token_id
    prefix_len, slot_ids = case["prefix_len"], case["slot_ids"]
    rows = []
    try:
        with torch.no_grad():
            for ids, specs in zip(case["prompts"], case["images"]):
                x = torch.tensor([ids])
                blocks["ids"] = torch.tensor([[0] * prefix_len + [1] * (len(ids) - prefix_len)])
                kwargs = {}
                if specs:
                    images = [check_image(s, vp) for s in specs]
                    kwargs = {"pixel_values": torch.from_numpy(np.concatenate([px for px, _ in images])),
                              "image_grid_thw": torch.tensor([g for _, g in images]),
                              "mm_token_type_ids": (x == image_token_id).int()}
                model.model.rope_deltas = None
                model.model(input_ids=x, use_cache=False, **kwargs)
                h = captured["h"][0, -1].float()
                slot = embed[slot_ids].float() @ h
                lse = torch.logsumexp(torch.cat([embed[i:i + 32768].float() @ h
                                                 for i in range(0, embed.shape[0], 32768)]), dim=0)
                rows.append({"slot_logits": slot.tolist(), "lse": float(lse)})
    finally:
        mq.create_causal_mask = original
        hook.remove()
    return rows


def render_check_prompts(tok, image: bool) -> tuple[list[list[int]], int]:
    """Flide's onejev rendering of CHECK_QUESTIONS (noul) over CHECK_STATE, with one image
    token after the state text if image."""
    mark = "\x00DECIDE-SUFFIX\x00"

    def suffix(instructions):
        return (f"Question: {instructions}\n\nOptions:\n"
                "A. yes: the statement is true / the answer is yes\n"
                "B. no: the statement is false / the answer is no\n\n"
                "Answer with one letter: A, B.")

    def render(text):
        if image:
            content = [{"type": "text", "text": f"<state>\n{CHECK_STATE}\n"}, {"type": "image"},
                       {"type": "text", "text": f"\n</state>\n\n{text}"}]
        else:
            content = f"<state>\n{CHECK_STATE}\n</state>\n\n{text}"
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
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

    cases = json.loads(ref_path.read_text())
    meta = json.loads((model_dir / "decide.json").read_text())
    if tokenizer:
        # Importing it registers the tokenizer ops with Cores created afterwards.
        import openvino_tokenizers  # noqa: F401
    core = ov.Core()
    if tokenizer:
        tok = core.compile_model(model_dir / "openvino_tokenizer.xml", "CPU")
        ids = tok([TOKENIZER_PROBE])["input_ids"].reshape(-1).tolist()
        if ids != TOKENIZER_PROBE_IDS:
            fail(f"tokenizer IR maps the probe to {ids}, want {TOKENIZER_PROBE_IDS}")
        log(f"tokenizer IR: probe -> {ids}")

    f32 = {"INFERENCE_PRECISION_HINT": "f32"}
    compiled = core.compile_model(core.read_model(model_dir / "fork.xml"), "CPU", f32)
    want = sum(1 for p in meta["past"] if p["name"].endswith(".recurrent"))
    fused = sum(1 for op in compiled.get_runtime_model().get_ordered_ops()
                if op.get_rt_info()["layerType"].astype(str) == "GatedDeltaNet")
    if fused != want:
        fail(f"{fused} GatedDeltaNet nodes in the compiled model, want {want} (one per linear-attention layer)")
    log(f"GatedDeltaNet fusion: {fused}/{want}")
    vision = core.compile_model(core.read_model(model_dir / "vision.xml"), "CPU", f32).create_infer_request()
    req = compiled.create_infer_request()
    image_token_id, hidden, vp = meta["image_token_id"], meta["hidden_size"], meta["vision"]

    def image_feed(rows: list[list[int]], specs: list[dict], t: int) -> dict:
        """image_embeds/image_index of a chunk whose rows hold specs' images in order."""
        index = [b * t + i for b, ids in enumerate(rows) for i, tid in enumerate(ids) if tid == image_token_id]
        embeds = np.zeros((0, hidden), np.float32)
        if specs:
            images = [check_image(s, vp) for s in specs]
            res = vision.infer({"pixel_values": np.concatenate([px for px, _ in images]),
                                "image_grid_thw": np.array([g for _, g in images], np.int64)})
            embeds = res["image_embeds"]
        if embeds.shape != (len(index), hidden):
            fail(f"image_embeds {embeds.shape} for {len(index)} image tokens")
        return {"image_embeds": embeds, "image_index": np.array(index, np.int64)}

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    for case in cases:
        slot_ids = np.array(case["slot_ids"], dtype=np.int64)
        p, prompts, images = case["prefix_len"], case["prompts"], case["images"]
        positions = []
        for ids, specs in zip(prompts, images):
            grids = [check_image(s, vp)[1] for s in specs]
            positions.append(np.array(rope_positions(ids, grids, image_token_id, vp["merge_size"]), np.int64))
        # Prefix images are the leading images of every prompt.
        n_prefix = len(np.flatnonzero(np.diff(np.r_[0, np.array(prompts[0][:p]) == image_token_id]) == 1))

        feed = {"input_ids": np.array([prompts[0][:p]], dtype=np.int64), "lengths": np.array([p], dtype=np.int64),
                "offset": np.zeros(1, dtype=np.int64), "position_ids": positions[0][:, None, :p],
                "read_index": np.array([p - 1], dtype=np.int64), "slot_ids": slot_ids,
                **image_feed([prompts[0][:p]], images[0][:n_prefix], p)}
        for spec in meta["past"]:
            feed[spec["name"]] = np.zeros([0 if d == -1 else d for d in spec["shape"]], dtype=np.float32)
        res = req.infer(feed)
        prefix_cache = {spec["name"]: res["present" + spec["name"][len("past"):]].copy() for spec in meta["past"]}

        suffixes = [ids[p:] for ids in prompts]
        b, t = len(suffixes), max(len(s) for s in suffixes)
        pos = np.zeros((3, b, t), np.int64)
        for i, s in enumerate(suffixes):
            pos[:, i, :len(s)] = positions[i][:, p:]
        feed = {"input_ids": np.array([s + [0] * (t - len(s)) for s in suffixes], dtype=np.int64),
                "lengths": np.array([len(s) for s in suffixes], dtype=np.int64),
                "offset": np.full(b, p, dtype=np.int64), "position_ids": pos,
                "read_index": np.array([len(s) - 1 for s in suffixes], dtype=np.int64),
                "slot_ids": slot_ids,
                **image_feed(suffixes, [s for specs in images for s in specs[n_prefix:]], t)}
        for name, cache in prefix_cache.items():
            feed[name] = np.repeat(cache, b, axis=0)
        res = req.infer(feed)
        got_slots, got_lse = res["slot_logits"], res["lse"]

        n = case["n_options"]
        worst_dp, worst_logit, worst_lse = 0.0, 0.0, 0.0
        for i, row in enumerate(case["rows"]):
            want_logits = np.array(row["slot_logits"])
            pw, pg = softmax(want_logits[:n]), softmax(got_slots[i, :n].astype(np.float64))
            log(f"{case['name']} question {i}: torch p={np.round(pw, 4).tolist()} openvino p={np.round(pg, 4).tolist()}")
            if pw.argmax() != pg.argmax():
                fail(f"{case['name']} question {i}: top slot {pg.argmax()} != reference {pw.argmax()}")
            worst_dp = max(worst_dp, float(np.abs(pw - pg).max()))
            worst_logit = max(worst_logit, float(np.abs(want_logits - got_slots[i]).max()))
            worst_lse = max(worst_lse, abs(row["lse"] - float(got_lse[i])))
        log(f"{case['name']} fork exactness: max |dp| {worst_dp:.3e} (tolerance {tolerance:.0e}), "
            f"max |dlogit| {worst_logit:.3e} over {len(slot_ids)} slots, max |dlse| {worst_lse:.3e}")
        if not np.isfinite(got_slots).all() or not np.isfinite(got_lse).all():
            fail(f"{case['name']}: non-finite outputs")
        if worst_dp > tolerance:
            fail(f"{case['name']}: max |dp| {worst_dp:.3e} exceeds {tolerance:.0e}")
    log_peak("self-check")


def spawn_check(model_dir: Path, cases: list[dict], tolerance: float, tokenizer: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "reference.json"
        ref_path.write_text(json.dumps(cases))
        cmd = [sys.executable, os.path.abspath(__file__), "check", "--model", str(model_dir), "--reference",
               str(ref_path), "--tolerance", str(tolerance)]
        if not tokenizer:
            cmd.append("--no-tokenizer")
        if subprocess.run(cmd).returncode != 0:
            fail("self-check failed")


# --- commands ----------------------------------------------------------------------------


def export(model, out: Path, slot_ids: list[int], vp: dict) -> None:
    lm = text_lm(model)
    convert(lm, out)
    convert_vision(model.model.visual, out)
    tc = model.config.text_config
    meta = {"format": FORMAT, "slot_ids": slot_ids, "past": [{"name": n, "shape": s} for n, s in past_specs(tc)],
            "hidden_size": tc.hidden_size, "image_token_id": model.config.image_token_id,
            "vision_start_token_id": model.config.vision_start_token_id,
            "vision_end_token_id": model.config.vision_end_token_id, "vision": vp, "weights": "bf16"}
    (out / "decide.json").write_text(json.dumps(meta) + "\n")


def compress(src: Path, out: Path) -> None:
    """src's bf16 model directory with int8 fork and vision weights, into out."""
    import nncf
    import openvino as ov

    meta = json.loads((src / "decide.json").read_text())
    if meta.get("weights") != "bf16":
        fail(f"{src} has {meta.get('weights')!r} weights, not bf16")
    graphs = ("fork", "vision")
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.stem not in graphs and f.name != "decide.json":
            shutil.copy2(f, out / f.name)
    core = ov.Core()
    for name in graphs:
        model = nncf.compress_weights(core.read_model(src / f"{name}.xml"), mode=nncf.CompressWeightsMode.INT8_ASYM)
        ov.save_model(model, out / f"{name}.xml", compress_to_fp16=False)
        del model
        gc.collect()
        log(f"{name}.bin: {(out / f'{name}.bin').stat().st_size / 2**30:.2f} GiB int8")
    meta["weights"] = "int8"
    (out / "decide.json").write_text(json.dumps(meta) + "\n")


def cmd_export(src: Path, out: Path, reference_out: Path | None) -> None:
    from transformers import AutoTokenizer

    config = validate_source(src)
    vp = vision_params(json.loads((src / "processor_config.json").read_text()))
    tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    slot_ids = slot_token_ids(tok)
    m = vp["merge_size"]
    _, grid = check_image(CHECK_IMAGE, vp)
    n_image = grid[0] * grid[1] * grid[2] // (m * m)
    cases = []
    for name, image in (("text", False), ("image", True)):
        prompts, prefix_len = render_check_prompts(tok, image)
        if image:
            prompts = [expand_images(ids, config["image_token_id"], [n_image]) for ids in prompts]
            prefix_len += n_image - 1
        cases.append({"name": name, "prompts": prompts, "prefix_len": prefix_len, "slot_ids": slot_ids,
                      "n_options": 2, "images": [[CHECK_IMAGE] if image else [] for _ in prompts]})
        log(f"{name} check prompts: prefix {prefix_len} tokens, suffixes {[len(p) - prefix_len for p in prompts]}")

    out.mkdir(parents=True, exist_ok=True)
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer

    ov.save_model(convert_tokenizer(tok, with_detokenizer=False), out / "openvino_tokenizer.xml")
    del tok

    model = load_model(src)
    for case in cases:
        case["rows"] = reference(model, case, vp)
    export(model, out, slot_ids, vp)
    del model
    gc.collect()
    log_peak("export")

    for name in METADATA:
        shutil.move(src / name, out / name)
    spawn_check(out, cases, REAL_TOLERANCE, tokenizer=True)
    if reference_out:
        reference_out.write_text(json.dumps(cases))


def cmd_compress(src: Path, out: Path, reference: Path) -> None:
    compress(src, out)
    log_peak("compress")
    spawn_check(out, json.loads(reference.read_text()), INT8_TOLERANCE, tokenizer=True)


def cmd_selftest() -> None:
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    # Two groups: with one, the only full-attention layer is the last and the read token
    # cannot tell a block-bidirectional mask from a causal one.
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 512, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 8,
            "layer_types": [LINEAR, LINEAR, LINEAR, FULL] * 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
            "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "tie_word_embeddings": True,
            "initializer_range": 0.2,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                "mrope_section": [2, 1, 1], "mrope_interleaved": True}},
        vision_config={
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "in_channels": 3,
            "patch_size": 4, "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": 16,
            "out_hidden_size": 64, "hidden_act": "gelu_pytorch_tanh", "initializer_range": 0.2,
            "deepstack_visual_indexes": []},
        image_token_id=500, video_token_id=501, vision_start_token_id=502, vision_end_token_id=503,
        tie_word_embeddings=True)
    model = mq.Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.normal_(0, 0.3)
    model = finish_model(model)
    vp = {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2, "min_pixels": 64, "max_pixels": 4096,
          "resample": "bicubic", "rescale_factor": 1 / 255, "image_mean": [0.5, 0.4, 0.3], "image_std": [0.2, 0.3, 0.4]}
    slot_ids = list(range(32, 32 + 26))
    gen = torch.Generator().manual_seed(1)

    def text(n):
        return torch.randint(0, 480, (n,), generator=gen).tolist()

    def image(spec):
        _, (t, gh, gw) = check_image(spec, vp)
        return [502] + [500] * (t * gh * gw // 4) + [503]

    # Two prefix images of different shapes; question 0 also has one in its suffix.
    wide, tall, small = {"height": 16, "width": 24, "seed": 1}, {"height": 24, "width": 16, "seed": 2}, \
        {"height": 8, "width": 8, "seed": 3}
    text_prefix = text(37)
    prefix = text(9) + image(wide) + text(5) + image(tall) + text(7)
    cases = [
        {"name": "text", "prompts": [text_prefix + text(n) for n in (11, 6)], "prefix_len": len(text_prefix),
         "images": [[], []]},
        {"name": "image", "prompts": [prefix + text(4) + image(small) + text(3), prefix + text(6)],
         "prefix_len": len(prefix), "images": [[wide, tall, small], [wide, tall]]},
    ]

    # f32 compute over the same bf16-rounded weights isolates graph errors from bf16 rounding.
    reference_model = finish_model(model.float())
    for case in cases:
        case.update(slot_ids=slot_ids, n_options=4)
        case["rows"] = reference(reference_model, case, vp)
    model = finish_model(model.to(torch.bfloat16))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        export(model, out / "bf16", slot_ids, vp)
        del model, reference_model
        gc.collect()
        spawn_check(out / "bf16", cases, SELFTEST_TOLERANCE, tokenizer=False)
        compress(out / "bf16", out / "int8")
        spawn_check(out / "int8", cases, INT8_TOLERANCE, tokenizer=False)
    log("selftest ok")


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(prog="decide-export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"decide-export {__version__} (format {FORMAT})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="export a HF checkpoint directory to a decide model directory")
    ex.add_argument("--src", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--reference", type=Path, help="also write the self-check's torch reference here, for compress")
    co = sub.add_parser("compress", help="compress an exported model directory's weights to int8")
    co.add_argument("--model", type=Path, required=True)
    co.add_argument("--out", type=Path, required=True)
    co.add_argument("--reference", type=Path, required=True, help="export --reference of the model")
    sub.add_parser("selftest", help="export a tiny random model and check it against transformers")
    ck = sub.add_parser("check")
    ck.add_argument("--model", type=Path, required=True)
    ck.add_argument("--reference", type=Path, required=True)
    ck.add_argument("--tolerance", type=float, required=True)
    ck.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()
    if args.cmd == "export":
        cmd_export(args.src, args.out, args.reference)
    elif args.cmd == "compress":
        cmd_compress(args.model, args.out, args.reference)
    elif args.cmd == "selftest":
        cmd_selftest()
    else:
        run_check(args.model, args.reference, args.tolerance, not args.no_tokenizer)


if __name__ == "__main__":
    main()
