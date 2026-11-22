"""siglip-export: export a fixed-resolution SigLIP (2) checkpoint to OpenVINO IR.

SigLIP model format 1
=====================

A model directory holds vision.xml/.bin, text.xml/.bin, openvino_tokenizer.xml/.bin,
siglip.json, reference.json and the HF metadata (config.json, preprocessor_config.json,
tokenizer.json, tokenizer.model, tokenizer_config.json). Weights are stored as fp16, all
ports are f32 or i64. N (batch) is dynamic, everything else static.

vision.xml  pixel_values [N,3,S,S] f32 -> image_embeds [N,E] f32
text.xml    input_ids    [N,L]     i64 -> text_embeds  [N,E] f32
openvino_tokenizer.xml  string [N] -> input_ids [N,L] i64 (and attention_mask, unused)

S = image_size, E = embedding_size, L = max_text_length. The embeddings are the towers'
pooler_output, not normalised. Pixels are RGB resized to SxS with the resample filter,
then (v * rescale_factor - image_mean[c]) / image_std[c]. Text is lowercased, tokenized,
followed by <eos>, truncated and right padded with <pad> to L; there is no attention mask.
The tokenizer IR does all of that. The zero-shot probability that image i matches text t
is sigmoid(logit_scale * cos(image_i, text_t) + logit_bias).

siglip.json: {"format": 1, "source": "google/...", "revision": "<sha>", "weights": "fp16",
              "embedding_size": E, "image_size": S, "max_text_length": L,
              "resample": "bilinear", "rescale_factor": ..., "image_mean": [...],
              "image_std": [...], "logit_scale": ..., "logit_bias": ...}

reference.json: {"images": [{"a": .., "b": .., "embeds": [E floats]}, ...],
                 "texts": [{"prompt": "...", "input_ids": [L ids], "embeds": [E floats]}, ...]}
Reference image (a, b) is v[c][y][x] = (a*x + b*y + 85*c + 128*((x/16 + y/16) % 2)) % 256
(integer arithmetic) before rescale and normalisation; embeds are the f32 transformers outputs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

FORMAT = 1
METADATA = ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json")
RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}
MAX_TEXT_LENGTH = 64
REFERENCE_IMAGES = ({"a": 3, "b": 5}, {"a": 7, "b": 2})
# Mixed case on purpose: the tokenizer IR must lowercase.
REFERENCE_PROMPTS = ("A photo of a cat", "Two dogs playing in the snow.")
EXPORT_MIN_COSINE = 0.999


def log(msg: str) -> None:
    print(f"siglip-export: {msg}", file=sys.stderr, flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"siglip-export: {msg}")


def reference_pixels(spec: dict, meta: dict):
    import numpy as np

    s = meta["image_size"]
    c, y, x = np.mgrid[0:3, 0:s, 0:s]
    v = (spec["a"] * x + spec["b"] * y + 85 * c + 128 * ((x // 16 + y // 16) % 2)) % 256
    mean = np.array(meta["image_mean"]).reshape(3, 1, 1)
    std = np.array(meta["image_std"]).reshape(3, 1, 1)
    return ((v * meta["rescale_factor"] - mean) / std).astype(np.float32)[None]


def lowercasing(tok):
    """tok with a Lowercase normalizer ahead of its own, so the tokenizer IR lowercases too."""
    from tokenizers import normalizers

    bt = tok.backend_tokenizer
    bt.normalizer = normalizers.Sequence([normalizers.Lowercase()] + ([bt.normalizer] if bt.normalizer else []))
    return tok


def cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def convert(module, example, shape, ty, in_name: str, out_name: str, path: Path) -> None:
    import openvino as ov
    import torch

    with torch.no_grad():
        model = ov.convert_model(module, example_input=example, input=[(ov.PartialShape(shape), ty)])
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        fail(f"{path.name}: {len(model.inputs)} inputs, {len(model.outputs)} outputs, want 1 and 1")
    model.inputs[0].get_tensor().set_names({in_name})
    model.outputs[0].get_tensor().set_names({out_name})
    ov.save_model(model, path, compress_to_fp16=True)
    log(f"{path.with_suffix('.bin').name}: {path.with_suffix('.bin').stat().st_size / 2**30:.2f} GiB")


def cmd_export(src: Path, out: Path, source: str, revision: str) -> None:
    import shutil

    import numpy as np
    import openvino as ov
    import torch
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoTokenizer, SiglipModel

    config = json.loads((src / "config.json").read_text())
    if config.get("model_type") != "siglip":
        fail(f"model_type {config.get('model_type')!r} is not siglip")
    ip = json.loads((src / "preprocessor_config.json").read_text())
    if ip.get("image_processor_type") != "SiglipImageProcessor" or not (ip["do_resize"] and ip["do_rescale"]
                                                                        and ip["do_normalize"]):
        fail("preprocessor is not a resize/rescale/normalize SiglipImageProcessor")
    if ip["size"]["height"] != ip["size"]["width"]:
        fail(f"non-square image size {ip['size']}")

    model = SiglipModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa",
                                        local_files_only=True).eval()
    vc, tc = model.config.vision_config, model.config.text_config
    if ip["size"]["height"] != vc.image_size:
        fail(f"preprocessor size {ip['size']} != vision image_size {vc.image_size}")
    if vc.hidden_size != tc.hidden_size or tc.projection_size != tc.hidden_size:
        fail(f"vision {vc.hidden_size} / text {tc.hidden_size}->{tc.projection_size} embeddings differ")
    meta = {"format": FORMAT, "source": source, "revision": revision, "weights": "fp16",
            "embedding_size": vc.hidden_size, "image_size": vc.image_size, "max_text_length": MAX_TEXT_LENGTH,
            "resample": RESAMPLE[ip["resample"]], "rescale_factor": ip["rescale_factor"],
            "image_mean": ip["image_mean"], "image_std": ip["image_std"],
            "logit_scale": model.logit_scale.exp().item(), "logit_bias": model.logit_bias.item()}

    tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    ids = tok([p.lower() for p in REFERENCE_PROMPTS], padding="max_length", max_length=MAX_TEXT_LENGTH,
              truncation=True)["input_ids"]

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model.vision_model

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).pooler_output

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model.text_model

        def forward(self, input_ids):
            return self.m(input_ids=input_ids).pooler_output

    vision, text = Vision().eval(), Text().eval()
    pixels = torch.from_numpy(np.concatenate([reference_pixels(s, meta) for s in REFERENCE_IMAGES]))
    input_ids = torch.tensor(ids, dtype=torch.int64)
    with torch.no_grad():
        image_embeds, text_embeds = vision(pixels), text(input_ids)
    reference = {
        "images": [dict(s, embeds=e.tolist()) for s, e in zip(REFERENCE_IMAGES, image_embeds)],
        "texts": [{"prompt": p, "input_ids": i, "embeds": e.tolist()}
                  for p, i, e in zip(REFERENCE_PROMPTS, ids, text_embeds)],
    }
    for t in reference["texts"]:
        log(f"reference {t['prompt']!r} -> {t['input_ids'][:12]}...")

    out.mkdir(parents=True, exist_ok=True)
    s = vc.image_size
    convert(vision, (pixels,), [-1, 3, s, s], ov.Type.f32, "pixel_values", "image_embeds", out / "vision.xml")
    convert(text, (input_ids,), [-1, MAX_TEXT_LENGTH], ov.Type.i64, "input_ids", "text_embeds", out / "text.xml")
    del vision, text, model
    gc.collect()

    ov.save_model(convert_tokenizer(lowercasing(tok), with_detokenizer=False, max_length=MAX_TEXT_LENGTH,
                                    use_max_padding=True, truncation=True), out / "openvino_tokenizer.xml")
    for name in METADATA:
        shutil.copy2(src / name, out / name)
    (out / "siglip.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "reference.json").write_text(json.dumps(reference) + "\n")

    # A fresh process, so the check sees only what was written and not torch's memory.
    cmd = [sys.executable, os.path.abspath(__file__), "check", "--model", str(out),
           "--min-cosine", str(EXPORT_MIN_COSINE)]
    if subprocess.run(cmd).returncode != 0:
        fail("self-check failed")


def cmd_check(model_dir: Path, min_cosine: float) -> None:
    """Runs model_dir's IR on CPU in f32 against its reference.json."""
    import numpy as np
    import openvino as ov
    import openvino_tokenizers  # noqa: F401  registers the tokenizer ops

    meta = json.loads((model_dir / "siglip.json").read_text())
    ref = json.loads((model_dir / "reference.json").read_text())
    if meta.get("format") != FORMAT:
        fail(f"siglip.json format {meta.get('format')}, want {FORMAT}")
    core = ov.Core()
    f32 = {"INFERENCE_PRECISION_HINT": "f32"}
    e, worst = meta["embedding_size"], 1.0

    tok = core.compile_model(model_dir / "openvino_tokenizer.xml", "CPU")
    got = tok([t["prompt"] for t in ref["texts"]])["input_ids"].tolist()
    for t, ids in zip(ref["texts"], got):
        if ids != t["input_ids"]:
            fail(f"tokenizer IR: {t['prompt']!r} -> {ids}, want {t['input_ids']}")
    log(f"tokenizer IR matches {len(got)} reference prompts")

    for name, feed, cases in (
            ("vision", np.concatenate([reference_pixels(c, meta) for c in ref["images"]]), ref["images"]),
            ("text", np.array(got, np.int64), ref["texts"])):
        out = core.compile_model(model_dir / f"{name}.xml", "CPU", f32)(feed)[0]
        if out.shape != (len(cases), e):
            fail(f"{name}: output shape {out.shape}, want {(len(cases), e)}")
        for i, c in enumerate(cases):
            cos = cosine(out[i], c["embeds"])
            log(f"{name} case {i}: cosine {cos:.6f}")
            worst = min(worst, cos)
            if not cos >= min_cosine:
                fail(f"{name} case {i}: cosine {cos:.6f} < {min_cosine}")

    img = np.array([c["embeds"] for c in ref["images"]])
    txt = np.array([t["embeds"] for t in ref["texts"]])
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    logits = meta["logit_scale"] * img @ txt.T + meta["logit_bias"]
    log(f"self-check ok: worst cosine {worst:.6f}, reference zero-shot logits {logits.round(3).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="siglip-export", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="export a HF checkpoint directory to a SigLIP model directory")
    ex.add_argument("--src", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--source", required=True, help="HF repo id, recorded in siglip.json")
    ex.add_argument("--revision", required=True, help="HF commit, recorded in siglip.json")
    ck = sub.add_parser("check", help="check a model directory against its reference.json")
    ck.add_argument("--model", type=Path, required=True)
    ck.add_argument("--min-cosine", type=float, required=True)
    args = ap.parse_args()
    if args.cmd == "export":
        cmd_export(args.src, args.out, args.source, args.revision)
    else:
        cmd_check(args.model, args.min_cosine)


if __name__ == "__main__":
    main()
