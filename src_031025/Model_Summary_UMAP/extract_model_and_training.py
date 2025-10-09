#!/usr/bin/env python3
"""
Extract model architecture + training details from Keras .h5 files
- No placeholders: prints only what is present in the file
- Works with h5py if tf.keras load_model fails
Outputs:
  - JSON + Markdown summaries in the chosen output directory
"""

import json
import h5py
import numpy as np
from pathlib import Path
import argparse
import sys
import platform

try:
    import tensorflow as tf  # optional; used only if available
except Exception:
    tf = None

def _sum_param_count_from_h5_weights(f: h5py.File) -> int:
    """Sum parameter counts directly from model_weights group."""
    if 'model_weights' not in f:
        return 0
    total = 0
    def walk(group):
        nonlocal total
        for k, v in group.items():
            if isinstance(v, h5py.Dataset):
                total += int(np.prod(v.shape))
            elif isinstance(v, h5py.Group):
                walk(v)
    walk(f['model_weights'])
    return total

def _safe_attr_bytes(attrs, key):
    if key not in attrs:
        return None
    val = attrs[key]
    if isinstance(val, (bytes, bytearray)):
        return val.decode('utf-8')
    return val

def extract_from_h5(h5_path: Path):
    out = {
        "file": str(h5_path),
        "backend": None,
        "keras_version": None,
        "model_config": None,
        "training_config": None,
        "param_count": None,
        "input_shapes": [],
        "layers": []
    }
    with h5py.File(h5_path, 'r') as f:
        out["backend"] = _safe_attr_bytes(f.attrs, 'backend')
        out["keras_version"] = _safe_attr_bytes(f.attrs, 'keras_version')

        # model_config / training_config
        mc = _safe_attr_bytes(f.attrs, 'model_config')
        if mc:
            try:
                mc_json = json.loads(mc)
                out["model_config"] = mc_json
                # collect InputLayer shapes and layer names/types
                conf = mc_json.get("config", {})
                # Keras Sequential vs Functional
                if "layers" in conf:
                    layers = conf["layers"]
                else:
                    layers = conf.get("layers", [])
                for L in layers:
                    cname = L.get("class_name", "")
                    cfg = L.get("config", {})
                    if cname.lower() == "inputlayer":
                        out["input_shapes"].append(cfg.get("batch_input_shape"))
                    out["layers"].append({"class_name": cname, "config_keys": list(cfg.keys())})
            except Exception:
                pass

        tc = _safe_attr_bytes(f.attrs, 'training_config')
        if tc:
            try:
                out["training_config"] = json.loads(tc)
            except Exception:
                pass

        out["param_count"] = _sum_param_count_from_h5_weights(f)
    return out

def to_markdown(summary: dict, model_name: str) -> str:
    lines = []
    lines.append(f"### {model_name}")
    lines.append("")
    lines.append(f"- **File:** `{summary['file']}`")
    if summary.get("backend"): lines.append(f"- **Backend:** {summary['backend']}")
    if summary.get("keras_version"): lines.append(f"- **Keras version:** {summary['keras_version']}")
    if summary.get("param_count") is not None: lines.append(f"- **Total parameters:** {summary['param_count']:,}")
    if summary.get("input_shapes"):
        shapes = ", ".join([str(s) for s in summary["input_shapes"] if s])
        lines.append(f"- **Input shapes (from InputLayer):** {shapes}")
    tc = summary.get("training_config") or {}
    opt = tc.get("optimizer_config", {})
    if opt:
        lines.append(f"- **Optimizer:** {opt.get('class_name')}")
        lr = (opt.get("config") or {}).get("learning_rate")
        if lr is not None:
            lines.append(f"- **Learning rate:** {lr}")
    loss = tc.get("loss")
    if loss: lines.append(f"- **Loss:** {loss}")
    metrics = tc.get("metrics")
    if metrics: lines.append(f"- **Metrics:** {metrics}")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-model", type=Path, required=True, help="Path to full Transformer .h5")
    ap.add_argument("--reduced-model", type=Path, required=True, help="Path to distilled MLP .h5")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, path in [("Transformer", args.final_model), ("Reduced_MLP", args.reduced_model)]:
        if not path.exists():
            print(f"[ERROR] Missing model file: {path}", file=sys.stderr)
            sys.exit(1)
        summaries[name] = extract_from_h5(path)

    # Environment info
    env = {
        "python": platform.python_version(),
        "tensorflow": getattr(tf, "__version__", None),
        "system": platform.platform()
    }

    # Save JSON
    out_json = {
        "environment": env,
        "models": summaries
    }
    (args.out_dir / "model_and_training_summary.json").write_text(json.dumps(out_json, indent=2))

    # Save Markdown
    md = ["# Model & Training Summary", ""]
    md.append(f"- Python: {env['python']}")
    if env["tensorflow"]: md.append(f"- TensorFlow: {env['tensorflow']}")
    md.append(f"- System: {env['system']}")
    md.append("")
    md.append(to_markdown(summaries["Transformer"], "Transformer"))
    md.append(to_markdown(summaries["Reduced_MLP"], "Reduced MLP"))
    (args.out_dir / "model_and_training_summary.md").write_text("\n".join(md))

    print(f"[OK] Wrote JSON and Markdown to: {args.out_dir}")

if __name__ == "__main__":
    main()
