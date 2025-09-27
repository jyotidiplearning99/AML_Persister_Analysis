#!/usr/bin/env python3
import os, sys, json, traceback, h5py

def try_extract_config_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "model_config" in f.attrs:
            raw = f.attrs["model_config"]
            raw = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            return json.loads(raw)
        for k in f.keys():
            grp = f[k]
            if hasattr(grp, "attrs") and "model_config" in grp.attrs:
                raw = grp.attrs["model_config"]
                raw = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                return json.loads(raw)
    return None

def text_summary_from_config(cfg):
    lines = []
    if not isinstance(cfg, dict):
        return ["<unrecognized config>"]
    cls = cfg.get("class_name")
    lines.append(f"Model class: {cls}")
    # Handle common Keras formats
    layers = None
    if "config" in cfg:
        if isinstance(cfg["config"], dict) and "layers" in cfg["config"]:
            layers = cfg["config"]["layers"]
        elif isinstance(cfg["config"], dict) and "config" in cfg["config"] and "layers" in cfg["config"]["config"]:
            layers = cfg["config"]["config"]["layers"]
    if layers is None:
        lines.append("(layers not found in config; printing raw JSON keys)")
        lines.append(", ".join(sorted(cfg.keys())))
        return lines

    for i, layer in enumerate(layers):
        cname = layer.get("class_name")
        lcfg  = layer.get("config", {})
        name  = lcfg.get("name", f"layer_{i}")
        core_keys = ("units","filters","kernel_size","strides","padding","activation",
                     "rate","pool_size","data_format","dilation_rate","use_bias")
        core = ", ".join([f"{k}={lcfg[k]}" for k in core_keys if k in lcfg])
        lines.append(f"{i:3d}. {name:<30} {cname:<22} {core}")
    return lines

def safe_write(path, content, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        f.write(content)

def inspect_one(h5_path):
    base = os.path.splitext(os.path.basename(h5_path))[0]
    print(f"\n=== Inspecting {h5_path} ===")
    # 1) Try full load with tf.keras
    try:
        import tensorflow as tf
        print("[*] Attempting tf.keras.models.load_model ...")
        model = tf.keras.models.load_model(h5_path, compile=False)
        # Text summary
        summary_lines = []
        model.summary(print_fn=summary_lines.append)
        safe_write(f"{base}_summary.txt", "\n".join(summary_lines))

        # JSON architecture
        try:
            json_str = model.to_json()
            safe_write(f"{base}_architecture.json", json_str)
            print(f"[+] Saved JSON to {base}_architecture.json and summary to {base}_summary.txt")
        except Exception:
            print("[!] Could not serialize model to JSON (custom/lambda layers?).")

        # Diagram: DOT (always) and PNG (if graphviz present)
        try:
            from tensorflow.keras.utils import model_to_dot
            graph = model_to_dot(model, show_shapes=True, show_layer_names=True, dpi=200)
            dot_str = graph.to_string()
            safe_write(f"{base}_architecture.dot", dot_str)
            print(f"[+] Saved DOT graph to {base}_architecture.dot")
            # Try PNG (requires Graphviz)
            try:
                graph.write_png(f"{base}_architecture.png")
                print(f"[+] Saved PNG diagram to {base}_architecture.png")
            except Exception:
                print("[!] Could not render PNG (Graphviz not available). DOT saved; render locally with:")
                print(f"    dot -Tpng {base}_architecture.dot -o {base}_architecture.png")
        except Exception:
            print("[!] Could not create graph DOT/PNG (missing pydot?).")

        return
    except Exception:
        print("[!] tf.keras load failed; falling back to raw HDF5 inspection.")
        # Uncomment to debug:
        # traceback.print_exc()

    # 2) Fallback: HDF5 config extraction (works for weights-only metadata if present)
    try:
        cfg = try_extract_config_from_h5(h5_path)
        if cfg is None:
            print("[!] No model_config found. This file is likely WEIGHTS-ONLY.")
            print("    Rebuild your model code and call model.load_weights(<file.h5>).")
            safe_write(f"{base}_note.txt",
                       "No model_config found in H5. Likely weights-only. "
                       "Recreate the model code and use model.load_weights().")
            return
        safe_write(f"{base}_model_config.json", json.dumps(cfg, indent=2))
        lines = text_summary_from_config(cfg)
        safe_write(f"{base}_summary.txt", "\n".join(lines))
        print(f"[+] Saved raw config to {base}_model_config.json and a summary to {base}_summary.txt")
    except Exception:
        print("[!] Failed to parse model_config from HDF5.")
        traceback.print_exc()

def main():
    targets = sys.argv[1:] or ["model_reduced.h5", "final_model.h5"]
    for t in targets:
        if os.path.exists(t):
            inspect_one(t)
        else:
            print(f"[!] File not found: {t}")

if __name__ == "__main__":
    main()
