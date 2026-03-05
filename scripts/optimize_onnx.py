#!/usr/bin/env python3
"""
SDXL ONNX Optimization Script (Folder-structure aware, stable version)
"""

import argparse
import json
from pathlib import Path
import shutil
import onnx
from onnxruntime.transformers import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def copy_external_data(src_dir: Path, dst_dir: Path, base_name: str = "model.onnx"):
    """Copy only external data files related to model.onnx (do NOT overwrite model.onnx)."""
    # Copy any external data variants:
    # model.onnx.data*, model.onnx_data*, model.onnx_data_00000, etc.
    for data_file in src_dir.glob(f"{base_name}*"):
        if data_file.name == base_name:
            continue
        shutil.copy(data_file, dst_dir / data_file.name)


def optimize_text_encoder(src: Path, dst: Path):
    print(f"Optimizing TextEncoder: {src}")
    try:
        opt_model = optimizer.optimize_model(
            str(src),
            model_type="bert",
            num_heads=0,
            hidden_size=0,
            optimization_options=optimizer.FusionOptions("bert"),
        )
        opt_model.save_model_to_file(str(dst))
        print("  Transformer optimization applied")
    except Exception as e:
        print(f"  Optimization failed: {e}")
        shutil.copy(src, dst)


def quantize_model(src: Path, dst: Path, allow_unet: bool = False):
    name = src.name.lower()
    if "unet" in name and not allow_unet:
        print("  Skipping UNet quantization")
        shutil.copy(src, dst)
        return

    print(f"Quantizing {src}...")
    try:
        quantize_dynamic(
            str(src),
            str(dst),
            weight_type=QuantType.QUInt8,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
        print("  Quantization OK")
    except Exception as e:
        print(f"  Quantization failed: {e}")
        shutil.copy(src, dst)


def mobile_adjust(model_path: Path):
    model = onnx.load(str(model_path))
    if getattr(model, "ir_version", 0) > 8:
        print(f"  Lowering IR version {model.ir_version} -> 8")
        model.ir_version = 8
    onnx.save(model, str(model_path))


def add_metadata(model_path: Path, metadata: dict):
    model = onnx.load(str(model_path))
    for k, v in metadata.items():
        meta = model.metadata_props.add()
        meta.key = k
        meta.value = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    onnx.save(model, str(model_path))


def get_io_info(model_path: Path) -> dict:
    model = onnx.load(str(model_path))

    def shape(t):
        return [
            (d.dim_value if d.dim_value > 0 else (d.dim_param if d.dim_param else None))
            for d in t.type.tensor_type.shape.dim
        ]

    def dtype(t):
        m = {1: "float32", 7: "int64", 10: "float16", 11: "double"}
        return m.get(t.type.tensor_type.elem_type, "unknown")

    inputs = {i.name: {"shape": shape(i), "dtype": dtype(i)} for i in model.graph.input}
    outputs = {o.name: {"shape": shape(o), "dtype": dtype(o)} for o in model.graph.output}
    return {"inputs": inputs, "outputs": outputs}


# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------

def process_component(name: str, src_dir: Path, dst_dir: Path, quantize: bool = False, quantize_unet: bool = False):
    print(f"\n=== Processing {name} ===")

    src_model = src_dir / "model.onnx"
    if not src_model.exists():
        print(f"  Skipped: {src_model} not found")
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_dir / "tmp_model.onnx"
    out_path = dst_dir / "model.onnx"

    # 1) Optimization
    if "text_encoder" in name:
        optimize_text_encoder(src_model, tmp_path)
    else:
        shutil.copy(src_model, tmp_path)

    # 2) Quantization
    if quantize:
        quantize_model(tmp_path, out_path, allow_unet=quantize_unet)
    else:
        shutil.copy(tmp_path, out_path)

    # 3) Copy external data FIRST (but do not overwrite model.onnx)
    copy_external_data(src_dir, dst_dir, base_name="model.onnx")

    # 4) Mobile adjustments (UNet is skipped to avoid OOM)
    if "unet" not in name:
        mobile_adjust(out_path)
    else:
        print("  Skipping mobile_adjust for UNet (too large for onnx.load)")

    # 5) Metadata（UNet は巨大なので IO 解析もスキップ推奨）
    if "unet" not in name:
        io_info = get_io_info(out_path)
        add_metadata(out_path, {
            "component": name,
            "optimized": True,
            "quantized": quantize,
            "io_info": io_info,
        })
    else:
        add_metadata(out_path, {
            "component": name,
            "optimized": True,
            "quantized": quantize,
            "io_info": "skipped_for_unet_due_to_size",
        })

    if tmp_path.exists():
        tmp_path.unlink()

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="onnx")
    parser.add_argument("--output-dir", type=str, default="onnx_optimized")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quantize-unet", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    components = {
        "unet": input_dir / "unet",
        "text_encoder": input_dir / "text_encoder",
        "text_encoder_2": input_dir / "text_encoder_2",
        "vae_decoder": input_dir / "vae_decoder",
        "vae_encoder": input_dir / "vae_encoder",
    }

    manifest = {"models": {}, "version": "1.0", "framework": "onnx_runtime_mobile"}

    for name, src in components.items():
        out = process_component(
            name=name,
            src_dir=src,
            dst_dir=output_dir / name,
            quantize=args.quantize,
            quantize_unet=args.quantize_unet,
        )
        if out:
            # UNet の IO 情報もここではスキップしておく
            if "unet" not in name:
                io_info = get_io_info(out)
            else:
                io_info = "skipped_for_unet_due_to_size"

            manifest["models"][name] = {
                "size_mb": round(out.stat().st_size / (1024 * 1024), 2),
                "io_info": io_info,
            }

    for folder in ("tokenizer", "tokenizer_2", "scheduler"):
        s = input_dir / folder
        d = output_dir / folder
        if s.exists():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
            print(f"Copied {folder}")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Summary ===")
    for name, info in manifest["models"].items():
        print(f"{name}: {info['size_mb']} MB")


if __name__ == "__main__":
    main()
