#!/usr/bin/env python3
"""
ONNX Optimization Script

This script performs ONNX Runtime optimizations and optional quantization for SDXL
components. It does NOT perform any opset version conversion (no onnxscript/version_converter).
It assumes inputs are already exported with the desired opset (e.g. opset 16) and
only runs optimizer/quantization and lightweight mobile compat adjustments.
"""

import argparse
import json
from pathlib import Path
import shutil
import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType


def optimize_with_ort(input_path: str, output_path: str, model_type: str) -> bool:
    """Run ONNX Runtime transformer optimizer for supported transformer models.

    For non-transformer models (UNet/VAE), the optimizer may not apply; in that case
    the input model is copied to the output path unchanged.
    """
    print(f"Optimizing {input_path} with ONNX Runtime optimizer (model_type={model_type})...")

    try:
        if model_type in ("bert", "gpt2"):
            # Use the transformer optimizer where applicable
            opt_model = optimizer.optimize_model(
                input_path,
                model_type=model_type,
                num_heads=0,  # auto-detect
                hidden_size=0,  # auto-detect
                optimization_options=optimizer.FusionOptions(model_type),
            )
            opt_model.save_model_to_file(output_path)
            print(f"  Optimized and saved to {output_path}")
        else:
            # For other model types, do not attempt graph/ops rewriting beyond simple copy
            shutil.copy(input_path, output_path)
            print("  No transformer optimizations available for this model type; copied original")
        return True

    except Exception as e:
        print(f"  Optimization failed: {e}")
        try:
            shutil.copy(input_path, output_path)
        except Exception:
            pass
        return False


def quantize_model(input_path: str, output_path: str, quantize_unet: bool = False):
    """Apply dynamic quantization using ONNX Runtime quantizer.

    By default, UNet quantization is skipped unless quantize_unet=True (it can degrade quality).
    """
    print(f"Quantizing {input_path}...")

    if "unet" in Path(input_path).name.lower() and not quantize_unet:
        print("  Skipping UNet quantization (use --quantize-unet to enable)")
        shutil.copy(input_path, output_path)
        return

    try:
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
            },
        )
        orig_size = Path(input_path).stat().st_size / (1024 * 1024)
        new_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  Quantized: {orig_size:.1f} MB -> {new_size:.1f} MB")
    except Exception as e:
        print(f"  Quantization failed: {e}")
        shutil.copy(input_path, output_path)


def add_metadata(model_path: str, metadata: dict):
    print(f"Adding metadata to {model_path}...")
    model = onnx.load(model_path)
    for k, v in metadata.items():
        meta = model.metadata_props.add()
        meta.key = k
        meta.value = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    onnx.save(model, model_path)


def get_model_io_info(model_path: str) -> dict:
    model = onnx.load(model_path)

    def get_shape(tensor):
        return [
            (dim.dim_value if (hasattr(dim, 'dim_value') and dim.dim_value > 0) else (dim.dim_param if hasattr(dim, 'dim_param') else None))
            for dim in tensor.type.tensor_type.shape.dim
        ]

    def get_dtype(tensor):
        dtype_map = {1: "float32", 7: "int64", 10: "float16", 11: "double"}
        return dtype_map.get(tensor.type.tensor_type.elem_type, "unknown")

    inputs = {inp.name: {"shape": get_shape(inp), "dtype": get_dtype(inp)} for inp in model.graph.input}
    outputs = {out.name: {"shape": get_shape(out), "dtype": get_dtype(out)} for out in model.graph.output}
    return {"inputs": inputs, "outputs": outputs}


def validate_model(model_path: str) -> bool:
    print(f"Validating {model_path} with ONNX Runtime...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
        print("  Validation succeeded")
        return True
    except Exception as e:
        print(f"  Validation failed: {e}")
        return False


def optimize_for_mobile(model_path: str, output_path: str):
    """Apply lightweight mobile compatibility adjustments without changing opset.

    This avoids any op rewriting (e.g. Resize) that would require opset conversion.
    """
    print(f"Applying mobile adjustments to {model_path}...")
    model = onnx.load(model_path)
    # Reduce IR version if necessary for older mobile runtimes
    if getattr(model, 'ir_version', 0) > 8:
        print(f"  Lowering IR version from {model.ir_version} to 8 for mobile compatibility")
        model.ir_version = 8
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser(description="Optimize SDXL ONNX models (no opset conversion)")
    parser.add_argument("--input-dir", type=str, default="onnx", help="Input directory with ONNX files")
    parser.add_argument("--output-dir", type=str, default="onnx_optimized", help="Output directory for optimized ONNX files")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization to text encoders")
    parser.add_argument("--quantize-unet", action="store_true", help="Also quantize UNet (may reduce quality)")
    parser.add_argument("--mobile", action="store_true", default=True, help="Apply mobile-specific optimizations")
    parser.add_argument("--skip-validation", action="store_true", help="Skip model validation")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "text_encoder_1.onnx": {"type": "bert", "quantize": args.quantize},
        "text_encoder_2.onnx": {"type": "bert", "quantize": args.quantize},
        "unet.onnx": {"type": "unet", "quantize": args.quantize and args.quantize_unet},
        "vae_decoder.onnx": {"type": "vae", "quantize": False},
    }

    for model_name, cfg in models.items():
        src = input_dir / model_name
        if not src.exists():
            print(f"Skipping {model_name} - not found in {input_dir}")
            continue

        tmp = output_dir / f"tmp_{model_name}"
        out = output_dir / model_name

        # 1) Run ONNX Runtime optimizer where applicable (no opset conversion)
        optimize_with_ort(str(src), str(tmp), cfg["type"])

        # 2) Quantize or copy
        if cfg.get("quantize", False):
            quantize_model(str(tmp), str(out), args.quantize_unet)
        else:
            shutil.copy(str(tmp), str(out))

        # 3) Mobile tweaks
        if args.mobile:
            optimize_for_mobile(str(out), str(out))

        # 4) Metadata and validation
        io_info = get_model_io_info(str(out))
        metadata = {
            "model_name": model_name.replace('.onnx', ''),
            "model_type": "sdxl",
            "component": cfg["type"],
            "io_info": io_info,
            "optimized": True,
            "quantized": cfg.get("quantize", False),
        }
        add_metadata(str(out), metadata)

        if not args.skip_validation:
            validate_model(str(out))

        # cleanup
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

        print()

    # Copy tokenizer/scheduler directories if present
    for name in ("tokenizer_1", "tokenizer_2", "scheduler"):
        s = input_dir / name
        d = output_dir / name
        if s.exists():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
            print(f"Copied {name} to {d}")

    # Write manifest
    manifest = {"models": {}, "version": "1.0", "framework": "onnx_runtime_mobile"}
    for mf in output_dir.glob('*.onnx'):
        manifest['models'][mf.name] = {"size_mb": round(mf.stat().st_size / (1024*1024), 2), "io_info": get_model_io_info(str(mf))}

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    # Summary
    print('\n' + '='*50)
    print('Optimization Summary')
    print('='*50)
    total = 0.0
    for mf in sorted(output_dir.glob('*.onnx')):
        sz = mf.stat().st_size / (1024*1024)
        total += sz
        print(f"  {mf.name}: {sz:.1f} MB")
    print(f"\nTotal model size: {total:.1f} MB")


if __name__ == '__main__':
    main()
