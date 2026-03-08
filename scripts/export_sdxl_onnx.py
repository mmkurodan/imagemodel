#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path


REQUIRED_DIRS = (
    "unet",
    "text_encoder",
    "text_encoder_2",
    "vae_decoder",
    "scheduler",
    "tokenizer",
    "tokenizer_2",
)
REQUIRED_ONNX_COMPONENTS = ("unet", "text_encoder", "text_encoder_2", "vae_decoder")


def validate_export(output_dir: Path) -> None:
    missing_dirs = [name for name in REQUIRED_DIRS if not (output_dir / name).exists()]
    if missing_dirs:
        raise RuntimeError(f"Missing exported directories: {', '.join(missing_dirs)}")

    missing_onnx = [
        f"{name}/model.onnx"
        for name in REQUIRED_ONNX_COMPONENTS
        if not (output_dir / name / "model.onnx").exists()
    ]
    if missing_onnx:
        raise RuntimeError(f"Missing exported ONNX files: {', '.join(missing_onnx)}")


def write_manifest(output_dir: Path, model_id: str) -> None:
    manifest = {
        "model_id": model_id,
        "runtime": "onnxruntime-android",
        "required_components": list(REQUIRED_ONNX_COMPONENTS),
    }
    with (output_dir / "android_onnx_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def export_sdxl(model_id: str, output_dir: Path, revision: str | None = None) -> None:
    print(f"=== Export model: {model_id} ===")
    try:
        from optimum.onnxruntime import ORTStableDiffusionXLPipeline
    except ImportError as exc:
        raise RuntimeError(
            "optimum[onnxruntime] is required. Install with: pip install -r scripts/requirements.txt"
        ) from exc

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    kwargs = {
        "export": True,
        "provider": "CPUExecutionProvider",
    }
    if revision:
        kwargs["revision"] = revision
    if hf_token:
        kwargs["token"] = hf_token

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id, **kwargs)
    pipeline.save_pretrained(str(output_dir))

    validate_export(output_dir)
    write_manifest(output_dir, model_id)
    print(f"=== Export completed: {output_dir} ===")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SDXL from Hugging Face and export Android-ready ONNX assets."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onnx"),
    )
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    export_sdxl(args.model_id, args.output_dir, args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
