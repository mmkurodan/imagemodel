#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import onnx
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel


def export_unet(unet: UNet2DConditionModel, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CPU での export 用に float32 に変換
    unet = unet.to(torch.float32)

    # Dummy inputs (dynamic shape export, float32)
    batch = 1
    height = 512
    width = 512
    sample = torch.randn(batch, 4, height // 8, width // 8, dtype=torch.float32)
    timestep = torch.tensor([1.0], dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch, 77, 2048, dtype=torch.float32)

    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        unet,
        (sample, timestep, encoder_hidden_states),
        str(onnx_path),
        opset_version=17,
        do_constant_folding=True,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch", 2: "height", 3: "width"},
            "out_sample": {0: "batch", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch"},
        },
    )

    # External data split
    model = onnx.load(str(onnx_path))
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        location="",
        size_threshold=1024,
    )


def export_text_encoder(model: CLIPTextModel, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CPU での export 用に float32 に変換
    model = model.to(torch.float32)

    dummy = torch.randint(0, 10000, (1, 77))
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
    )


def export_vae(vae: AutoencoderKL, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CPU での export 用に float32 に変換
    vae = vae.to(torch.float32)

    dummy = torch.randn(1, 4, 64, 64)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        vae,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={"latent_sample": {0: "batch"}},
    )


def copy_json(src_dir: Path, dst_dir: Path, names):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        dst = dst_dir / name
        if src.exists():
            dst.write_bytes(src.read_bytes())


def write_manifest(out_dir: Path, model_id: str):
    manifest = {
        "model_id": model_id,
        "runtime": "onnxruntime-android",
        "components": [
            "unet",
            "text_encoder",
            "text_encoder_2",
            "vae_decoder",
            "tokenizer",
            "tokenizer_2",
            "scheduler",
        ],
    }
    with (out_dir / "android_onnx_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def export_sdxl(model_id: str, output_dir: Path):
    print(f"Loading SDXL base: {model_id}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # ロードは fp16 で OK（メモリ節約）
        use_safetensors=True,
    )

    # Export components (各 export 内で float32 に変換)
    export_unet(pipe.unet, output_dir / "unet")
    export_text_encoder(pipe.text_encoder, output_dir / "text_encoder")
    export_text_encoder(pipe.text_encoder_2, output_dir / "text_encoder_2")
    export_vae(pipe.vae, output_dir / "vae_decoder")

    # Copy tokenizer / scheduler
    copy_json(pipe.tokenizer, output_dir / "tokenizer", ["vocab.json", "merges.txt"])
    copy_json(pipe.tokenizer_2, output_dir / "tokenizer_2", ["vocab.json", "merges.txt"])
    copy_json(pipe.scheduler, output_dir / "scheduler", ["scheduler_config.json"])

    # model_index.json
    (output_dir / "model_index.json").write_text(
        json.dumps(pipe.config, indent=2)
    )

    write_manifest(output_dir, model_id)

    print(f"Export completed: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sdxl-onnx"),
    )
    args = parser.parse_args()

    export_sdxl(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
