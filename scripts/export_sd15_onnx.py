#!/usr/bin/env python3

import argparse
import json
import threading
import time
from pathlib import Path

import torch
import onnx
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTextModel


# ============================================================
# GitHub Actions keep-alive
# ============================================================
def keep_alive():
    while True:
        print("Export running... (keep-alive)")
        time.sleep(60)


# ============================================================
# UNet Wrapper（Linear + Conv2d を float32 に揃える）
# ============================================================
class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

        # Linear と Conv2d の weight/bias を float32 に変換
        for module in self.unet.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                if module.weight is not None:
                    module.weight.data = module.weight.data.float()
                if module.bias is not None:
                    module.bias.data = module.bias.data.float()

    def forward(self, sample, timestep, encoder_hidden_states):
        sample = sample.float()
        timestep = timestep.float()
        encoder_hidden_states = encoder_hidden_states.float()
        return self.unet(sample, timestep, encoder_hidden_states)


# ============================================================
# UNet Export（Linear + Conv2d を float32 化した UNetWrapper を使用）
# ============================================================
def export_unet(unet: UNet2DConditionModel, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ★ SD1.5 でも dtype mismatch を防ぐため必須
    unet = UNetWrapper(unet)

    batch = 1
    height = 512
    width = 512

    sample = torch.randn(batch, 4, height // 8, width // 8, dtype=torch.float32)
    timestep = torch.tensor([1.0], dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch, 77, 768, dtype=torch.float32)

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


# ============================================================
# Text Encoder Export（float32）
# ============================================================
def export_text_encoder(model: CLIPTextModel, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

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


# ============================================================
# VAE Export（float32）
# ============================================================
def export_vae(vae: AutoencoderKL, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export VAE decoder explicitly (AutoencoderKL.forward performs encoding by default)
    class VaeDecoderWrapper(torch.nn.Module):
        def __init__(self, vae_module):
            super().__init__()
            self.vae = vae_module
            self.vae.to(torch.float32)

        def forward(self, latent_sample):
            # decode returns image tensor [batch, 3, H, W]
            return self.vae.decode(latent_sample)

    wrapper = VaeDecoderWrapper(vae).to(torch.float32)

    dummy = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={"latent_sample": {0: "batch"}},
    )


# ============================================================
# Utility
# ============================================================
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
            "vae_decoder",
            "tokenizer",
            "scheduler",
        ],
    }
    with (out_dir / "android_onnx_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


# ============================================================
# Main Export
# ============================================================
def export_sd15(model_id: str, output_dir: Path):
    print(f"Loading SD1.5 base: {model_id}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe.unet.set_attn_processor(AttnProcessor())

    export_unet(pipe.unet, output_dir / "unet")
    export_text_encoder(pipe.text_encoder, output_dir / "text_encoder")
    export_vae(pipe.vae, output_dir / "vae_decoder")

    pipe.tokenizer.save_pretrained(output_dir / "tokenizer")
    pipe.scheduler.save_pretrained(output_dir / "scheduler")

    (output_dir / "model_index.json").write_text(
        json.dumps(pipe.config, indent=2)
    )

    write_manifest(output_dir, model_id)

    print(f"Export completed: {output_dir}")


# ============================================================
# Entry Point
# ============================================================
def main():
    threading.Thread(target=keep_alive, daemon=True).start()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", type=Path, default=Path("sd15-base"))
    args = parser.parse_args()

    export_sd15(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
