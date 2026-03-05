import argparse
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

def export_sdxl(model_id, output_dir, fp16_unet=False):
    print("=== Loading SDXL model ===")
    pipe = ORTStableDiffusionXLPipeline.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider",
        opset=16,
    )

    print("=== Saving ONNX models ===")
    pipe.save_pretrained(output_dir)

    print("=== Export completed ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fp16-unet", action="store_true")
    args = parser.parse_args()

    export_sdxl(args.model_id, args.output_dir, args.fp16_unet)

if __name__ == "__main__":
    main()
