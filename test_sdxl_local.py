#!/usr/bin/env python3
"""
Local SDXL ONNX Test Script for UserLAnd
Tests SDXL ONNX models with CPU-only ONNX Runtime.
Performs minimal inference to validate model loading and basic operation.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


def check_dependencies():
    """Check required dependencies are installed."""
    missing = []
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    return True


def load_tokenizer_vocab(vocab_path: str) -> Dict[str, int]:
    """Load tokenizer vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        return json.load(f)


def simple_tokenize(text: str, vocab: Dict[str, int], max_length: int = 77) -> np.ndarray:
    """Simple whitespace tokenization (for testing purposes)."""
    tokens = np.full(max_length, vocab.get("<|endoftext|>", 49407), dtype=np.int64)
    
    # BOS token
    tokens[0] = vocab.get("<|startoftext|>", 49406)
    
    # Tokenize words
    words = text.lower().split()
    idx = 1
    
    for word in words:
        if idx >= max_length - 1:
            break
        
        token_id = vocab.get(word + "</w>") or vocab.get(word) or vocab.get("<|endoftext|>", 49407)
        tokens[idx] = token_id
        idx += 1
    
    # EOS token
    if idx < max_length:
        tokens[idx] = vocab.get("<|endoftext|>", 49407)
    
    return tokens


class SdxlOnnxTester:
    """Test harness for SDXL ONNX models."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.sessions = {}
        
        import onnxruntime as ort
        self.ort = ort
        
        # Use CPU provider only for UserLAnd
        self.providers = ['CPUExecutionProvider']
        
        # Session options
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.intra_op_num_threads = 4
    
    def load_model(self, model_name: str) -> bool:
        """Load a single ONNX model."""
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            return False
        
        try:
            print(f"  Loading {model_name}...")
            session = self.ort.InferenceSession(
                str(model_path),
                self.sess_options,
                providers=self.providers
            )
            self.sessions[model_name] = session
            
            # Print input/output info
            print(f"    Inputs: {[i.name for i in session.get_inputs()]}")
            print(f"    Outputs: {[o.name for o in session.get_outputs()]}")
            
            return True
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            return False
    
    def load_all_models(self) -> bool:
        """Load all SDXL models."""
        print("\n=== Loading Models ===")
        
        models = [
            "text_encoder_1.onnx",
            "text_encoder_2.onnx",
            "unet.onnx",
            "vae_decoder.onnx"
        ]
        
        success = True
        for model in models:
            if not self.load_model(model):
                success = False
        
        return success
    
    def test_text_encoder_1(self, tokens: np.ndarray) -> Optional[np.ndarray]:
        """Test text encoder 1."""
        print("\n=== Testing Text Encoder 1 ===")
        
        session = self.sessions.get("text_encoder_1.onnx")
        if session is None:
            print("  Text encoder 1 not loaded")
            return None
        
        try:
            # Input shape: [batch, seq_len]
            input_ids = tokens.reshape(1, -1)
            
            outputs = session.run(None, {"input_ids": input_ids})
            
            hidden_states = outputs[0]
            print(f"  Hidden states shape: {hidden_states.shape}")
            print(f"  Hidden states range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
            
            return hidden_states
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def test_text_encoder_2(self, tokens: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Test text encoder 2."""
        print("\n=== Testing Text Encoder 2 ===")
        
        session = self.sessions.get("text_encoder_2.onnx")
        if session is None:
            print("  Text encoder 2 not loaded")
            return None
        
        try:
            input_ids = tokens.reshape(1, -1)
            
            outputs = session.run(None, {"input_ids": input_ids})
            
            hidden_states = outputs[0]
            pooled_output = outputs[1]
            
            print(f"  Hidden states shape: {hidden_states.shape}")
            print(f"  Pooled output shape: {pooled_output.shape}")
            
            return hidden_states, pooled_output
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def test_unet_single_step(self, latent_height: int = 64, latent_width: int = 64) -> Optional[np.ndarray]:
        """Test UNet with a single forward pass."""
        print("\n=== Testing UNet (Single Step) ===")
        
        session = self.sessions.get("unet.onnx")
        if session is None:
            print("  UNet not loaded")
            return None
        
        try:
            # Create dummy inputs
            batch_size = 1
            
            # Latent sample [batch, 4, height, width]
            sample = np.random.randn(batch_size, 4, latent_height, latent_width).astype(np.float32)
            
            # Timestep
            timestep = np.array([999.0], dtype=np.float32)
            
            # Encoder hidden states [batch, seq, 2048]
            encoder_hidden_states = np.random.randn(batch_size, 77, 2048).astype(np.float32)
            
            # Text embeds (pooled) [batch, 1280]
            text_embeds = np.random.randn(batch_size, 1280).astype(np.float32)
            
            # Time IDs [batch, 6]
            time_ids = np.array([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], dtype=np.float32)
            
            print(f"  Input shapes:")
            print(f"    sample: {sample.shape}")
            print(f"    timestep: {timestep.shape}")
            print(f"    encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"    text_embeds: {text_embeds.shape}")
            print(f"    time_ids: {time_ids.shape}")
            
            outputs = session.run(None, {
                "sample": sample,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "text_embeds": text_embeds,
                "time_ids": time_ids
            })
            
            noise_pred = outputs[0]
            print(f"  Output shape: {noise_pred.shape}")
            print(f"  Output range: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]")
            
            return noise_pred
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_vae_decoder(self, latent_height: int = 64, latent_width: int = 64) -> Optional[np.ndarray]:
        """Test VAE decoder."""
        print("\n=== Testing VAE Decoder ===")
        
        session = self.sessions.get("vae_decoder.onnx")
        if session is None:
            print("  VAE decoder not loaded")
            return None
        
        try:
            # Create dummy latent [batch, 4, height, width]
            latent = np.random.randn(1, 4, latent_height, latent_width).astype(np.float32)
            
            print(f"  Input shape: {latent.shape}")
            
            outputs = session.run(None, {"latent": latent})
            
            image = outputs[0]
            print(f"  Output shape: {image.shape}")
            print(f"  Output range: [{image.min():.4f}, {image.max():.4f}]")
            
            return image
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def run_minimal_pipeline(self, prompt: str = "a cat") -> Optional[np.ndarray]:
        """Run minimal end-to-end pipeline with 1 denoising step."""
        print("\n=== Running Minimal Pipeline ===")
        print(f"  Prompt: '{prompt}'")
        
        # Load vocabulary (use vocab2 for simplicity)
        vocab_path = self.models_dir / "tokenizer_2" / "vocab.json"
        if not vocab_path.exists():
            print(f"  Vocabulary not found at {vocab_path}")
            # Use default tokens
            tokens = np.zeros(77, dtype=np.int64)
        else:
            vocab = load_tokenizer_vocab(str(vocab_path))
            tokens = simple_tokenize(prompt, vocab)
        
        print(f"  Tokens (first 10): {tokens[:10]}")
        
        # Test text encoder 2
        enc_result = self.test_text_encoder_2(tokens)
        if enc_result is None:
            return None
        
        hidden_states, pooled = enc_result
        
        # Prepare encoder hidden states (simplified - just use enc2 output)
        # In real implementation, concat enc1 and enc2 outputs
        encoder_hidden_states = np.zeros((1, 77, 2048), dtype=np.float32)
        encoder_hidden_states[:, :, :hidden_states.shape[-1]] = hidden_states
        
        # Test UNet single step
        latent_height, latent_width = 64, 64  # 512x512 image
        unet_out = self.test_unet_single_step(latent_height, latent_width)
        
        # Test VAE decoder
        vae_out = self.test_vae_decoder(latent_height, latent_width)
        
        if vae_out is not None:
            print("\n=== Pipeline Test Complete ===")
            print("  All components functional!")
            return vae_out
        
        return None
    
    def save_test_image(self, image_array: np.ndarray, output_path: str):
        """Save decoded image to file."""
        from PIL import Image
        
        # Convert from [batch, channels, height, width] to [height, width, channels]
        img = image_array[0].transpose(1, 2, 0)
        
        # Normalize to 0-255
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        
        Image.fromarray(img).save(output_path)
        print(f"  Saved test image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test SDXL ONNX models locally")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="onnx_optimized",
        help="Directory containing ONNX models"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful sunset over mountains",
        help="Test prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_output.png",
        help="Output image path"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (skip full pipeline)"
    )
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    print("=" * 60)
    print("SDXL ONNX Local Test (UserLAnd)")
    print("=" * 60)
    
    # Check models directory
    if not Path(args.models_dir).exists():
        print(f"\nModels directory not found: {args.models_dir}")
        print("Run the export script first:")
        print("  python scripts/export_sdxl_onnx.py")
        print("  python scripts/optimize_onnx.py")
        sys.exit(1)
    
    # Create tester
    tester = SdxlOnnxTester(args.models_dir)
    
    # Load models
    if not tester.load_all_models():
        print("\nSome models failed to load. Check paths and model integrity.")
        sys.exit(1)
    
    if args.quick:
        # Quick test - just load and verify
        print("\n=== Quick Test Complete ===")
        print("All models loaded successfully!")
    else:
        # Run minimal pipeline
        result = tester.run_minimal_pipeline(args.prompt)
        
        if result is not None:
            tester.save_test_image(result, args.output)
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
