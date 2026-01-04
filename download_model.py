#!/usr/bin/env python3
"""Download Fish Speech models from Hugging Face"""

import subprocess
import sys
from pathlib import Path

def download_models():
    """Download openaudio-s1-mini model from Hugging Face"""
    
    fish_speech_dir = Path.home() / "Development/fish-speech"
    checkpoint_dir = fish_speech_dir / "checkpoints/openaudio-s1-mini"
    
    print("=" * 60)
    print("Fish Speech Model Downloader")
    print("=" * 60)
    print()
    print(f"Target directory: {checkpoint_dir}")
    print("Model: openaudio-s1-mini (~3.3GB)")
    print()
    
    # Check if already downloaded
    model_file = checkpoint_dir / "model.pth"
    codec_file = checkpoint_dir / "codec.pth"
    
    if model_file.exists() and codec_file.exists():
        print("✅ Models already downloaded!")
        print(f"   - model.pth: {model_file.stat().st_size / 1e9:.1f}GB")
        print(f"   - codec.pth: {codec_file.stat().st_size / 1e9:.1f}GB")
        print()
        response = input("Download again? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    # Check for huggingface-cli
    try:
        subprocess.run(["huggingface-cli", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: huggingface-cli not found!")
        print()
        print("Please install it first:")
        print("  pip install huggingface-hub")
        return False
    
    # Check authentication
    print("Checking Hugging Face authentication...")
    try:
        result = subprocess.run(["huggingface-cli", "whoami"], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Not logged in to Hugging Face")
            print()
            print("Please login first:")
            print("  huggingface-cli login")
            print()
            response = input("Login now? (Y/n): ")
            if response.lower() != 'n':
                subprocess.run(["huggingface-cli", "login"])
            else:
                return False
    except Exception as e:
        print(f"❌ Error checking authentication: {e}")
        return False
    
    # Download model
    print()
    print("Downloading openaudio-s1-mini model...")
    print("This will take several minutes depending on your connection.")
    print()
    
    try:
        subprocess.run([
            "huggingface-cli", "download",
            "fishaudio/openaudio-s1-mini",
            "--local-dir", str(checkpoint_dir),
            "--local-dir-use-symlinks", "False"
        ], check=True)
        
        print()
        print("✅ Download complete!")
        print(f"   Models saved to: {checkpoint_dir}")
        print()
        
        # Verify files
        if model_file.exists() and codec_file.exists():
            print("Downloaded files:")
            print(f"   - model.pth: {model_file.stat().st_size / 1e9:.1f}GB")
            print(f"   - codec.pth: {codec_file.stat().st_size / 1e9:.1f}GB")
            return True
        else:
            print("⚠️  Warning: Some files may be missing")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading model: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        return False

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
