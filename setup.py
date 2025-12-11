#!/usr/bin/env python3
"""
Model Setup Script

Downloads the CosyVoice2-0.5B model and sets up the environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA not available, will use CPU (slow)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_vllm():
    """Check vLLM availability."""
    try:
        import vllm
        print(f"✓ vLLM available: v{vllm.__version__}")
        return True
    except ImportError:
        print("✗ vLLM not installed (optional, for acceleration)")
        return False


def download_model(model_id: str, local_dir: str):
    """Download model from ModelScope."""
    print(f"\nDownloading {model_id}...")
    print(f"Target directory: {local_dir}")
    
    try:
        from modelscope import snapshot_download
        snapshot_download(model_id, local_dir=local_dir)
        print(f"✓ Model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        print("\nTrying git clone as fallback...")
        
        try:
            os.makedirs(local_dir, exist_ok=True)
            url = f"https://www.modelscope.cn/{model_id}.git"
            subprocess.run(
                ["git", "clone", url, local_dir],
                check=True
            )
            print(f"✓ Model cloned successfully")
            return True
        except Exception as e2:
            print(f"✗ Git clone also failed: {e2}")
            return False


def clone_cosyvoice():
    """Clone CosyVoice repository."""
    repo_path = Path("CosyVoice")
    
    if repo_path.exists():
        print("✓ CosyVoice repository already exists")
        return True
    
    print("\nCloning CosyVoice repository...")
    try:
        subprocess.run(
            ["git", "clone", "--recursive", 
             "https://github.com/FunAudioLLM/CosyVoice.git"],
            check=True
        )
        print("✓ Repository cloned successfully")
        
        # Update submodules
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd="CosyVoice",
            check=True
        )
        return True
    except Exception as e:
        print(f"✗ Failed to clone: {e}")
        return False


def setup_symlinks():
    """Create symlinks for CosyVoice modules."""
    cosyvoice_path = Path("CosyVoice")
    
    # Check if CosyVoice exists
    if not cosyvoice_path.exists():
        print("✗ CosyVoice directory not found")
        return False
    
    # Create symlinks
    links = [
        ("cosyvoice", cosyvoice_path / "cosyvoice"),
        ("third_party", cosyvoice_path / "third_party"),
    ]
    
    for link_name, target in links:
        link = Path(link_name)
        if link.exists():
            print(f"✓ {link_name} already exists")
            continue
        
        try:
            if sys.platform == "win32":
                # Windows requires special handling for symlinks
                import shutil
                if target.is_dir():
                    # Copy on Windows if symlink fails
                    try:
                        os.symlink(target.absolute(), link, target_is_directory=True)
                    except OSError:
                        print(f"  Creating junction for {link_name}")
                        subprocess.run(
                            ["cmd", "/c", "mklink", "/J", str(link), str(target.absolute())],
                            check=True
                        )
            else:
                os.symlink(target.absolute(), link)
            print(f"✓ Created link: {link_name} -> {target}")
        except Exception as e:
            print(f"✗ Failed to create link {link_name}: {e}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup CosyVoice2 TTS Server")
    parser.add_argument(
        "--model-dir", "-d",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="Directory to download model to"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model download"
    )
    parser.add_argument(
        "--skip-repo",
        action="store_true",
        help="Skip CosyVoice repo cloning"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("CosyVoice2 TTS Server Setup")
    print("=" * 60)
    
    # Check environment
    print("\n[1/4] Checking environment...")
    check_cuda()
    check_vllm()
    
    # Clone CosyVoice
    print("\n[2/4] Setting up CosyVoice repository...")
    if not args.skip_repo:
        clone_cosyvoice()
    else:
        print("  Skipped (--skip-repo)")
    
    # Setup symlinks
    print("\n[3/4] Setting up symlinks...")
    setup_symlinks()
    
    # Download model
    print("\n[4/4] Downloading model...")
    if not args.skip_model:
        download_model("iic/CosyVoice2-0.5B", args.model_dir)
    else:
        print("  Skipped (--skip-model)")
    
    # Create data directory
    os.makedirs("data/speakers", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("""
Next steps:
  1. Install remaining dependencies:
     pip install -r requirements.txt

  2. Start the server:
     python run_server.py

  3. Test the API:
     curl http://localhost:8000/health
     
  4. View API docs:
     http://localhost:8000/docs
""")


if __name__ == "__main__":
    main()
