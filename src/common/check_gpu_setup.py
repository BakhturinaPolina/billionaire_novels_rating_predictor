#!/usr/bin/env python3
"""
Quick script to verify GPU setup for BERTopic training.
Checks PyTorch CUDA, cuML availability, and basic functionality.
"""

import sys

# Note: LD_LIBRARY_PATH for cuML/CUDA libraries should be set by .venv/bin/activate.d/setup_cuml_paths.sh
# This script expects the virtual environment to be activated with proper library paths

def check_pytorch():
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA not available in PyTorch (CPU mode only)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_cuml():
    """Check cuML availability and basic functionality."""
    try:
        from cuml.manifold import UMAP as GPU_UMAP
        from cuml.cluster import HDBSCAN as GPU_HDBSCAN
        print("✓ cuML imported successfully")
        
        # Quick functionality test
        import numpy as np
        test_data = np.random.rand(100, 64).astype(np.float32)
        
        umap = GPU_UMAP(n_components=5, random_state=42)
        reduced = umap.fit_transform(test_data)
        print(f"✓ GPU UMAP test passed (output shape: {reduced.shape})")
        
        hdbscan = GPU_HDBSCAN(min_cluster_size=5)
        labels = hdbscan.fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"✓ GPU HDBSCAN test passed (found {n_clusters} clusters)")
        
        print("\n🎉 GPU acceleration for UMAP/HDBSCAN is READY!")
        print("   Your BERTopic script will use GPU for dimensionality reduction and clustering.")
        return True
        
    except ImportError as e:
        print("✗ cuML not installed")
        print(f"  Error: {e}")
        print("\n  To install cuML (CUDA 12.x):")
        print("  pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com")
        print("\n  See INSTALL_GPU.md for detailed instructions.")
        return False
    except Exception as e:
        print(f"✗ cuML import succeeded but test failed: {e}")
        return False

def check_cpu_fallback():
    """Check CPU fallback packages."""
    try:
        from umap import UMAP
        import hdbscan
        print("✓ CPU fallback packages (umap-learn, hdbscan) available")
        return True
    except ImportError as e:
        print(f"⚠ CPU fallback package missing: {e}")
        return False

def main():
    print("=" * 60)
    print("BERTopic GPU Setup Verification")
    print("=" * 60)
    print()
    
    pytorch_ok = check_pytorch()
    print()
    
    cuml_ok = check_cuml()
    print()
    
    cpu_fallback_ok = check_cpu_fallback()
    print()
    
    print("=" * 60)
    if cuml_ok and pytorch_ok:
        print("✅ Setup: GPU acceleration enabled")
    elif pytorch_ok and cpu_fallback_ok:
        print("⚠️  Setup: CPU mode (GPU unavailable or cuML not installed)")
    else:
        print("❌ Setup: Issues detected - see errors above")
    print("=" * 60)
    
    return 0 if (cuml_ok or cpu_fallback_ok) else 1

if __name__ == "__main__":
    sys.exit(main())

