"""
Shared utilities for BERTopic training scripts.

This module contains common functions used across different BERTopic training
workflows, including GPU availability checks, logging setup, and output directory
management.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Cache GPU functionality test result (test once per script execution)
_GPU_FUNCTIONALITY_TESTED = False
_GPU_FUNCTIONALITY_RESULT = None


def _setup_cuml_library_paths():
    """
    Set up LD_LIBRARY_PATH and preload libraries for cuML/CuPy.
    This replicates the logic from venv/bin/activate.d/setup_cuml_paths.sh
    and preloads libraries so CuPy/CuML can find them at runtime.
    """
    # Find site-packages directory
    site_packages = None
    for path in sys.path:
        if 'site-packages' in path:
            site_packages = Path(path)
            break
    
    if not site_packages:
        return
    
    # Library paths to add (same as activation script)
    cuml_lib_path = site_packages / "libcuml" / "lib64"
    nvrtc_lib_path = site_packages / "nvidia" / "cuda_nvrtc" / "lib"
    
    # Build new library paths
    new_paths = []
    if cuml_lib_path.exists() and cuml_lib_path.is_dir():
        new_paths.append(str(cuml_lib_path))
    
    if nvrtc_lib_path.exists() and nvrtc_lib_path.is_dir():
        new_paths.append(str(nvrtc_lib_path))
    
    # Update LD_LIBRARY_PATH
    if new_paths:
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        # Check if paths are already in LD_LIBRARY_PATH
        needs_update = False
        for new_path in new_paths:
            if new_path not in current_ld_path.split(":"):
                needs_update = True
                break
        
        if needs_update:
            # Prepend new paths to existing LD_LIBRARY_PATH
            new_ld_path = ":".join(new_paths)
            if current_ld_path:
                new_ld_path += ":" + current_ld_path
            os.environ["LD_LIBRARY_PATH"] = new_ld_path
    
    # Preload critical libraries using ctypes so CuPy/CuML can find them
    # This is necessary because Python's ctypes loader may not respect LD_LIBRARY_PATH
    # changes made after process start, and CuPy loads libraries at import time
    try:
        import ctypes
        
        # Preload libnvrtc.so.12 (required by CuPy for JIT compilation)
        nvrtc_lib_file = nvrtc_lib_path / "libnvrtc.so.12"
        if nvrtc_lib_file.exists():
            try:
                ctypes.CDLL(str(nvrtc_lib_file), mode=ctypes.RTLD_GLOBAL)
            except (OSError, AttributeError):
                # Library already loaded or failed, continue
                pass
        
        # Preload libcuml++.so (required by cuML)
        cuml_lib_file = cuml_lib_path / "libcuml++.so"
        if cuml_lib_file.exists():
            try:
                ctypes.CDLL(str(cuml_lib_file), mode=ctypes.RTLD_GLOBAL)
            except (OSError, AttributeError):
                # Library already loaded or failed, continue
                pass
    except Exception:
        # If preloading fails, continue - LD_LIBRARY_PATH should still work
        pass


# Set up library paths at module import
_setup_cuml_library_paths()


def check_gpu_availability() -> bool:
    """
    Early GPU availability check - runs once at module load.
    Returns True if GPU is available and functional, False otherwise.
    Detects driver/runtime mismatches before training starts.

    This function ensures GPU acceleration is used whenever possible,
    falling back to CPU only when GPU is unavailable or non-functional.
    """
    global _GPU_FUNCTIONALITY_TESTED, _GPU_FUNCTIONALITY_RESULT

    if _GPU_FUNCTIONALITY_TESTED:
        return _GPU_FUNCTIONALITY_RESULT

    print("🔍 Checking GPU availability for BERTopic acceleration...")

    # Quick check: Can we even import cuML?
    try:
        from cuml.manifold import UMAP as GPU_UMAP
        from cuml.cluster import HDBSCAN as GPU_HDBSCAN
        print("✓ cuML libraries imported successfully")
    except ImportError as e:
        print(f"⚠️  cuML not available: {e}")
        print("   Falling back to CPU-based UMAP and HDBSCAN")
        _GPU_FUNCTIONALITY_TESTED = True
        _GPU_FUNCTIONALITY_RESULT = False
        return False

    # Check PyTorch CUDA availability for embeddings
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  PyTorch CUDA not available - embeddings will use CPU")
    except ImportError:
        print("⚠️  PyTorch not available - embeddings will use CPU")

    # Try a minimal functionality test with better error reporting
    try:
        import numpy as np
        test_data = np.random.rand(50, 32).astype(np.float32)

        # Test UMAP
        umap_test = GPU_UMAP(n_components=2, random_state=42, verbose=False)
        umap_result = umap_test.fit_transform(test_data)
        print(f"✓ GPU UMAP test passed (shape: {umap_result.shape})")

        # Test HDBSCAN
        hdbscan_test = GPU_HDBSCAN(min_cluster_size=5, verbose=False)
        hdbscan_labels = hdbscan_test.fit_predict(umap_result)
        n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        print(f"✓ GPU HDBSCAN test passed (found {n_clusters} clusters)")

        print("🎉 GPU acceleration is READY! BERTopic will use GPU for UMAP/HDBSCAN")
        _GPU_FUNCTIONALITY_TESTED = True
        _GPU_FUNCTIONALITY_RESULT = True
        return True

    except Exception as e:
        print(f"❌ GPU functionality test failed: {e}")
        print("   This may be due to:")
        print("   - CUDA version mismatch")
        print("   - Driver issues")
        print("   - Memory constraints")
        print("   Falling back to CPU-based algorithms")
        _GPU_FUNCTIONALITY_TESTED = True
        _GPU_FUNCTIONALITY_RESULT = False
        return False


def get_optimal_device() -> str:
    """
    Determine the optimal device for sentence transformer embeddings.
    Always prefers GPU when available, falls back to CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works by creating a small tensor
                test_tensor = torch.tensor([1.0], device='cuda')
                device_name = torch.cuda.get_device_name(0)
                print(f"📍 Using CUDA device for embeddings: {device_name}")
                return "cuda"
            except Exception as e:
                print(f"⚠️  CUDA reported available but failed test: {e}")
                print("📍 Falling back to CPU for embeddings")
                return "cpu"
        else:
            print("📍 Using CPU for embeddings (CUDA not available)")
            return "cpu"
    except ImportError:
        print("📍 Using CPU for embeddings (PyTorch not available)")
        return "cpu"


def setup_output_dirs(out_dir: Path) -> Dict[str, Path]:
    """Create organized output directory structure."""
    dirs = {
        "logs": out_dir / "logs",
        "models": out_dir / "models",
        "embeddings": out_dir / "models" / "embeddings",
        "csv": out_dir / "csv",
        "runs": out_dir / "csv" / "runs",
        "json": out_dir / "json",
        "txt": out_dir / "txt",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(logs_dir: Path) -> logging.Logger:
    """Setup logging to write to logs/ directory."""
    log_path = logs_dir / "train.log"
    logger = logging.getLogger("bertopic_training")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
