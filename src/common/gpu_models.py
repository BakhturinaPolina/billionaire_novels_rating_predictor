"""
GPU-accelerated model utilities for BERTopic.

This module provides RAPIDS (cuML) implementations of UMAP and HDBSCAN
for GPU acceleration. Always uses GPU versions - no CPU fallback.

Requirements:
    - RAPIDS cuML installed (CUDA 12.x)
    - CUDA-compatible GPU
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

# Global flag to track if RAPIDS is available
_RAPIDS_AVAILABLE = None
_RAPIDS_CHECKED = False


def check_rapids_availability() -> bool:
    """
    Check if RAPIDS (cuML) is available and functional.
    
    Returns:
        True if RAPIDS is available and working, False otherwise.
    """
    global _RAPIDS_AVAILABLE, _RAPIDS_CHECKED
    
    if _RAPIDS_CHECKED:
        return _RAPIDS_AVAILABLE
    
    _RAPIDS_CHECKED = True
    
    try:
        import cudf
        from cuml.manifold import UMAP
        from cuml.cluster import HDBSCAN
        
        # Quick functionality test
        import numpy as np
        test_data = np.random.rand(100, 64).astype(np.float32)
        
        # Test UMAP
        umap_test = UMAP(n_components=5, random_state=42)
        _ = umap_test.fit_transform(test_data)
        
        # Test HDBSCAN
        hdbscan_test = HDBSCAN(min_cluster_size=5)
        _ = hdbscan_test.fit_predict(test_data)
        
        _RAPIDS_AVAILABLE = True
        logger.info("✅ RAPIDS (cuML) is available and functional")
        print("✅ RAPIDS (cuML) is available and functional")
        return True
        
    except ImportError as e:
        _RAPIDS_AVAILABLE = False
        logger.error(f"❌ RAPIDS (cuML) not available: {e}")
        print(f"❌ RAPIDS (cuML) not available: {e}")
        print("   Install with: pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com")
        return False
    except Exception as e:
        _RAPIDS_AVAILABLE = False
        logger.error(f"❌ RAPIDS (cuML) import succeeded but test failed: {e}")
        print(f"❌ RAPIDS (cuML) test failed: {e}")
        return False


def get_gpu_umap_hdbscan(
    n_neighbors: int = 15,
    n_components: int = 5,
    min_dist: float = 0.0,
    umap_metric: str = "cosine",
    hdb_min_cluster_size: int = 10,
    hdb_min_samples: int = 5,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    seed: int = 42,
    verbose: bool = False
) -> Tuple:
    """
    Get GPU-accelerated UMAP and HDBSCAN models using RAPIDS (cuML).
    
    This function ALWAYS uses RAPIDS - no CPU fallback.
    Raises an error if RAPIDS is not available.
    
    Args:
        n_neighbors: UMAP n_neighbors parameter
        n_components: UMAP n_components parameter
        min_dist: UMAP min_dist parameter
        umap_metric: UMAP metric parameter
        hdb_min_cluster_size: HDBSCAN min_cluster_size parameter
        hdb_min_samples: HDBSCAN min_samples parameter
        hdbscan_metric: HDBSCAN metric parameter
        hdbscan_cluster_selection_method: HDBSCAN cluster_selection_method
        seed: Random seed for reproducibility
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (umap_model, hdbscan_model)
        
    Raises:
        ImportError: If RAPIDS (cuML) is not available
        RuntimeError: If GPU is not available or RAPIDS test fails
    """
    # Check RAPIDS availability first
    if not check_rapids_availability():
        raise ImportError(
            "RAPIDS (cuML) is required but not available. "
            "Install with: pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com"
        )
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for RAPIDS but not available. "
            "Ensure you have a CUDA-compatible GPU and CUDA drivers installed."
        )
    
    # Import RAPIDS models
    from cuml.manifold import UMAP
    from cuml.cluster import HDBSCAN
    
    logger.info("🚀 Using GPU-accelerated models (RAPIDS cuML)")
    print("🚀 Using GPU-accelerated models (RAPIDS cuML)")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    logger.info(f"   UMAP params: n_neighbors={n_neighbors}, n_components={n_components}, "
                f"min_dist={min_dist}, metric={umap_metric}")
    logger.info(f"   HDBSCAN params: min_cluster_size={hdb_min_cluster_size}, "
                f"min_samples={hdb_min_samples}, metric={hdbscan_metric}, "
                f"cluster_selection_method={hdbscan_cluster_selection_method}")
    
    print(f"   UMAP: n_neighbors={n_neighbors}, n_components={n_components}, "
          f"min_dist={min_dist}, metric={umap_metric}")
    print(f"   HDBSCAN: min_cluster_size={hdb_min_cluster_size}, "
          f"min_samples={hdb_min_samples}, metric={hdbscan_metric}")
    
    # Create UMAP model
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=umap_metric,
        random_state=seed,
        verbose=verbose
    )
    
    # Create HDBSCAN model
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdb_min_cluster_size,
        min_samples=hdb_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        prediction_data=True,
        gen_min_span_tree=True
    )
    
    logger.info("✅ GPU models initialized successfully")
    print("✅ GPU models initialized successfully")
    
    return umap_model, hdbscan_model


def check_cuda_available() -> bool:
    """
    Check if CUDA is available in PyTorch.
    
    Returns:
        True if CUDA is available, False otherwise.
    """
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available in PyTorch")
        print("⚠️  CUDA not available in PyTorch")
        return False
    
    logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    return True


def print_gpu_status():
    """Print comprehensive GPU and RAPIDS status."""
    print("\n" + "=" * 80)
    print("GPU & RAPIDS Status Check")
    print("=" * 80)
    
    # Check PyTorch CUDA
    print("\n[PyTorch CUDA]")
    cuda_ok = check_cuda_available()
    
    # Check RAPIDS
    print("\n[RAPIDS cuML]")
    rapids_ok = check_rapids_availability()
    
    print("\n" + "=" * 80)
    if cuda_ok and rapids_ok:
        print("✅ GPU acceleration READY")
    elif cuda_ok:
        print("⚠️  CUDA available but RAPIDS not available")
    else:
        print("❌ GPU acceleration NOT available")
    print("=" * 80 + "\n")
    
    return cuda_ok and rapids_ok

