"""
GPU Memory Monitoring and Management Utilities.

This module provides utilities for monitoring and managing GPU memory during
BERTopic training and optimization. Includes memory tracking, cleanup, and
OOM error handling.
"""

import gc
import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_usage() -> Dict:
    """
    Get comprehensive GPU memory usage statistics.
    
    Returns:
        Dictionary with memory statistics:
        - available: bool - Whether CUDA is available
        - allocated_gb: float - Currently allocated memory (GB)
        - reserved_gb: float - Currently reserved memory (GB)
        - total_gb: float - Total GPU memory (GB)
        - free_gb: float - Free memory (GB)
        - utilization_pct: float - Memory utilization percentage
    """
    if not torch.cuda.is_available():
        return {
            'available': False,
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'total_gb': 0.0,
            'free_gb': 0.0,
            'utilization_pct': 0.0
        }
    
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    
    utilization_pct = (reserved / total) * 100 if total > 0 else 0.0
    
    return {
        'available': True,
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': free,
        'utilization_pct': utilization_pct
    }


def print_gpu_memory_usage(label: str = "GPU Memory", verbose: bool = True) -> Dict:
    """
    Print GPU memory usage statistics.
    
    Args:
        label: Label for the memory report
        verbose: Whether to print to console
        
    Returns:
        Memory usage dictionary
    """
    mem = get_gpu_memory_usage()
    
    if not mem['available']:
        if verbose:
            print(f"{label}: CUDA not available")
        return mem
    
    if verbose:
        print(f"{label}:")
        print(f"   Allocated: {mem['allocated_gb']:.2f} GB")
        print(f"   Reserved: {mem['reserved_gb']:.2f} GB")
        print(f"   Free: {mem['free_gb']:.2f} GB")
        print(f"   Total: {mem['total_gb']:.2f} GB")
        print(f"   Utilization: {mem['utilization_pct']:.1f}%")
    
    return mem


def cleanup_gpu_memory(verbose: bool = True) -> Dict:
    """
    Comprehensive GPU memory cleanup.
    
    Clears PyTorch CUDA cache, synchronizes, and runs Python garbage collection.
    
    Args:
        verbose: Whether to print cleanup information
        
    Returns:
        Dictionary with cleanup results:
        - success: bool - Whether cleanup succeeded
        - before_gb: float - Memory before cleanup (GB)
        - after_gb: float - Memory after cleanup (GB)
        - freed_gb: float - Memory freed (GB)
    """
    if not torch.cuda.is_available():
        if verbose:
            print("🧹 Memory cleanup: CUDA not available")
        return {
            'success': False,
            'reason': 'CUDA not available',
            'before_gb': 0.0,
            'after_gb': 0.0,
            'freed_gb': 0.0
        }
    
    # Get memory before cleanup
    before = get_gpu_memory_usage()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Python garbage collection
    gc.collect()
    
    # Force another cache clear after GC
    torch.cuda.empty_cache()
    
    # Get memory after cleanup
    after = get_gpu_memory_usage()
    
    freed = before['allocated_gb'] - after['allocated_gb']
    
    if verbose:
        print(f"🧹 Memory cleanup:")
        print(f"   Before: {before['allocated_gb']:.2f} GB allocated, {before['reserved_gb']:.2f} GB reserved")
        print(f"   After: {after['allocated_gb']:.2f} GB allocated, {after['reserved_gb']:.2f} GB reserved")
        if freed > 0:
            print(f"   ✓ Freed: {freed:.2f} GB")
        else:
            print(f"   (No memory freed - may be in use)")
    
    logger.info(f"GPU memory cleanup: {before['allocated_gb']:.2f} GB -> {after['allocated_gb']:.2f} GB (freed {freed:.2f} GB)")
    
    return {
        'success': True,
        'before_gb': before['allocated_gb'],
        'after_gb': after['allocated_gb'],
        'freed_gb': freed
    }


def check_memory_available(required_gb: float = 1.0) -> Tuple[bool, Dict]:
    """
    Check if sufficient GPU memory is available.
    
    Args:
        required_gb: Required memory in GB
        
    Returns:
        Tuple of (is_available, memory_info)
    """
    mem = get_gpu_memory_usage()
    
    if not mem['available']:
        return False, mem
    
    available = mem['free_gb'] >= required_gb
    
    return available, mem


def log_memory_usage(log_file: Optional[Path] = None, label: str = "Memory Check") -> Dict:
    """
    Log memory usage to file and return statistics.
    
    Args:
        log_file: Path to log file (optional)
        label: Label for this memory check
        
    Returns:
        Memory usage dictionary
    """
    mem = get_gpu_memory_usage()
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'label': label,
        'memory': mem
    }
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to log file (JSON lines format)
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Memory usage logged to {log_file}: {label} - {mem['allocated_gb']:.2f} GB allocated")
    
    logger.debug(f"Memory usage logged: {label} - {mem['allocated_gb']:.2f} GB allocated")
    
    return mem


def detect_memory_leak(memory_history: List[Dict], threshold_increase_gb: float = 1.0) -> Optional[Dict]:
    """
    Detect potential memory leaks by analyzing memory history.
    
    Args:
        memory_history: List of memory usage dictionaries (from log_memory_usage)
        threshold_increase_gb: Threshold for memory increase to consider a leak (GB)
        
    Returns:
        Dictionary with leak detection results, or None if no leak detected
    """
    if len(memory_history) < 2:
        return None
    
    # Get initial and final memory
    initial = memory_history[0]['memory']
    final = memory_history[-1]['memory']
    
    if not initial.get('available') or not final.get('available'):
        return None
    
    increase = final['allocated_gb'] - initial['allocated_gb']
    
    if increase > threshold_increase_gb:
        return {
            'leak_detected': True,
            'initial_gb': initial['allocated_gb'],
            'final_gb': final['allocated_gb'],
            'increase_gb': increase,
            'threshold_gb': threshold_increase_gb,
            'warning': f"Potential memory leak detected: {increase:.2f} GB increase over {len(memory_history)} measurements"
        }
    
    return {
        'leak_detected': False,
        'increase_gb': increase
    }


def handle_oom_error(error: Exception, current_batch_size: Optional[int] = None) -> Dict:
    """
    Handle CUDA Out of Memory errors.
    
    Args:
        error: The OOM exception
        current_batch_size: Current batch size (if applicable)
        
    Returns:
        Dictionary with recovery suggestions:
        - recovered: bool - Whether recovery is possible
        - suggested_batch_size: int - Suggested batch size reduction
        - cleanup_performed: bool - Whether cleanup was performed
    """
    if not isinstance(error, RuntimeError) or "out of memory" not in str(error).lower():
        return {
            'recovered': False,
            'reason': 'Not an OOM error',
            'suggested_batch_size': current_batch_size,
            'cleanup_performed': False
        }
    
    print("⚠️ CUDA Out of Memory error detected!")
    print("   Attempting recovery...")
    
    # Perform cleanup
    cleanup_result = cleanup_gpu_memory(verbose=True)
    
    # Suggest batch size reduction
    suggested_batch_size = None
    if current_batch_size:
        # Reduce by 50%
        suggested_batch_size = max(1, int(current_batch_size * 0.5))
        print(f"   Suggested batch size reduction: {current_batch_size} -> {suggested_batch_size}")
    
    # Check if recovery is possible
    mem = get_gpu_memory_usage()
    recovered = mem['free_gb'] > 0.5  # At least 0.5 GB free
    
    if recovered:
        print("   ✓ Recovery possible - sufficient memory available after cleanup")
    else:
        print("   ❌ Recovery not possible - insufficient memory available")
        print("   Consider:")
        print("     - Reducing batch size further")
        print("     - Reducing dataset size")
        print("     - Using a smaller model")
    
    return {
        'recovered': recovered,
        'suggested_batch_size': suggested_batch_size,
        'cleanup_performed': True,
        'free_memory_gb': mem['free_gb']
    }


class MemoryTracker:
    """Context manager for tracking peak GPU memory usage."""
    
    def __init__(self, label: str):
        self.label = label
        self.initial = None
        self.cuda_available = torch.cuda.is_available()
    
    def __enter__(self):
        if self.cuda_available:
            # Reset peak stats
            torch.cuda.reset_peak_memory_stats(0)
            # Get initial memory
            self.initial = get_gpu_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cuda_available:
            print(f"📊 Peak memory usage ({self.label}): CUDA not available")
            return False
        
        # Get peak memory
        peak_allocated = torch.cuda.max_memory_allocated(0) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(0) / 1e9
        
        # Get final memory
        final = get_gpu_memory_usage()
        
        peak_info = {
            'available': True,
            'label': self.label,
            'initial_allocated_gb': self.initial['allocated_gb'],
            'peak_allocated_gb': peak_allocated,
            'peak_reserved_gb': peak_reserved,
            'final_allocated_gb': final['allocated_gb'],
            'memory_increase_gb': peak_allocated - self.initial['allocated_gb']
        }
        
        print(f"📊 Peak memory usage ({self.label}):")
        print(f"   Initial: {self.initial['allocated_gb']:.2f} GB")
        print(f"   Peak: {peak_allocated:.2f} GB (allocated), {peak_reserved:.2f} GB (reserved)")
        print(f"   Final: {final['allocated_gb']:.2f} GB")
        print(f"   Increase: {peak_info['memory_increase_gb']:.2f} GB")
        
        logger.info(f"Peak memory: {peak_allocated:.2f} GB for {self.label}")
        
        return False  # Don't suppress exceptions


def track_memory_peak(label: str = "Operation") -> MemoryTracker:
    """
    Track peak memory usage during an operation.
    
    Use as a context manager:
        with track_memory_peak("My Operation"):
            # Your code here
            pass
    
    Args:
        label: Label for this operation
        
    Returns:
        MemoryTracker context manager
    """
    return MemoryTracker(label)


def enforce_memory_limit(required_gb: float = 2.0, abort_on_insufficient: bool = False) -> Tuple[bool, Dict]:
    """
    Check if sufficient memory is available and optionally abort.
    
    Args:
        required_gb: Required memory in GB
        abort_on_insufficient: Whether to abort if memory is insufficient
        
    Returns:
        Tuple of (is_sufficient, memory_info)
    """
    available, mem = check_memory_available(required_gb)
    
    if not available:
        print(f"⚠️ WARNING: Insufficient GPU memory!")
        print(f"   Required: {required_gb:.2f} GB")
        print(f"   Available: {mem['free_gb']:.2f} GB")
        print(f"   Total: {mem['total_gb']:.2f} GB")
        print(f"   Utilization: {mem['utilization_pct']:.1f}%")
        
        if abort_on_insufficient:
            raise RuntimeError(
                f"Insufficient GPU memory: {mem['free_gb']:.2f} GB available, "
                f"{required_gb:.2f} GB required. Consider reducing batch size or dataset size."
            )
        else:
            print("   Continuing anyway (abort_on_insufficient=False)")
    
    return available, mem


def adjust_batch_size(current_batch_size: int, required_memory_per_batch_gb: float = 0.01) -> int:
    """
    Dynamically adjust batch size based on available GPU memory.
    
    Args:
        current_batch_size: Current batch size
        required_memory_per_batch_gb: Estimated memory per batch in GB
        
    Returns:
        Adjusted batch size (may be smaller than current)
    """
    if not torch.cuda.is_available():
        return current_batch_size
    
    mem = get_gpu_memory_usage()
    available_gb = mem['free_gb']
    
    # Calculate maximum batch size based on available memory
    # Reserve 1 GB for overhead
    usable_memory_gb = max(0.5, available_gb - 1.0)
    max_batch_size = int(usable_memory_gb / required_memory_per_batch_gb)
    
    # Adjust batch size if needed
    if max_batch_size < current_batch_size:
        print(f"📉 Adjusting batch size: {current_batch_size} -> {max_batch_size}")
        print(f"   Available memory: {available_gb:.2f} GB")
        print(f"   Estimated memory per batch: {required_memory_per_batch_gb:.4f} GB")
        return max(1, max_batch_size)
    
    return current_batch_size

