import os
import time
from typing import Tuple, Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

__all__ = [
    "detect_cpu_cores",
    "get_safe_core_count",
    "get_cpu_load_percent",
    "get_free_ram_gb",
    "print_hw_summary",
]


def detect_cpu_cores() -> Tuple[int, int]:
    """Return logical and physical CPU core counts (logical, physical)."""
    logical = os.cpu_count() or 1
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False) or logical
        except Exception:
            physical = logical
    else:
        # Fallback heuristic: assume SMT/HT => physical ~= logical/2
        physical = max(1, logical // 2)
    return logical, physical


def get_safe_core_count(reserve: int = 1) -> int:
    """Return a conservative core count leaving *reserve* CPUs idle."""
    logical, physical = detect_cpu_cores()
    return max(1, physical - reserve)


def get_cpu_load_percent(interval: float = 0.1) -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.cpu_percent(interval=interval)
    except Exception:
        return None


def get_free_ram_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except Exception:
        return None


def print_hw_summary():
    logical, physical = detect_cpu_cores()
    load = get_cpu_load_percent()  # type: ignore[arg-type]
    ram = get_free_ram_gb()

    parts = [f"CPUs: {logical} logical / {physical} physical"]
    if load is not None:
        parts.append(f"load {load:.0f}%")
    if ram is not None:
        parts.append(f"free RAM {ram:.1f} GB")
    summary = " – ".join(parts)
    print(summary)
