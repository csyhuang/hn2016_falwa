#!/usr/bin/env python
"""
Script to clear Numba cache files.

Run this after modifying Numba-decorated functions to force recompilation.

Usage:
    python scripts/clear_numba_cache.py
"""
from typing import Optional
import shutil
from pathlib import Path


def clear_numba_cache(base_dir: Optional[str]=None):
    """Clear all Numba cache files in the project."""
    if base_dir is None:
        # Find the project root (where src/ is)
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)
    
    cleared_count = 0
    
    # Find all __pycache__ directories
    for pycache_dir in base_dir.rglob("__pycache__"):
        # Remove Numba cache files (.nbc, .nbi)
        for pattern in ["*.nbc", "*.nbi"]:
            for cache_file in pycache_dir.glob(pattern):
                print(f"Removing: {cache_file}")
                cache_file.unlink()
                cleared_count += 1
    
    # Also check for .numba_cache directories (older Numba versions)
    for numba_cache in base_dir.rglob(".numba_cache"):
        print(f"Removing directory: {numba_cache}")
        shutil.rmtree(numba_cache)
        cleared_count += 1
    
    if cleared_count == 0:
        print("No Numba cache files found.")
    else:
        print(f"\nCleared {cleared_count} cache file(s)/directory(ies).")
    
    return cleared_count


if __name__ == "__main__":
    clear_numba_cache(
        base_dir="/Users/clarehuang/Library/CloudStorage/Dropbox/GitHub/hn2016_falwa/")

