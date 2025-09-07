import os
from functools import lru_cache
from typing import Optional


DEFAULT_DATASET_DIRNAME = "Building Data Genome Project 2 dataset"


def find_dataset_dir(start_dir: str = ".") -> Optional[str]:
    """Attempt to locate the BDG2 dataset directory.

    Searches current dir and one level down; returns absolute path or None.
    """
    candidates = [
        os.path.join(start_dir, DEFAULT_DATASET_DIRNAME),
        os.path.join(os.path.dirname(os.path.abspath(start_dir)), DEFAULT_DATASET_DIRNAME),
    ]
    for c in candidates:
        if os.path.isdir(c):
            # Minimal validation: expect a couple CSVs to exist
            if any(os.path.isfile(os.path.join(c, f)) for f in ("electricity.csv", "weather.csv", "metadata.csv")):
                return os.path.abspath(c)
    # Fallback: scan a bit deeper (maxdepth ~2)
    for root, dirs, files in os.walk(start_dir):
        if os.path.basename(root) == DEFAULT_DATASET_DIRNAME:
            return os.path.abspath(root)
    return None


@lru_cache(maxsize=1)
def get_dataset_dir(explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path and os.path.isdir(explicit_path):
        return os.path.abspath(explicit_path)
    env_override = os.getenv("BDG2_DATASET_DIR")
    if env_override and os.path.isdir(env_override):
        return os.path.abspath(env_override)
    return find_dataset_dir()


def memory_cache(ttl_seconds: int = 0):
    """Simple cache decorator placeholder (Streamlit provides caching).

    In non-Streamlit contexts, this acts as a no-op.
    """
    def dec(fn):
        return fn
    return dec

