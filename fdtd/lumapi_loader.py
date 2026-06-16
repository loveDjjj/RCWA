from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def load_lumapi(lumapi_dir: str | Path | None = None):
    if lumapi_dir is None:
        raise ValueError("runtime.lumapi_dir must be set in the FDTD defaults config")
    base = Path(lumapi_dir)
    lumapi_py = base / "lumapi.py"
    if not lumapi_py.exists():
        raise FileNotFoundError(f"lumapi.py not found: {lumapi_py}")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(base))
    spec = importlib.util.spec_from_file_location("lumapi", str(lumapi_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load lumapi spec from {lumapi_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
