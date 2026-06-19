"""Process-wide determinism configuration for the off-LLM ML models.

Miner and validator must produce near-identical per-asset sentiment so batches
clear the validator's agreement gate. Neural inference is never bit-identical
across *different* GPUs, but within a given install most run-to-run jitter is
avoidable — cuDNN autotuning can pick different kernels under concurrent GPU
memory pressure, cuBLAS GEMMs use a nondeterministic workspace, and CUDA RNG is
unseeded. Pinning these removes that jitter at NO quality cost: the math is
unchanged, only the algorithm selection and seeds are fixed.

Call `configure_determinism()` once at process start, before any model forward
pass (it is idempotent). `CUBLAS_WORKSPACE_CONFIG` must be set before CUDA
initialises, so the launch scripts also export it; the os.environ line here is a
fallback for non-script entry points.
"""
import os
import random

_CONFIGURED = False


def configure_determinism() -> None:
    """Pin cuDNN/cuBLAS/RNG to deterministic, reproducible behaviour. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except Exception:
        pass

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    # No autotuning/benchmarking: cuDNN picks a single fixed deterministic kernel
    # instead of choosing by benchmark (which varies with free memory under load).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Best-effort deterministic kernels everywhere else; warn_only so an op lacking
    # a deterministic implementation degrades to a warning instead of crashing.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
