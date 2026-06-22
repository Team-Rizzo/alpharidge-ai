"""configure_determinism() pins cuDNN/cuBLAS/RNG for reproducible neural output."""
import os


def test_configure_sets_cudnn_and_cublas():
    import torch
    from alpharidge_ai.analyzer.determinism import configure_determinism
    configure_determinism()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")  # set for cuBLAS reproducibility


def test_idempotent_and_does_not_raise():
    from alpharidge_ai.analyzer.determinism import configure_determinism
    configure_determinism()
    configure_determinism()  # second call is a no-op, must not raise
