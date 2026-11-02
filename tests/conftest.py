import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def pytest_collection_modifyitems(config, items):
    if not TORCH_AVAILABLE or torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _seed():
    if TORCH_AVAILABLE:
        torch.manual_seed(0)  # fixed seed so failures reproduce
