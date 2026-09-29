"""State the library carries between tests, and the devices a test runs on.

The FINUFFT substitution is a process-wide setting that installs itself the
first time anything needs it, so a test that turns it off, loosens its
tolerance or lets BART's gridder answer would otherwise hand that on to
whatever runs next.  This puts it back.
"""

import pytest
import torch

from bartorch import _cuda, _finufft


@pytest.fixture(autouse=True)
def _finufft_defaults():
    yield
    try:
        _finufft.use_in_tools(True)
    except RuntimeError:
        pass


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not _cuda.available(),
                reason="no CUDA device, or the library was built without CUDA",
            ),
        ),
    ]
)
def device(request):
    """Each device this machine and library can run on."""
    return request.param
