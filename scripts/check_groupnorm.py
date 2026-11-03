"""Check and time the fused group norm (group norm, scale and shift, SiLU) on a card.

    python scripts/check_groupnorm.py                 # the library's own pass
    python scripts/check_groupnorm.py --standalone    # groupnorm.cu alone, built by nvcc

With ``--standalone`` the one translation unit is compiled into a library of
its own and used in place of ``libbartorch``'s pass, and the Python side is
read from this checkout without importing bartorch, so the kernel can be
checked on a card without building BART: it needs ``nvcc`` and torch.

The pass is compared with float64 arithmetic in each element type, then timed
at the activations of a diffusion UNet's levels against the way such a network
runs it in eager torch (a float32 group norm, a cast back, the scale and shift,
SiLU) and against a float16 group norm alone, with the memory floor (one read
for the statistics, one read and one write for the rest) for reference.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess  # nosec B404 - runs nvcc, a fixed command
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

gn = None


def load_module(standalone_file: bool):
    if not standalone_file:
        from bartorch.learning import _groupnorm

        return _groupnorm
    import importlib.util

    path = ROOT / "src" / "bartorch" / "learning" / "_groupnorm.py"
    spec = importlib.util.spec_from_file_location("gn_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standalone(arch: str | None) -> ctypes.CDLL:
    """``groupnorm.cu`` compiled on its own, with the ctypes signatures of the library's ABI."""
    out = Path(tempfile.mkdtemp()) / "libgn.so"
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
    command = [
        "nvcc", "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-I", str(ROOT / "src" / "csrc"),
        "-o", str(out), str(ROOT / "src" / "csrc" / "ops" / "groupnorm.cu"),
    ]  # fmt: skip
    subprocess.run(command, check=True)  # nosec B603 - nvcc on this checkout's source
    lib = ctypes.CDLL(str(out))
    lib.bartorch_gn_built.restype = ctypes.c_int
    lib.bartorch_gn_workspace.restype = ctypes.c_int64
    lib.bartorch_gn_workspace.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    v = ctypes.c_void_p
    lib.bartorch_gn_silu.restype = ctypes.c_int
    lib.bartorch_gn_silu.argtypes = [
        v, v, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        v, v, v, v, ctypes.c_float, ctypes.c_int, ctypes.c_int, v, v,
    ]  # fmt: skip
    return lib


def exact(x, groups, weight, bias, scale, shift):
    n, c = x.shape[:2]
    g = x.double().reshape(n, groups, -1)
    g = (g - g.mean(-1, keepdim=True)) / (g.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    shape = (1, c, *([1] * (x.ndim - 2)))
    h = g.reshape(x.shape) * weight.double().reshape(shape) + bias.double().reshape(shape)
    shape = (n, c, *([1] * (x.ndim - 2)))
    return F.silu(h * (1 + scale.double().reshape(shape)) + shift.double().reshape(shape))


def check(dtype, shape, groups=32):
    torch.manual_seed(0)
    n, c = shape[:2]
    x = (2 + 3 * torch.randn(shape, device="cuda")).to(dtype)
    weight, bias = torch.randn(c, device="cuda"), torch.randn(c, device="cuda")
    scale, shift = 0.3 * torch.randn(n, c, device="cuda"), torch.randn(n, c, device="cuda")
    with torch.no_grad():
        made = gn.group_norm_silu(x, groups, weight, bias, scale, shift)
    want = exact(x, groups, weight, bias, scale, shift)
    return float((made.double() - want).abs().max() / want.abs().max())


def timed(fn, repeat=20):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def bench(channels, extent, dtype=torch.float16, groups=32):
    x = torch.randn(1, channels, *(extent,) * 3, device="cuda", dtype=dtype)
    weight, bias = torch.ones(channels, device="cuda"), torch.zeros(channels, device="cuda")
    scale, shift = torch.zeros(1, channels, device="cuda"), torch.zeros(1, channels, device="cuda")
    s5, t5 = scale.reshape(1, -1, 1, 1, 1).to(dtype), shift.reshape(1, -1, 1, 1, 1).to(dtype)

    def eager():  # guided diffusion's GroupNorm32, then FiLM, then SiLU
        h = F.group_norm(x.float(), groups, weight, bias).type(x.dtype)
        return F.silu(h * (1 + s5) + t5)

    def half_norm():
        return F.silu(F.group_norm(x, groups, weight.to(dtype), bias.to(dtype)) * (1 + s5) + t5)

    def fused():
        return gn.group_norm_silu(x, groups, weight, bias, scale, shift)

    with torch.no_grad():
        row = {"channels": channels, "extent": extent}
        row["eager fp32 norm ms"] = 1e3 * timed(eager)
        row["eager fp16 norm ms"] = 1e3 * timed(half_norm)
        row["fused ms"] = 1e3 * timed(fused)
    row["bytes moved MB (floor)"] = 3 * x.numel() * x.element_size() / 1e6
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--standalone", action="store_true", help="build groupnorm.cu alone with nvcc")
    ap.add_argument(
        "--arch", help="compute capability without the dot, e.g. 86; the card's by default"
    )
    args = ap.parse_args()
    global gn
    gn = load_module(args.standalone)
    if args.standalone:
        lib = standalone(args.arch)
        gn._library = lambda: lib
    if not gn.kernels_available():
        raise SystemExit("no card, or no CUDA pass in this library")
    print(torch.cuda.get_device_name(), flush=True)

    worst = {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in (
            (2, 64, 9, 10, 11),
            (1, 128, 32, 32, 32),
            (1, 64, 64, 64, 64),
            (3, 512, 8, 8, 8),
        ):
            key = str(dtype).removeprefix("torch.")
            worst[key] = max(worst.get(key, 0.0), check(dtype, shape))
    print("largest relative error against float64:", json.dumps(worst), flush=True)

    for channels, extent in (
        (64, 64),
        (128, 64),
        (128, 32),
        (256, 32),
        (256, 16),
        (512, 16),
        (512, 8),
    ):
        print(json.dumps(bench(channels, extent)), flush=True)


if __name__ == "__main__":
    main()
