"""Check and time the neighbourhood-attention CUDA passes on a card.

    python scripts/check_attention.py                 # the library's own passes
    python scripts/check_attention.py --standalone    # attention.cu alone, built by nvcc

With ``--standalone`` the one translation unit is compiled into a library of
its own and used in place of ``libbartorch``'s passes, and the Python side is
read from this checkout without importing bartorch, so a change to the kernels
can be checked on a card without building BART: it needs ``nvcc`` and torch.

The passes and their gradients are compared with the plain-torch path in
float64, in each element type, and then timed at the sizes a denoiser meets:
forward and forward-backward, against the plain-torch path, with peak memory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

#: The module under test, set by main(): the package's, or with --standalone the file itself,
#: loaded without importing bartorch, whose library may predate these passes.
na = None


def load_module(standalone_file: bool):
    if not standalone_file:
        from bartorch.learning import _neighborhood

        return _neighborhood
    import importlib.util

    path = ROOT / "src" / "bartorch" / "learning" / "_neighborhood.py"
    spec = importlib.util.spec_from_file_location("na3d_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standalone(arch: str | None) -> ctypes.CDLL:
    """``attention.cu`` compiled on its own, with the ctypes signatures of the library's ABI."""
    out = Path(tempfile.mkdtemp()) / "libna3d.so"
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
    command = [
        "nvcc", "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-I", str(ROOT / "src" / "csrc"),
        "-o", str(out), str(ROOT / "src" / "csrc" / "ops" / "attention.cu"),
    ]  # fmt: skip
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(out))
    floats = ctypes.POINTER(ctypes.c_float)
    ints = [
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.bartorch_na3d_built.restype = ctypes.c_int
    for name, first, third in (
        ("gather_dot", ctypes.c_void_p, floats),
        ("gather_sum", floats, ctypes.c_void_p),
        ("scatter_sum", floats, ctypes.c_void_p),
    ):
        fn = getattr(lib, f"bartorch_na3d_{name}")
        fn.restype = ctypes.c_int
        fn.argtypes = [first, ctypes.c_void_p, third, *ints, ctypes.c_void_p]
    return lib


def check(dtype, channels, window, shape=(2, 2, 11, 9, 13)):
    """Largest error of the output and of each gradient against the float64 plain-torch path."""
    torch.manual_seed(0)
    full = (*shape, channels)
    base = [torch.randn(full, device="cuda", dtype=torch.float64) for _ in range(3)]
    bias64 = 0.1 * torch.randn(shape[1], window**3, device="cuda", dtype=torch.float64)
    grad = torch.randn(full, device="cuda", dtype=torch.float64)

    q, k, v = (t.to(dtype).requires_grad_() for t in base)
    bias = bias64.to(dtype).requires_grad_()
    made = na.neighborhood_attention3d(q, k, v, window, bias)
    made.backward(grad.to(dtype))

    rq, rk, rv = (t.clone().requires_grad_() for t in base)
    rb = bias64.clone().requires_grad_()
    groups = shape[0] * shape[1]
    flat = [t.reshape(groups, -1, channels) for t in (rq, rk, rv)]
    want = na._attend_reference(*flat, shape[2:], window, rb, shape[1]).reshape(full)
    want.backward(grad)
    pairs = {"out": (made, want), "dq": (q.grad, rq.grad), "dk": (k.grad, rk.grad)}
    pairs.update({"dv": (v.grad, rv.grad), "dbias": (bias.grad, rb.grad)})
    return {k: float((a.double() - b).abs().max() / b.abs().max()) for k, (a, b) in pairs.items()}


def timed(fn, repeat=5):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat, (torch.cuda.max_memory_allocated() - base) / 1e9


def bench(extent, heads, channels, window, dtype=torch.float16):
    block = na.NeighborhoodAttention3d(heads * channels, heads, window=window).cuda()
    x = torch.randn(1, heads * channels, *(extent,) * 3, device="cuda", requires_grad=True)
    row = {"extent": extent, "heads": heads, "channels/head": channels, "window": window}

    def forward():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            block(x)

    def train():
        with torch.autocast("cuda", dtype=dtype):
            block(x).float().square().mean().backward()

    row["forward s"], row["forward GB"] = timed(forward)
    row["train s"], row["train GB"] = timed(train)
    saved = na.kernels_available
    na.kernels_available = lambda: False
    try:
        row["torch forward s"], row["torch forward GB"] = timed(forward)
    except torch.cuda.OutOfMemoryError:
        row["torch forward s"] = row["torch forward GB"] = "out of memory"
    finally:
        na.kernels_available = saved
        torch.cuda.empty_cache()
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--standalone", action="store_true", help="build attention.cu alone with nvcc")
    ap.add_argument(
        "--arch", help="compute capability without the dot, e.g. 89; the card's by default"
    )
    args = ap.parse_args()
    global na
    na = load_module(args.standalone)
    if args.standalone:
        lib = standalone(args.arch)
        na._library = lambda: lib
        na._float_pointer = lambda address: ctypes.cast(
            ctypes.c_void_p(address), ctypes.POINTER(ctypes.c_float)
        )
    if not na.kernels_available():
        raise SystemExit("no card, or no CUDA passes in this library")
    print(torch.cuda.get_device_name(), flush=True)

    worst = {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for channels in (8, 16, 32, 64):
            for window in (3, 5, 7):
                errors = check(dtype, channels, window)
                key = str(dtype).removeprefix("torch.")
                worst[key] = max(worst.get(key, 0.0), *errors.values())
    print("largest relative error, output and gradients:", json.dumps(worst), flush=True)

    for extent, heads, channels, window in (
        (64, 1, 16, 5), (64, 2, 16, 5), (32, 2, 32, 5),
        (32, 4, 32, 7), (16, 4, 32, 7), (16, 8, 16, 9),
    ):  # fmt: skip
        print(json.dumps(bench(extent, heads, channels, window)), flush=True)


if __name__ == "__main__":
    main()
