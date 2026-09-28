"""One OpenMP runtime in the process, whichever of torch and bartorch comes first.

LLVM's runtime ends the process when a second copy of itself initialises, so
on macOS and Windows the library is linked against the runtime torch carries
(cmake/openmp.cmake).  Each case runs in a fresh interpreter, exercises
torch's thread pool and FINUFFT's together, and reads the images the process
has loaded.  On Linux the toolchain's runtime is used and GNU's tolerates a
second copy, so there the transform is exercised and the count only reported.
"""

import subprocess
import sys
import textwrap

import pytest

PROBE = textwrap.dedent(
    """
    import ctypes, os, sys

    def openmp_runtimes():
        names = ("libomp", "libiomp5", "libgomp", "vcomp")
        if sys.platform == "darwin":
            dyld = ctypes.CDLL(None)
            dyld._dyld_image_count.restype = ctypes.c_uint32
            dyld._dyld_get_image_name.restype = ctypes.c_char_p
            dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            images = [dyld._dyld_get_image_name(i).decode()
                      for i in range(dyld._dyld_image_count())]
        elif sys.platform == "win32":
            from ctypes import wintypes
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.EnumProcessModules.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD)]
            psapi.GetModuleFileNameExW.argtypes = [
                wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
            process = kernel32.GetCurrentProcess()
            modules = (wintypes.HMODULE * 4096)()
            needed = wintypes.DWORD()
            psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed))
            images = []
            for handle in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
                name = ctypes.create_unicode_buffer(1024)
                psapi.GetModuleFileNameExW(process, handle, name, 1024)
                images.append(name.value)
        else:
            with open("/proc/self/maps") as maps:
                images = sorted({line.split()[-1] for line in maps if "/" in line})
        return sorted({p for p in images if os.path.basename(p).lower().startswith(names)})

    order = sys.argv[1]
    if order == "torch-first":
        import torch
        import bartorch
    else:
        import bartorch
        import torch
    import bartorch.tools as bt
    from bartorch import _finufft

    torch.set_num_threads(4)
    bartorch.set_num_threads(4)
    a = torch.randn(512, 512)
    for _ in range(4):
        a = torch.tanh(a @ a.T / 512)

    n = 96
    traj = bt.traj(x=n, y=64, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    _finufft.reset_counters()
    for _ in range(4):
        y = bartorch.nufft(img, traj)
    assert _finufft.operators_built()[0] == 4, _finufft.operators_built()
    assert float(y.abs().max()) > 0
    a = torch.tanh(a @ a.T / 512)

    for path in openmp_runtimes():
        print("OPENMP", path)
    """
)


def _runtimes(order: str) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-c", PROBE, order], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, (
        f"the process ended with {done.returncode}\n{done.stdout}\n{done.stderr}"
    )
    return [
        line[len("OPENMP ") :] for line in done.stdout.splitlines() if line.startswith("OPENMP")
    ]


@pytest.mark.parametrize("order", ["torch-first", "bartorch-first"])
def test_threaded_torch_and_threaded_finufft_share_one_openmp_runtime(order):
    runtimes = _runtimes(order)
    print(order, runtimes)
    if sys.platform in ("darwin", "win32"):
        assert len(runtimes) == 1, f"{len(runtimes)} OpenMP runtimes loaded: {runtimes}"
        assert "torch" in runtimes[0].replace("\\", "/").lower(), (
            f"the one runtime loaded is not torch's: {runtimes[0]}"
        )


def test_loading_modifies_no_installed_file():
    """No binary in torch or in this package is rewritten to make the pair one runtime."""
    import hashlib
    from pathlib import Path

    import torch

    from bartorch._lib import library_path

    roots = [Path(torch.__file__).resolve().parent / "lib", Path(library_path()).resolve().parent]

    def digests():
        found = {}
        for root in roots:
            for path in root.glob("*"):
                if path.is_file() and (
                    path.suffix in (".so", ".dylib", ".dll") or ".so." in path.name
                ):
                    found[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return found

    before = digests()
    assert before
    _runtimes("bartorch-first")
    assert digests() == before
