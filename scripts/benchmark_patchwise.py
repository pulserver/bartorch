"""Time and peak card memory of a U-Net applied to a host-resident volume.

One denoiser call over a complex multi-channel volume, the way an iteration of
a learned reconstruction makes it: the whole volume moved to the card, and
patch by patch with and without overlapped copies.  Run on the card the
reconstruction is to run on::

    python scripts/benchmark_patchwise.py --size 160 --channels 10 --patch 64 --batch 2
"""

from __future__ import annotations

import argparse
import time

import torch

from bartorch import learning


def measure(call, repeats: int):
    """Best wall time of ``call`` and the peak memory allocated on the card during it."""
    call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best, torch.cuda.max_memory_allocated() / 2**20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=128, help="voxels along each axis")
    parser.add_argument("--channels", type=int, default=5, help="complex channels")
    parser.add_argument("--patch", type=int, default=64, help="patch extent along each axis")
    parser.add_argument("--batch", type=int, default=2, help="patches per network call")
    parser.add_argument(
        "--widths", type=int, nargs="+", default=[32, 64, 128, 256], help="U-Net widths"
    )
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this measures a card, and none is available")

    device = torch.device("cuda")
    net = learning.UNet(2 * arguments.channels, spatial=3, widths=tuple(arguments.widths))
    net.to(device).eval()
    volume = torch.randn(
        1, arguments.channels, *(arguments.size,) * 3, dtype=torch.complex64
    ).pin_memory()
    weights = sum(p.numel() for p in net.parameters())
    print(
        f"{torch.cuda.get_device_name(device)}; volume {tuple(volume.shape)} complex64 "
        f"({volume.numel() * 8 / 2**20:.0f} MiB on the host); "
        f"U-Net {weights / 1e6:.2f} M weights"
    )

    dtype = learning.Patchwise(net, (arguments.patch,) * 3, device=device).dtype
    cases = {"whole volume on the card": learning.ComplexNet(net, channels=1)}
    for overlap in (False, True):
        patchwise = learning.Patchwise(
            net, (arguments.patch,) * 3, device=device, batch=arguments.batch, overlap=overlap
        )
        name = f"patches of {arguments.patch}, {'overlapped' if overlap else 'serial'} copies"
        cases[name] = learning.ComplexNet(patchwise, channels=1)

    for name, denoiser in cases.items():
        if denoiser.net is net:

            def call(denoiser=denoiser):
                with torch.autocast("cuda", dtype=dtype):
                    return denoiser(volume.to(device, non_blocking=True)).cpu()

        else:

            def call(denoiser=denoiser):
                return denoiser(volume)

        try:
            with torch.no_grad():
                seconds, peak = measure(call, arguments.repeats)
        except torch.OutOfMemoryError:
            print(f"{name:<40} out of memory")
            torch.cuda.empty_cache()
            continue
        print(f"{name:<40} {seconds:7.2f} s   peak {peak:8.0f} MiB")


if __name__ == "__main__":
    main()
