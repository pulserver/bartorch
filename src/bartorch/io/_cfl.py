"""Reading and writing BART CFL file pairs."""

from __future__ import annotations

import numpy as np

_BART_DIMS = 16


def readcfl(name: str) -> np.ndarray:
    """Read the CFL pair ``name.hdr`` / ``name.cfl`` into a NumPy array.

    Parameters
    ----------
    name : str
        Path without the extension.

    Returns
    -------
    numpy.ndarray
        ``complex64``, Fortran order, with BART's dimension vector as its shape
        (``x`` first), not the reversed C-order shape of a bartorch tensor;
        trailing singleton dimensions are dropped.

    Raises
    ------
    ValueError
        If the data file does not hold as many elements as the header states.
    """
    with open(name + ".hdr") as f:
        lines = [ln.strip() for ln in f if not ln.startswith("#")]
    dims = [int(x) for x in lines[0].split()]

    while dims and dims[-1] == 1:
        dims.pop()

    n = int(np.prod(dims)) if dims else 1
    arr = np.fromfile(name + ".cfl", dtype=np.complex64)

    if arr.size != n:
        raise ValueError(f"CFL data size mismatch: header says {n} elements, file has {arr.size}.")

    return arr.reshape(dims, order="F")


def writecfl(name: str, array: np.ndarray) -> None:
    """Write ``array`` as the CFL pair ``name.hdr`` / ``name.cfl``.

    Parameters
    ----------
    name : str
        Path without the extension.
    array : numpy.ndarray
        Array whose shape is BART's dimension vector (``x`` first); written as
        ``complex64`` in Fortran order.  A C-order tensor is written in BART's
        order by passing ``tensor.numpy().T``.
    """
    array = np.asarray(array, dtype=np.complex64)
    dims = list(array.shape)
    padded = dims + [1] * (_BART_DIMS - len(dims))

    with open(name + ".hdr", "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in padded) + "\n")

    array.ravel(order="F").tofile(name + ".cfl")
