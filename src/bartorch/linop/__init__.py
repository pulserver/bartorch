"""Linear operators, BART's own and Python-defined, composable into one BART operator."""

from __future__ import annotations

from bartorch.linop._base import LinearOperator
from bartorch.linop._basic import (
    FFT,
    ComponentDiagonal,
    Conj,
    Diagonal,
    Identity,
    MultiplySum,
    Zero,
)
from bartorch.linop._combine import block, block_diag, concatenate, hstack, stack
from bartorch.linop._mri import CartesianSense, FieldCorrected, WaveSense
from bartorch.linop._nufft import NUFFT
from bartorch.linop._sense import NoncartesianSense
from bartorch.linop._shape import (
    Extract,
    Flip,
    Hankel,
    Mean,
    Pad,
    Permute,
    Real,
    Repeat,
    Reshape,
    Resize,
    Roll,
    ScaledSum,
    Sum,
    Transpose,
)
from bartorch.linop._signal import Convolve, Gradient, Matrix

__all__ = [
    "block",
    "block_diag",
    "concatenate",
    "hstack",
    "stack",
    "CartesianSense",
    "ComponentDiagonal",
    "Conj",
    "Convolve",
    "Diagonal",
    "Extract",
    "FFT",
    "FieldCorrected",
    "Flip",
    "Gradient",
    "Hankel",
    "Identity",
    "LinearOperator",
    "Matrix",
    "Mean",
    "MultiplySum",
    "NUFFT",
    "NoncartesianSense",
    "Pad",
    "Permute",
    "Real",
    "Repeat",
    "Reshape",
    "Resize",
    "Roll",
    "ScaledSum",
    "Sum",
    "Transpose",
    "WaveSense",
    "Zero",
]
