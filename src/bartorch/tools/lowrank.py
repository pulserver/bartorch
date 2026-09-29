"""Low-rank completion: k-space and image series completed by a rank constraint alone."""

from __future__ import annotations

from bartorch import _call

__all__: list[str] = []

#: Commands in this section without a hand-written wrapper, built from the catalogue.
_DERIVED = ("lrmatrix", "sake")

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]
