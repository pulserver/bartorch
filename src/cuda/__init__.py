"""bartorch's library compiled with CUDA, for the CUDA major version in this package's name.

Nothing here is imported.  bartorch loads ``libbartorch`` from this directory
in place of its own CPU build when torch is built for the same CUDA major
version.  ``pip install bartorch[cu12]`` or ``bartorch[cu13]`` installs it.
"""
