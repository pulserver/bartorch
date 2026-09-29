Examples
========

A course in reconstruction with bartorch, in six sections read in order, and a
set of standalone tours.

Each lesson of the course states its aim and learning objectives, and builds
on the lessons before it: **Basics** covers BART's arrays and commands and a
Cartesian reconstruction; **Parallel imaging** the coil sensitivities and the
channel noise; **Regularization** the penalty terms and the operators and
solvers a reconstruction is assembled from; **Non-Cartesian imaging** the
non-uniform transform, radial SENSE and dynamic imaging; **Model-based
reconstruction** subspace and signal models in the encoding; and **Learned
regularization** denoisers and unrolled networks in BART's iterations. The
**Tours** show the corrections and motion tracking of :mod:`bartorch.tools`,
each on its own.

The concepts the examples use -- the encoding model, regularized least
squares, non-uniform transforms, nonlinear inversion -- are introduced in
:doc:`../explanation/index`.

Running them needs a built ``bartorch``, ``brainweb-dl``, which downloads the
BrainWeb phantoms several examples build their images from, and
``matplotlib`` and ``cmap`` for the figures::

    pip install bartorch brainweb-dl matplotlib cmap

The learned-regularization section additionally requires ``lightning``,
``torchio``, ``monai`` and ``deepinv``, and downloads the DRUNet weights
``deepinv`` distributes; the tours require ``SimpleITK`` and ``PyHySCO``,
which is GPL-3.0-only and not distributed with bartorch::

    pip install lightning torchio monai deepinv SimpleITK PyHySCO
