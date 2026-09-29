Examples
========

Reconstructions executed when the documentation is built: a course in six
sections read in order, and standalone tours of the corrections applied around
a reconstruction.  :doc:`/examples/index` lists every lesson.

Running them needs a built ``bartorch``, ``brainweb-dl``, which downloads the
BrainWeb phantoms several examples build their images from, and
``matplotlib`` and ``cmap`` for the figures::

    pip install bartorch brainweb-dl matplotlib cmap

The learned-regularization section additionally requires ``lightning``,
``torchio``, ``monai`` and ``deepinv``, and downloads the DRUNet weights
``deepinv`` distributes; the tours require ``SimpleITK`` and ``PyHySCO``,
which is GPL-3.0-only and not distributed with bartorch::

    pip install lightning torchio monai deepinv SimpleITK PyHySCO
