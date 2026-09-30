Examples
========

Reconstructions executed when the documentation is built, so every figure and
printed number on these pages is produced by the code shown.  The first six
sections are a course read in order: each lesson states its aim and builds on
the lessons before it.  The tours are standalone.  Every page can be
downloaded as a Python script or a notebook, or opened in Colab.  The concepts
are in :doc:`/explanation/index`, and the interfaces in :doc:`/api/index`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Subject
   * - :doc:`01-basics/index`
     - Tensors and BART's dimensions, and a first reconstruction from undersampled Cartesian k-space
   * - :doc:`02-parallel-imaging/index`
     - Coil sensitivity estimation, nonlinear inversion, and noise prewhitening
   * - :doc:`03-regularization/index`
     - Tikhonov, wavelet and total-variation penalties, and the operator-and-solver form
   * - :doc:`04-non-cartesian/index`
     - The NUFFT, density compensation, radial SENSE and dynamic golden-angle imaging
   * - :doc:`05-model-based/index`
     - Subspace :math:`T_1` mapping and :math:`T_2` estimation through a signal model
   * - :doc:`06-learning/index`
     - Plug-and-play, unrolled and self-supervised networks, and uncertainty
   * - :doc:`07-tours/index`
     - Corrections applied before and after a reconstruction

The scripts need a built ``bartorch``, ``brainweb-dl``, which downloads the
BrainWeb phantoms several examples build their images from, and
``matplotlib`` and ``cmap`` for the figures::

    pip install bartorch brainweb-dl matplotlib cmap

The learned-regularization section additionally requires ``lightning``,
``torchio``, ``monai`` and ``deepinv``, and the tours ``SimpleITK`` and
``PyHySCO``; each section page names what it needs.
