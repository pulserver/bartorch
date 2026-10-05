Examples
========

Reconstructions executed when the documentation is built, so every figure and
printed number on these pages is produced by the code shown.  Every page can be
downloaded as a Python script or a notebook, or opened in Colab.  The concepts
are in :doc:`/explanation/index`, and the interfaces in :doc:`/api/index`.

The Course is the shortest coherent path that gives a new user the framework's
core mental model and enough practical competence to work independently.  Tours
are useful applications, advanced branches or specialised workflows that are
not necessary for that core competence.

Course
------

The first six sections are the course, read in order.  Each lesson states its
aim and learning objectives and links to the lesson before and after it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Subject
   * - :doc:`01-basics/index`
     - Tensors and BART's dimensions, and a first reconstruction from undersampled Cartesian k-space
   * - :doc:`02-parallel-imaging/index`
     - Coil sensitivity estimation and nonlinear inversion
   * - :doc:`03-regularization/index`
     - Tikhonov, wavelet and total-variation penalties, and the operator-and-solver form
   * - :doc:`04-non-cartesian/index`
     - The NUFFT, density compensation and radial SENSE
   * - :doc:`05-model-based/index`
     - :math:`T_2` estimation through a signal model fitted directly to k-space
   * - :doc:`06-learning/index`
     - Plug-and-play reconstruction and an unrolled network trained through ADMM

Tours
-----

The tours are standalone.  Each opens with its objective and the course
lessons it assumes, and has no previous or next page.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Subject
   * - :doc:`07-tours/index`
     - Corrections applied before and after a reconstruction, and prewhitening of channel noise
   * - :doc:`08-workflows/index`
     - Dynamic golden-angle imaging, subspace :math:`T_1` mapping and maps from DICOM images
   * - :doc:`09-learning-workflows/index`
     - Networks for complex volumes, staged and self-supervised training, annealed plug-and-play and uncertainty

The scripts need a built ``bartorch`` with its ``io`` extra, ``brainweb-dl``, which downloads the
BrainWeb phantoms several examples build their images from, and
``matplotlib`` and ``cmap`` for the figures::

    pip install 'bartorch[io]' brainweb-dl matplotlib cmap

The learned-regularization lessons and tours additionally require ``lightning``,
``torchio``, ``monai`` and ``deepinv``, and the corrections tours ``SimpleITK``
and ``PyHySCO``; each section page names what it needs.
