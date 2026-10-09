Examples
========

Every page here runs when the documentation is built, so every figure and
every number on it comes from the code you see. Each page has an *Open in
Colab* button, so you need nothing but a browser to run it.
The ideas behind them are in :doc:`/explanation/index`, and every function they
call is in :doc:`/api/index`.

Course
------

Eleven lessons that take you from a k-space tensor to an image, then through
the reconstructions BART is known for, and finally to writing and training
your own. Read them in order: each one starts from what the previous one built.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Lesson
     - The question it answers
   * - :doc:`1. Your data as tensors <01-basics/01-tensors-and-commands>`
     - How does BART's data look in Python, and how do I call a BART command on it?
   * - :doc:`2. From k-space to an image <01-basics/02-from-kspace-to-image>`
     - How do I reconstruct an undersampled Cartesian scan, step by step?
   * - :doc:`3. Coil sensitivities <02-parallel-imaging/01-coil-calibration>`
     - Where do the coil maps come from, and how much does their quality matter?
   * - :doc:`4. Joint reconstruction of image and sensitivity maps <02-parallel-imaging/02-nonlinear-inversion>`
     - What do I do when the calibration region is too small to estimate the maps?
   * - :doc:`5. Choosing a regularizer <03-regularization/01-regularized-reconstruction>`
     - Tikhonov, wavelets or total variation, and how strong?
   * - :doc:`6. Custom encoding and solver <03-regularization/02-operators-and-solvers>`
     - How do I write the reconstruction myself, as an operator and a solver, and get exactly ``pics``?
   * - :doc:`7. Basics of non-Cartesian imaging <04-non-cartesian/01-trajectories-and-transforms>`
     - What are a trajectory, the NUFFT, density compensation and the point spread function?
   * - :doc:`8. Radial SENSE <04-non-cartesian/02-radial-sense>`
     - How do I reconstruct an undersampled golden-angle radial scan?
   * - :doc:`9. Nonlinear inversion for T2 mapping <05-model-based/01-quantitative-models>`
     - Why fit a signal model to the raw data instead of to the images?
   * - :doc:`10. Plug-and-play reconstruction <06-learning/01-plug-and-play>`
     - How do I plug a pretrained network into BART's iterations, with no training?
   * - :doc:`11. Training an unrolled network <06-learning/02-modl-with-admm>`
     - How do I train a network through BART's ADMM?

By the end you can reconstruct Cartesian and non-Cartesian data with BART's
methods, write an encoding BART has no command for, and put a network inside
the reconstruction.

Tours
-----

Each tour stands on its own: it opens with what it shows and the lessons it
assumes, so you can read any of them once you have the Course behind you.

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
