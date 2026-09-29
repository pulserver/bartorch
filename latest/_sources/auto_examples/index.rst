:orphan:

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
``deepinv`` distributes; the tours require ``SimpleITK``::

    pip install lightning torchio monai deepinv SimpleITK


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Basics
------

The first section of the course: BART's arrays and commands in Python, and a
Cartesian reconstruction from simulated k-space to a coil-combined image.

The first lesson relates tensor shapes to BART's dimension vectors, simulates
an acquisition with BART's analytical phantom and checks BART's FFT against
NumPy. The second follows undersampled Cartesian k-space through channel
compression, ESPIRiT calibration and :func:`bartorch.apps.pics`; the sections
after it take each of those steps in turn.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The first lesson of the course: how BART&#x27;s arrays, commands and Fourier conventions appear in Python.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_01-tensors-and-commands_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/01-tensors-and-commands`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tensors and commands</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reconstruction of an undersampled Cartesian acquisition, from the measured k-space to a coil-combined image.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_02-from-kspace-to-image_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/02-from-kspace-to-image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">From k-space to image</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Parallel imaging
----------------

The coil sensitivities a SENSE reconstruction depends on, and the channel
noise it propagates.

The first lesson compares three sensitivity estimates from the same
acquisition, :func:`bartorch.tools.caldir`, ESPIRiT and nonlinear inversion.
The second estimates the image and the sensitivities jointly where the
calibration region is too small for a separate calibration. The third measures
the effect of correlated channel noise on the reconstruction and removes it by
prewhitening.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three estimates of the receive sensitivities of a coil array from the same undersampled Cartesian acquisition, and the SENSE reconstructions they lead to.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_01-coil-calibration_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/01-coil-calibration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Coil sensitivity calibration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Estimating the image and the coil sensitivities together, from undersampled data whose fully sampled central region is too small for a separate calibration.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_02-nonlinear-inversion_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/02-nonlinear-inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The effect of channel-noise correlation on a SENSE reconstruction, and its removal by prewhitening with a noise measurement.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_03-noise-prewhitening_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/03-noise-prewhitening`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Noise prewhitening</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Regularization
--------------

Regularized least squares: the terms of :mod:`bartorch.priors`, and the
operators and solvers a reconstruction is assembled from.

The first lesson applies Tikhonov, wavelet and total-variation terms to one
undersampled SENSE problem and selects their weights. The second writes the
same reconstruction as a :mod:`bartorch.linop` encoding operator and a
:mod:`bartorch.optim` solver, the route for an encoding BART has no application
for.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three regularization terms applied to the same undersampled SENSE problem, and the dependence of the reconstruction error on the regularization weight.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_01-regularized-reconstruction_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/01-regularized-reconstruction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regularized reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same reconstruction written as an encoding operator and a solver rather than as a call to a BART application.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_02-operators-and-solvers_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/02-operators-and-solvers`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Operators and solvers</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Non-Cartesian imaging
---------------------

Sampling off the Cartesian grid.

The first lesson covers trajectories, the non-uniform Fourier transform,
density compensation and the point spread function the normal operator
convolves with. The second reconstructs an undersampled radial acquisition
with :class:`bartorch.linop.NoncartesianSense`. The third reconstructs a
continuous golden-angle acquisition as a time series with a temporal
regularizer.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The non-Cartesian interfaces: the trajectories bartorch.tools.traj generates, the non-uniform Fourier transform along one, the density compensation an adjoint reconstruction needs, and the point spread function the normal operator convolves with.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_01-trajectories-and-transforms_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/01-trajectories-and-transforms`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trajectories and transforms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An undersampled golden-angle radial acquisition reconstructed by regularized least squares, with the non-Cartesian SENSE operator">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_02-radial-sense_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/02-radial-sense`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial SENSE reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A continuously acquired golden-angle radial scan reconstructed as a time series, with a temporal regularizer compensating for the undersampling of each frame.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_03-dynamic-golden-angle_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/03-dynamic-golden-angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic golden-angle radial MRI</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Model-based reconstruction
--------------------------

Reconstructions that put a signal model into the encoding.

The first lesson constrains a series of four hundred radial frames to a
low-dimensional subspace of inversion-recovery curves and fits :math:`T_1`
from the coefficient maps. The second estimates :math:`T_2` maps from
multi-echo k-space directly, through a nonlinear forward operator, and
compares the result with fitting reconstructed echo images.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An inversion-recovery FLASH acquisition of four hundred frames, one spoke each, reconstructed into the coefficients of a signal subspace and fitted for T_1.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_01-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/01-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A multi-echo spin-echo acquisition fitted for T_2 in two ways: by reconstructing the echo images and fitting them afterwards, and by putting the signal model inside the forward operator and fitting the k-space directly.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_02-quantitative-models_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/02-quantitative-models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps straight from k-space</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Learned regularization
----------------------

Reconstructions whose regularizer is a network rather than a specified term.

The first lesson uses a pretrained denoiser as the proximal step of BART's
ADMM and FISTA iterations, with no training. The second unrolls BART's ADMM
with a convolutional denoiser in its proximal step and trains the network
through the iteration against fully sampled images.

The section additionally requires::

    pip install lightning torchio monai deepinv

The first lesson downloads the DRUNet weights ``deepinv`` distributes.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A pretrained image denoiser used as the proximal step of BART&#x27;s iterations, in place of a specified regularization term, for an undersampled Cartesian SENSE acquisition.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_01-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/01-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plug-and-play denoisers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An unrolled network for undersampled Cartesian SENSE: a convolutional denoiser in the proximal step of BART&#x27;s alternating-direction iteration, trained end to end against fully sampled images.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_02-modl-with-admm_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/02-modl-with-admm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MoDL, on BART's ADMM</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Tours
-----

Standalone examples of the corrections and the rigid-motion tracking in
:mod:`bartorch.tools`, each simulated from a known ground truth so that the
correction can be measured against it. They do not depend on one another or
on the course.

The section additionally requires::

    pip install SimpleITK


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two operations on Cartesian k-space before it is reconstructed: removing the readout oversampling, and weighting the measurement by an apodization window to suppress the Gibbs ringing of its truncation.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_01-readout-oversampling-and-apodization_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/01-readout-oversampling-and-apodization`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Readout oversampling and apodization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two corrections an echo-planar readout needs before its lines form a Cartesian k-space: the odd/even phase that produces the Nyquist ghost, estimated from a three-line navigator, and the resampling of samples taken on the gradient ramps onto the uniform grid.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_02-epi-ghost-and-ramp-sampling_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">EPI Nyquist ghost and ramp sampling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The intensity shading that a surface receive array leaves in a root-sum-of-squares image, estimated and removed by N4 bias field correction.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_03-bias-field_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/03-bias-field`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Receive bias field</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The geometric distortion a gradient coil&#x27;s nonlinearity produces over a large field of view, simulated from a spherical-harmonic description of the coil and corrected with bartorch.tools.Gradunwarp.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_04-gradient-nonlinearity_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/04-gradient-nonlinearity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient nonlinearity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The blurring that off-resonance produces in a spiral image, simulated for a known field map and removed with bartorch.tools.deblur.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_05-spiral-deblurring_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/05-spiral-deblurring`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Off-resonance blurring in spirals</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six-degree-of-freedom rigid head motion measured from three orthogonal radial navigator planes, and carried across a scan by a Kalman filter, with the rigid-motion functions of bartorch.tools.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_06-navigator-motion_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/06-navigator-motion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rigid head motion from navigators</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/01-basics/index.rst
   /auto_examples/02-parallel-imaging/index.rst
   /auto_examples/03-regularization/index.rst
   /auto_examples/04-non-cartesian/index.rst
   /auto_examples/05-model-based/index.rst
   /auto_examples/06-learning/index.rst
   /auto_examples/07-tours/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
