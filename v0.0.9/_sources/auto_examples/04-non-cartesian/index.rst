

.. _sphx_glr_auto_examples_04-non-cartesian:

Non-Cartesian imaging
---------------------

Radial and spiral trajectories sample k-space off the Cartesian grid.  Their
Fourier transform is a non-uniform FFT (NUFFT), their adjoint approximates an
inverse only after density compensation, and the normal operator of an
iterative reconstruction becomes a convolution with the point spread function
of the trajectory.  This section introduces trajectories, the NUFFT, density
compensation and the point spread function, and reconstructs an undersampled
golden-angle radial acquisition by non-Cartesian SENSE, with coil sensitivities
estimated from the radial data.  The extension to a time series with a temporal
regularizer is the tour :doc:`/auto_examples/08-workflows/01-dynamic-golden-angle`.
:doc:`/explanation/non-cartesian` defines the transform and its accuracy.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson introduces the building blocks of non-Cartesian reconstruction: radial, golden-angle and spiral trajectories, the non-uniform fast Fourier transform (NUFFT) that samples an image along them, the density compensation that an adjoint (gridding) reconstruction needs, and the point spread function (PSF) that describes the undersampling artefacts.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_01-trajectories-and-transforms_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/01-trajectories-and-transforms`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trajectories and transforms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson reconstructs an undersampled golden-angle radial acquisition with eight receive coils: the density-compensated gridding reconstruction first, then an iterative SENSE reconstruction with coil sensitivities estimated from the radial data themselves, with and without a total-variation penalty. The aim is to see which of the streak artefacts of radial undersampling the coil encoding removes, which the regularization removes, and what each costs.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_02-radial-sense_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/02-radial-sense`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial SENSE reconstruction</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/04-non-cartesian/01-trajectories-and-transforms
   /auto_examples/04-non-cartesian/02-radial-sense

