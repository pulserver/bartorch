

.. _sphx_glr_auto_examples_04-non-cartesian:

Non-Cartesian imaging
---------------------

Radial and spiral trajectories sample k-space off the grid, so the Fourier
transform becomes a NUFFT. In lesson 7 you meet trajectories, the NUFFT,
density compensation and the point spread function. In lesson 8 you
reconstruct an undersampled golden-angle radial scan by non-Cartesian SENSE,
with coil sensitivities estimated from the radial data themselves. The tour
:doc:`/auto_examples/08-workflows/01-dynamic-golden-angle` extends it to a time
series. :doc:`/explanation/non-cartesian` defines the transform and its
accuracy.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lessons 2 to 6 every sample sat on the Cartesian grid, so the FFT and its inverse were all the Fourier transforms you needed. In this lesson you leave the grid. You sample the same brain slice along radial and spiral trajectories, reconstruct it by gridding, and see what each step of that reconstruction does to the image.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_01-trajectories-and-transforms_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/01-trajectories-and-transforms`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">7. Basics of non-Cartesian imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 7 you computed radial trajectories, the NUFFT, density compensation and the point spread function, and saw that omitting spokes produces streak artefacts. In this lesson you reconstruct an undersampled golden-angle radial scan with eight receive coils: you estimate the coil sensitivities from the radial data, then reconstruct by gridding, by iterative SENSE, and by SENSE with a total-variation penalty.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_02-radial-sense_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/02-radial-sense`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">8. Radial SENSE</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/04-non-cartesian/01-trajectories-and-transforms
   /auto_examples/04-non-cartesian/02-radial-sense

