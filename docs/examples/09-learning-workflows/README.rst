Tours: learning workflows
-------------------------

What comes after lessons 10 and 11, when you train and use a learned
regularizer on real problems.

- Networks for complex multi-contrast volumes, applied patch by patch.
- Training an unrolled network in stages, within the memory of one iteration.
- Self-supervised training from undersampled data alone.
- Plug-and-play with a noise level that anneals over the iterations.
- Voxel-wise uncertainty, calibrated.

:doc:`/explanation/learned-reconstruction` explains where a network enters a
reconstruction. These tours also need::

    pip install lightning torchio monai deepinv
