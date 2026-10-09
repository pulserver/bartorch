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
