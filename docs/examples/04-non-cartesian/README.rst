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
