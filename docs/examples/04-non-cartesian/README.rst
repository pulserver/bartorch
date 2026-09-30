Non-Cartesian imaging
---------------------

Radial and spiral trajectories sample k-space off the Cartesian grid.  Their
Fourier transform is a non-uniform FFT (NUFFT), their adjoint approximates an
inverse only after density compensation, and the normal operator of an
iterative reconstruction becomes a convolution with the point spread function
of the trajectory.  This section introduces trajectories, the NUFFT, density
compensation and the point spread function; reconstructs an undersampled
golden-angle radial acquisition by non-Cartesian SENSE, with coil sensitivities
estimated from the radial data; and reconstructs a continuous golden-angle
acquisition as a time series with a temporal regularizer.
:doc:`/explanation/non-cartesian` defines the transform and its accuracy.
