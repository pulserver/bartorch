Regularization
--------------

Beyond the acceleration factor the coil geometry supports, the SENSE problem
is ill-conditioned: a least-squares solution amplifies noise, and the aliasing
of the undersampling is not fully resolved.  A regularized reconstruction adds
prior knowledge of the image as a penalty.  This section compares Tikhonov,
wavelet-sparsity (compressed sensing) and total-variation penalties on one
undersampled acquisition and shows how the regularization weight is chosen.  It
then writes the same reconstruction as an explicit encoding operator and an
iterative solver, the form required by an encoding for which BART has no
application.  :doc:`/explanation/inverse-problems` introduces the formulations
and the algorithms.
