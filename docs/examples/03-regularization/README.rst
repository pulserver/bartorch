Regularization
--------------

Regularized least squares: the terms of :mod:`bartorch.priors`, and the
operators and solvers a reconstruction is assembled from.

The first lesson applies Tikhonov, wavelet and total-variation terms to one
undersampled SENSE problem and selects their weights. The second writes the
same reconstruction as a :mod:`bartorch.linop` encoding operator and a
:mod:`bartorch.optim` solver, the route for an encoding BART has no application
for.
