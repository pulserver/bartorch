Model-based reconstruction
--------------------------

Quantitative MRI estimates tissue parameters such as :math:`T_1` and
:math:`T_2` from a series of images acquired at different contrasts.
Reconstructing each contrast separately and fitting a signal model afterwards
ignores the relation between the contrasts that the signal model states.  A
model-based reconstruction places that relation in the forward operator, so
that every contrast constrains the same unknowns.  This section fits a
nonlinear signal model directly to multi-echo k-space to estimate :math:`T_2`
maps, and compares the result with a voxel-wise fit of reconstructed images.
The linear subspace formulation and the fit to DICOM images are the tours
:doc:`/auto_examples/08-workflows/02-subspace-t1-mapping` and
:doc:`/auto_examples/08-workflows/03-maps-from-scanner-images`.
:doc:`/explanation/nonlinear` compares the formulations with reconstruction
followed by a voxel-wise fit.
