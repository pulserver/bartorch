Model-based reconstruction
--------------------------

Quantitative MRI estimates tissue parameters such as :math:`T_1` and
:math:`T_2` from a series of images acquired at different contrasts.
Reconstructing each contrast separately and fitting a signal model afterwards
ignores the relation between the contrasts that the signal model states.  A
model-based reconstruction places that relation in the forward operator, so
that every contrast constrains the same unknowns.  This section treats the two
standard formulations: a linear subspace model, in which inversion-recovery
signal curves are represented by a few temporal basis functions and
:math:`T_1` is fitted to the coefficient maps, and a nonlinear signal model,
through which :math:`T_2` maps are estimated directly from multi-echo k-space.
:doc:`/explanation/nonlinear` compares the two with reconstruction followed by
a voxel-wise fit.  The last lesson fits a signal model to magnitude
images read from DICOM, as a scanner exports them, and writes the map back.
