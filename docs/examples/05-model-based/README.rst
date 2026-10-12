Model-based reconstruction
--------------------------

To map :math:`T_2` you usually reconstruct one image per echo and fit a decay
in every voxel. In lesson 9 you put the decay into the forward model instead,
and fit the maps directly to the multi-echo k-space, so that every echo
constrains the same unknowns. The tours
:doc:`/auto_examples/08-workflows/02-subspace-t1-mapping` and
:doc:`/auto_examples/08-workflows/03-maps-from-scanner-images` show the linear
subspace alternative and a fit to DICOM images. :doc:`/explanation/nonlinear`
compares the formulations.
