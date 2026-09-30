Learned regularization
----------------------

A learned reconstruction replaces the hand-specified regularization term by a
neural network and keeps the encoding operator and the data consistency of the
iterative reconstruction.  This section starts with plug-and-play
reconstruction, in which a pretrained denoiser takes the place of the proximal
operator of ADMM and FISTA without any training, and proceeds to an unrolled
network trained through BART's ADMM (MoDL); networks for complex
multi-contrast volumes, applied patch by patch; staged and self-supervised
training of an unrolled network; plug-and-play with an annealed noise level;
and calibrated voxel-wise uncertainty.
:doc:`/explanation/learned-reconstruction` describes where a network enters a
reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv

The first lesson downloads the DRUNet weights ``deepinv`` distributes.
