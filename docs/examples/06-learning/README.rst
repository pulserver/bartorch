Learned regularization
----------------------

A learned reconstruction replaces the hand-specified regularization term by a
neural network and keeps the encoding operator and the data consistency of the
iterative reconstruction.  This section covers the two ways a network enters
BART's iterations: plug-and-play reconstruction, in which a pretrained denoiser
takes the place of the proximal operator of ADMM and FISTA without any
training, and an unrolled network trained through BART's ADMM (MoDL).
Networks for complex volumes, staged and self-supervised training, annealed
plug-and-play and uncertainty are the tours of
:doc:`/auto_examples/09-learning-workflows/index`.
:doc:`/explanation/learned-reconstruction` describes where a network enters a
reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv

The first lesson downloads the DRUNet weights ``deepinv`` distributes.
