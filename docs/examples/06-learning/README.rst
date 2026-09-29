Learned regularization
----------------------

Reconstructions whose regularizer is a network rather than a specified term.

The first lesson uses a pretrained denoiser as the proximal step of BART's
ADMM and FISTA iterations, with no training. The second unrolls BART's ADMM
with a convolutional denoiser in its proximal step and trains the network
through the iteration against fully sampled images. The third builds networks
for complex multi-contrast volumes and applies them patch by patch. The fourth
and fifth train an unrolled network in stages, and without references. The
sixth anneals the noise level of a plug-and-play denoiser, and the seventh
attaches calibrated error bars to a learned reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv

The first lesson downloads the DRUNet weights ``deepinv`` distributes.
