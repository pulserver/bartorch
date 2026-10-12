Learned regularization
----------------------

A learned reconstruction keeps the encoding operator and the data consistency
of an iterative reconstruction, and puts a network where the regularizer was.
In lesson 10 you take a pretrained denoiser and use it inside BART's ADMM and
FISTA, with no training at all: plug-and-play. In lesson 11 you train a network
through BART's ADMM, end to end (MoDL). The learning
workflows among the :doc:`/gallery/tours` go further.
:doc:`/explanation/learned-reconstruction` explains where a network enters a
reconstruction.
