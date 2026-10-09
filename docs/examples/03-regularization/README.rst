Regularization
--------------

Past the acceleration your coils can support, SENSE amplifies noise and leaves
aliasing behind. A regularizer adds what you know about the image. In lesson 5
you compare Tikhonov, wavelet and total-variation penalties on one scan and
choose their weight. In lesson 6 you write the same reconstruction yourself, as
an encoding operator and a solver, which is how you reconstruct anything BART
has no command for. :doc:`/explanation/inverse-problems` explains the
formulations and the iterations.
