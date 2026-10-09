Basics
------

You start from the data a scanner gives you, multichannel k-space, and end
with an image. Lesson 1 shows how BART's arrays look as PyTorch tensors and how
you call a BART command on them. Lesson 2 reconstructs an undersampled
Cartesian scan through the three steps the rest of the Course builds on:
channel compression, coil sensitivities from ESPIRiT, and a regularized SENSE
reconstruction. :doc:`/explanation/encoding` derives the signal model, and
:doc:`/explanation/data-layout` lists the array layouts.
