"""Training of reconstruction networks with ``lightning``, augmentation with ``torchio``.

This module imports ``lightning`` and ``torchio``, which ``pip install
bartorch[learning]`` installs; the rest of :mod:`bartorch.learning` does not,
and imports this module the first time one of its names is asked for.
"""

from __future__ import annotations

import math

import lightning
import torch
import torchio

from bartorch.learning._splitting import split

__all__ = ["RandomGain", "Reconstruction"]

_STAGES = ("denoiser", "greedy", "end-to-end")


def _l1(prediction: torch.Tensor, target: torch.Tensor, item: dict) -> torch.Tensor:
    """Mean modulus of the error, weighted voxel by voxel by ``item["weights"]`` if present."""
    error = (prediction - target).abs()
    weights = item.get("weights")
    return (error if weights is None else weights * error).mean()


def _normalized(residual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """SSDU's loss: the residual's l2 and l1 norms, each relative to the held-out data's."""
    tiny = torch.finfo(torch.float32).tiny
    l2 = residual.abs().square().sum().sqrt() / reference.abs().square().sum().sqrt().clamp_min(
        tiny
    )
    l1 = residual.abs().sum() / reference.abs().sum().clamp_min(tiny)
    return l2 + l1


class Reconstruction(lightning.LightningModule):
    r"""Lightning module training one of the three stages of an unrolled reconstruction network.

    * ``"denoiser"`` trains the network alone, on pairs of a degraded image and
      its reference.
    * ``"greedy"`` trains an :class:`~bartorch.learning.Unrolled` built with
      ``detach=True``, back-propagating a loss on each iteration's image before
      the next iteration runs; the losses grow geometrically, the last
      ``ratio`` times the first.
    * ``"end-to-end"`` takes the loss on the last image through the whole
      stack, which ``checkpoint=True`` makes affordable.

    Run in this order, each from the weights the previous one left, these are
    the staged training of Urman et al.

    Items are dictionaries and a batch is a list of them (``collate_fn=list``),
    since operators and k-space vary between subjects:

    * ``"denoiser"``: ``input`` and ``target`` images, and optionally
      ``step``, ``sigma`` and ``label`` passed to the network;
    * the unrolled stages: the data ``y`` and its encoding ``A``; the reference
      image ``target``, or the sampling ``pattern`` for self-supervised
      training; optionally ``x0``.

    Any item may carry ``weights``, multiplying the voxel-wise error of the
    default loss.  Batches are not moved to the trainer's device: a
    :class:`~bartorch.learning.Patchwise` network or an operator with operands
    on the host moves what it needs.

    An item without ``target`` is trained self-supervised (SSDU): its
    ``pattern`` is partitioned by :func:`~bartorch.learning.split` at every
    step, the network reconstructs from one part, and the loss is the
    normalized :math:`\ell_1 + \ell_2` error of the held-out k-space.  The
    validation partition is the same at every epoch.

    Optimization is manual: gradients are accumulated over ``accumulate``
    items, optionally clipped, and applied with Adam; the learning rate is
    multiplied by ``factor`` after ``patience`` epochs without improvement of
    the logged ``val_loss``.

    Parameters
    ----------
    model : nn.Module
        The network for ``"denoiser"``, an
        :class:`~bartorch.learning.Unrolled` for the other stages.
    stage : {"denoiser", "greedy", "end-to-end"}, default="end-to-end"
        Training stage.
    loss : callable, default=None
        ``loss(prediction, target, item)`` for supervised items; the mean
        modulus of the error when omitted.
    ratio : float, default=10.0
        Weight of the last iteration's loss relative to the first's, in the
        greedy stage.
    lr : float, default=1e-3
        Adam's learning rate.
    weight_decay : float, default=0.0
        Adam's weight decay.
    factor : float, default=0.5
        Multiplier of the learning rate on a plateau of ``val_loss``.
    patience : int, default=10
        Epochs without improvement of ``val_loss`` before the learning rate
        is reduced.
    accumulate : int, default=1
        Items whose gradients are summed before an optimizer step.
    clip : float, default=None
        Maximum norm of the gradient; no clipping when omitted.
    fraction : float, default=0.4
        Share of the acquired samples held out, for self-supervised items.
    split_options : dict, default=None
        Further keywords of :func:`~bartorch.learning.split`.

    Raises
    ------
    ValueError
        An unknown ``stage``; ``"greedy"`` with a model not built with
        ``detach=True``; ``accumulate`` below one.  In training, ``"greedy"``
        with a block whose image does not depend on that iteration's
        parameters, such as :class:`~bartorch.optim.ADMMBlock`, whose image is
        the conjugate-gradient update preceding the denoiser;
        :class:`~bartorch.optim.ISTBlock` qualifies.

    References
    ----------
    Urman Y, Nishimura M, Abraham D, Cao X, Setsompop K. Fully 3D unrolled
    magnetic resonance fingerprinting reconstruction via staged pretraining
    and implicit gridding. Magn Reson Med 96(5):2516-2529, 2026.
    https://doi.org/10.1002/mrm.70500

    Examples
    --------
    >>> stage = learning.Reconstruction(model, stage="greedy", accumulate=4)
    >>> trainer = lightning.Trainer(max_epochs=50, callbacks=[EarlyStopping("val_loss")])
    >>> trainer.fit(stage, DataLoader(train, collate_fn=list), DataLoader(valid, collate_fn=list))
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stage: str = "end-to-end",
        *,
        loss=None,
        ratio: float = 10.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        factor: float = 0.5,
        patience: int = 10,
        accumulate: int = 1,
        clip: float | None = None,
        fraction: float = 0.4,
        split_options: dict | None = None,
    ):
        super().__init__()
        if stage not in _STAGES:
            raise ValueError(f"stage is one of {_STAGES}, not {stage!r}")
        if "greedy" == stage and not getattr(model, "detach", False):
            raise ValueError(
                "greedy training releases each iteration's graph, so the stack is built with "
                "detach=True"
            )
        if accumulate < 1:
            raise ValueError(f"a step accumulates at least one item, not {accumulate}")
        self.model = model
        self.stage = stage
        self.loss = _l1 if loss is None else loss
        self.ratio = float(ratio)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.factor = float(factor)
        self.patience = int(patience)
        self.accumulate = int(accumulate)
        self.clip = None if clip is None else float(clip)
        self.fraction = float(fraction)
        self.split_options = dict(split_options or {})
        self.automatic_optimization = False
        self._seen = 0

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Leave the batch where the dataset put it."""
        return batch

    def configure_optimizers(self):
        """Adam over the trainable parameters, reduced on a plateau of ``val_loss``."""
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": plateau}}

    def training_step(self, batch, batch_idx):
        """Back-propagate each item of a list and step every ``accumulate`` items."""
        optimizer = self.optimizers()
        total = 0.0
        for item in batch:
            total += self._backward(item)
            self._seen += 1
            if 0 == self._seen % self.accumulate:
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                optimizer.step()
                optimizer.zero_grad()
        self.log("train_loss", total / len(batch), batch_size=len(batch), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Log the mean loss of the items as ``val_loss``."""
        with torch.no_grad():
            total = sum(float(self._loss(item, validating=True)) for item in batch)
        self.log("val_loss", total / len(batch), batch_size=len(batch), prog_bar=True)

    def on_validation_epoch_end(self):
        """Step the plateau scheduler on ``val_loss``."""
        if self.trainer.sanity_checking:
            return
        value = self.trainer.callback_metrics.get("val_loss")
        schedulers = self.lr_schedulers()
        if value is not None and schedulers is not None:
            schedulers.step(float(value))

    def _backward(self, item: dict) -> float:
        """Back-propagate the item's loss, one iteration at a time when greedy."""
        scale = 1.0 / self.accumulate
        if "greedy" != self.stage:
            loss = self._loss(item)
            self.manual_backward(scale * loss)
            return float(loss.detach())
        weights = self._weights()
        total = 0.0
        y, A, target = self._problem(item)
        learned = False
        for k, image in enumerate(self.model.steps(y, A, item.get("x0"))):
            loss = weights[k] * self._compare(image, target, item, A)
            if loss.requires_grad:
                self.manual_backward(scale * loss)
                learned = True
            total += float(loss.detach())
        if not learned:
            raise ValueError(
                "no iteration's image depends on that iteration's parameters, so greedy training "
                "has nothing to train: the block's image has to pass through its own denoiser, as "
                "ISTBlock's does and ADMMBlock's, updated before the denoiser, does not"
            )
        return total

    def _loss(self, item: dict, validating: bool = False) -> torch.Tensor:
        if "denoiser" == self.stage:
            keywords = {k: item[k] for k in ("sigma", "step", "label") if k in item}
            sigma = keywords.pop("sigma", None)
            x = item["input"][None]
            made = self.model(x, **keywords) if sigma is None else self.model(x, sigma, **keywords)
            return self.loss(made[0], item["target"], item)
        y, A, target = self._problem(item, validating)
        return self._compare(self.model(y, A, item.get("x0")), target, item, A)

    def _problem(self, item: dict, validating: bool = False):
        """The data and operator the network reconstructs from, and what its image is held to.

        A supervised item is held to its reference image.  A self-supervised
        one is reconstructed from part of its samples and held to the rest,
        which is returned as ``(held-out data, restricted operator)``.
        """
        y, A = item["y"], item["A"]
        if "target" in item:
            return y, A, item["target"]
        from bartorch.linop import Diagonal

        generator = torch.Generator().manual_seed(0) if validating else None
        keep, held = split(
            item["pattern"], self.fraction, generator=generator, **self.split_options
        )
        shape = tuple(A.oshape)

        def restricted(mask):
            mask = mask.to(y.device, torch.complex64)
            mask = mask.reshape((1,) * (len(shape) - mask.ndim) + tuple(mask.shape))
            return mask, Diagonal(mask, shape) @ A

        keep, seen = restricted(keep)
        held, unseen = restricted(held)
        return keep * y, seen, (held * y, unseen)

    def _compare(self, image, target, item, A) -> torch.Tensor:
        if not isinstance(target, tuple):
            return self.loss(image, target, item)
        from bartorch.linop._autograd import apply_forward

        data, unseen = target
        return _normalized(apply_forward(unseen, image) - data, data)

    def _weights(self) -> list[float]:
        """Per-iteration loss weights, growing geometrically to ``ratio`` times the first."""
        count = self.model.iterations
        growth = self.ratio ** (1.0 / (count - 1)) if count > 1 else 1.0
        raw = [growth**k for k in range(count)]
        return [w / math.fsum(raw) for w in raw]


class RandomGain(
    torchio.transforms.augmentation.RandomTransform, torchio.transforms.IntensityTransform
):
    r"""Multiply every image of a subject by one random complex gain (``torchio`` transform).

    The images hold complex values as real channels, laid out by
    :func:`~bartorch.learning.as_real`: ``2 k`` channels are ``k`` complex
    images, real parts first.  The gain is :math:`e^{s} e^{i \varphi}`, with
    :math:`\varphi` uniform in ``phase`` and :math:`s` uniform in
    ``log_scale``, the same for every image of the subject, so a degraded input
    and its reference stay consistent.  ``torchio``'s spatial transforms apply
    to the real and imaginary channels as they are; its other intensity
    transforms do not respect the complex structure.

    Parameters
    ----------
    phase : float or tuple of float, default=pi
        Range of :math:`\varphi`, in radians; a single value ``d`` is
        ``(-d, d)``.
    log_scale : float or tuple of float, default=0.0
        Range of :math:`s`; a single value ``d`` is ``(-d, d)``.
    **kwargs
        Passed to :class:`torchio.transforms.Transform`.
    """

    def __init__(self, phase=math.pi, log_scale=0.0, **kwargs):
        super().__init__(**kwargs)
        self.phase_range = self._parse_range(phase, "phase")
        self.log_scale_range = self._parse_range(log_scale, "log_scale")

    def apply_transform(self, subject):
        phi = self.sample_uniform(*self.phase_range)
        s = self.sample_uniform(*self.log_scale_range)
        c, d = math.exp(s) * math.cos(phi), math.exp(s) * math.sin(phi)
        for image in self.get_images(subject):
            data = image.data
            if data.shape[0] % 2:
                raise ValueError(
                    f"a complex image has real and imaginary channels, an even number, not "
                    f"{data.shape[0]}"
                )
            re, im = data.float().chunk(2)
            image.set_data(torch.cat([c * re - d * im, d * re + c * im]))
        return subject
