"""Training of reconstruction networks with ``lightning``, augmentation with ``torchio``.

This module imports ``lightning`` and ``torchio``, which ``pip install
bartorch[learning]`` installs; the rest of :mod:`bartorch.learning` does not.
"""

from __future__ import annotations

import math

import lightning
import torch
import torchio

from bartorch.learning.splitting import split

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
    """A reconstruction network trained in one of the three stages of an unrolled network.

    ``"denoiser"`` trains the network alone, on pairs of a degraded image and
    its reference, optionally with the iteration index and noise level it
    will meet in the unrolled network.  ``"greedy"`` trains an
    :class:`~bartorch.learning.Unrolled` built with ``detach=True``, taking a
    loss on each iteration's image and back-propagating it before the next
    iteration runs, so memory holds one iteration; the losses are weighted
    geometrically, the last ``ratio`` times the first.  ``"end-to-end"`` takes
    the loss on the last image alone, through the whole stack, which is
    affordable with ``checkpoint=True``.  Greedy training needs a block whose
    image passes through its own iteration's denoiser, as
    :class:`~bartorch.optim.ISTBlock`'s gradient step followed by the
    denoiser does; :class:`~bartorch.optim.ADMMBlock` returns the image of its
    conjugate-gradient update, which depends on the previous iteration's
    denoiser alone.  Run in that order, each stage
    starting from the weights the previous one left, these are the staged
    training of Urman et al.

    Items are dictionaries and a batch is a list of them (``collate_fn=list``),
    because operators and k-space vary between subjects:

    * ``"denoiser"``: ``input`` and ``target`` images, and optionally
      ``step``, ``sigma`` and ``label`` passed to the network.
    * the unrolled stages: ``y`` and ``A``, the measured data and its
      encoding operator; ``target``, the reference image, or ``pattern`` for
      self-supervised training; optionally ``x0``.

    Every item may carry ``weights``, multiplying the voxel-wise error of the
    default loss.  Items stay where the dataset put them: the batch is not
    moved to the device the trainer runs on, since a
    :class:`~bartorch.learning.Patchwise` network or an operator whose
    operands are on the host decides itself what crosses to the card.

    Without ``target`` the item is trained self-supervised: its ``pattern``
    is split by :func:`~bartorch.learning.split` at every step, the network
    reconstructs from ``pattern`` times the reconstruct part, and the loss is
    SSDU's normalised l1-l2 loss on the held-out part of the k-space.  The
    split drawn for validation is the same at every epoch.

    Optimization is manual, so that a greedy step can release each
    iteration's graph: gradients are accumulated over ``accumulate`` items,
    optionally clipped, and applied with Adam; the learning rate is reduced
    by ``factor`` after ``patience`` epochs without improvement of the logged
    ``val_loss``, which :class:`lightning.pytorch.callbacks.EarlyStopping` and
    :class:`lightning.pytorch.callbacks.ModelCheckpoint` also monitor.

    Parameters
    ----------
    model : nn.Module
        The network for ``"denoiser"``, an
        :class:`~bartorch.learning.Unrolled` for the other stages.
    stage : {"denoiser", "greedy", "end-to-end"}, default="end-to-end"
        What is trained, and how.
    loss : callable, default=None
        ``loss(prediction, target, item)`` for supervised items; the mean
        modulus of the error by default.
    ratio : float, default=10.0
        Weight of the last iteration's loss over the first's, in the greedy
        stage.
    lr : float, default=1e-3
        Adam's learning rate.
    weight_decay : float, default=0.0
        Adam's weight decay.
    factor, patience : float, int, default=0.5, 10
        Reduction of the learning rate on a plateau of ``val_loss``.
    accumulate : int, default=1
        Items whose gradients are summed before a step.
    clip : float, default=None
        Maximum norm of the gradient.
    fraction : float, default=0.4
        Share of the acquired samples held out, for self-supervised items.
    split_options : dict, default=None
        Further keywords of :func:`~bartorch.learning.split`.

    Examples
    --------
    >>> stage = learning.training.Reconstruction(model, stage="greedy", accumulate=4)
    >>> trainer = lightning.Trainer(max_epochs=50, callbacks=[EarlyStopping("val_loss")])
    >>> trainer.fit(stage, DataLoader(train, collate_fn=list), DataLoader(valid, collate_fn=list))

    References
    ----------
    Urman Y, Nishimura M, Abraham D, Cao X, Setsompop K. Fully 3D unrolled
    magnetic resonance fingerprinting reconstruction via staged pretraining
    and implicit gridding. arXiv:2601.17143, 2026.
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
        return batch

    def configure_optimizers(self):
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
        with torch.no_grad():
            total = sum(float(self._loss(item, validating=True)) for item in batch)
        self.log("val_loss", total / len(batch), batch_size=len(batch), prog_bar=True)

    def on_validation_epoch_end(self):
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
            return float(loss)
        weights = self._weights()
        total = 0.0
        y, A, target = self._problem(item)
        learned = False
        for k, image in enumerate(self.model.steps(y, A, item.get("x0"))):
            loss = weights[k] * self._compare(image, target, item, A)
            if loss.requires_grad:
                self.manual_backward(scale * loss)
                learned = True
            total += float(loss)
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
        from bartorch.linop.autograd import apply_forward

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
    """Multiply every image of a subject by one random complex gain.

    The images are complex values laid out as real channels by
    :func:`~bartorch.learning.as_real`, real parts first: a channel count of
    ``2 k`` is ``k`` complex images.  The gain is ``exp(s) exp(i phi)``, with
    ``phi`` uniform in ``phase`` and ``s`` uniform in ``log_scale``, and the
    same for every image of the subject, so that a degraded input and its
    reference, or an image and the k-space made from it, stay consistent.
    The spatial transforms of ``torchio`` apply to the real and imaginary
    channels as they are, since interpolating each part is interpolating
    the complex value; its other intensity transforms do not respect the
    complex structure.

    Parameters
    ----------
    phase : float or tuple of float, default=pi
        Range of ``phi`` in radians; a single value ``d`` is ``(-d, d)``.
    log_scale : float or tuple of float, default=0.0
        Range of ``s``; a single value ``d`` is ``(-d, d)``.
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
