"""Receive-field shading correction."""

import pytest
import torch

import bartorch.tools as bt

pytest.importorskip("SimpleITK", reason="bias field correction needs the correct extra")


@pytest.fixture
def shaded():
    """A uniform block seen through a left-to-right shading."""
    truth = torch.zeros(48, 48)
    truth[10:38, 10:38] = 1.0
    field = torch.linspace(0.5, 1.5, 48)[None].repeat(48, 1)
    return truth, field, truth * field


def test_a_smooth_shading_is_divided_out(shaded):
    truth, _, observed = shaded
    corrected = bt.bias_field_correct(observed)
    inside = truth > 0

    def unevenness(image):
        values = image[inside]
        return float(values.std() / values.mean())

    assert unevenness(corrected) < 0.5 * unevenness(observed)


def test_the_field_comes_back_when_asked(shaded):
    _, _, observed = shaded
    corrected, field = bt.bias_field_correct(observed, return_field=True)
    assert corrected.shape == field.shape == observed.shape
    assert torch.all(field > 0)


def test_the_result_keeps_the_dtype_and_device_of_the_image(shaded, device):
    _, _, observed = shaded
    tensor = observed.to(device=device, dtype=torch.float64)
    corrected = bt.bias_field_correct(tensor)
    assert corrected.device == tensor.device
    assert corrected.dtype == tensor.dtype


def test_a_supplied_mask_is_used(shaded):
    truth, _, observed = shaded
    corrected = bt.bias_field_correct(observed, mask=(truth > 0).to(torch.uint8))
    assert corrected.shape == observed.shape


def test_a_volume_is_corrected_as_readily_as_a_slice():
    volume = torch.zeros(16, 24, 24)
    volume[4:12, 6:18, 6:18] = 1.0
    field = torch.linspace(0.6, 1.4, 24)[None, None]
    assert bt.bias_field_correct(volume * field).shape == volume.shape


def test_a_four_dimensional_image_is_refused():
    with pytest.raises(ValueError, match="2D or 3D"):
        bt.bias_field_correct(torch.ones(2, 4, 8, 8))


def test_a_non_positive_shrink_factor_is_refused():
    with pytest.raises(ValueError, match="shrink_factor must be positive"):
        bt.bias_field_correct(torch.ones(8, 8), shrink_factor=0)
