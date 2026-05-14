import torch
import torch.nn.functional as F

from models.crop import crop_view_regions, jitter_rects


def test_crop_full_image_matches_input_when_size_is_unchanged():
    image = torch.arange(3 * 4 * 5, dtype=torch.float32).view(1, 3, 4, 5)
    rect = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

    crop, log_scale, valid = crop_view_regions(image, rect, (4, 5), input_hw=(4, 5))

    assert torch.allclose(crop, image)
    assert torch.allclose(log_scale, torch.zeros_like(log_scale))
    assert valid.tolist() == [True]


def test_crop_dimensions_use_height_width_order():
    image = torch.rand(1, 3, 20, 10)
    rect = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

    crop, _, _ = crop_view_regions(image, rect, (7, 3), input_hw=(20, 10))

    assert crop.shape == (1, 3, 7, 3)


def test_log_scale_ratios_are_log_render_over_slot_size():
    image = torch.rand(1, 3, 100, 200)
    rect = torch.tensor([[0.25, 0.25, 0.75, 0.75, 1.0]])

    _, log_scale, _ = crop_view_regions(image, rect, (25, 50), input_hw=(100, 200))

    assert torch.allclose(log_scale, torch.log(torch.tensor([[2.0, 2.0]])))


def test_jitter_rects_preserves_bounds_and_invisible_rects():
    rects = torch.tensor([[0.2, 0.2, 0.8, 0.8, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    jittered = jitter_rects(rects)

    assert torch.all(jittered[:, :4] >= 0.0)
    assert torch.all(jittered[:, :4] <= 1.0)
    assert torch.allclose(jittered[1], rects[1])

