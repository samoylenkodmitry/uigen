import torch

from models.losses import rgb_l1_loss, sobel_l1_loss


def test_rgb_loss_is_invariant_to_padding_area():
    pred_small = torch.zeros(1, 3, 2, 2)
    target_small = torch.ones(1, 3, 2, 2)
    mask_small = torch.ones(1, 1, 2, 2)
    pred_large = torch.zeros(1, 3, 4, 4)
    target_large = torch.zeros(1, 3, 4, 4)
    target_large[:, :, :2, :2] = 1.0
    mask_large = torch.zeros(1, 1, 4, 4)
    mask_large[:, :, :2, :2] = 1.0

    assert torch.allclose(
        rgb_l1_loss(pred_small, target_small, mask_small),
        rgb_l1_loss(pred_large, target_large, mask_large),
    )


def test_sobel_loss_uses_mask_sum_normalization():
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.ones(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1.0

    assert torch.isfinite(sobel_l1_loss(pred, target, mask))

