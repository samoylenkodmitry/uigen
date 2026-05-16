import torch

from models.slotnet_v2 import SlotNetV2


def test_slotnet_v2_forward_returns_slot_sized_prediction():
    model = SlotNetV2(base_channels=16)
    # Input small enough to keep the test fast; must be divisible by 16
    # because the encoder has four stride-2 stages.
    view = torch.rand(1, 3, 64, 32)
    rect = torch.tensor([[0.1, 0.1, 0.7, 0.6, 1.0]])
    state = torch.rand(1, 32)
    slot_id = torch.tensor([4])

    out = model(view, rect, state, slot_id, (12, 20), input_hw=(64, 32))

    assert out["prediction"].shape == (1, 7, 12, 20)
    assert out["crop"].shape == (1, 3, 12, 20)
    assert out["log_scale"].shape == (1, 2)
    assert out["valid"].shape == (1,)


def test_slotnet_v2_encode_decode_split():
    """The encode/decode_slot split allows sharing the encoder across slots."""
    model = SlotNetV2(base_channels=16)
    view = torch.rand(1, 3, 64, 32)
    f3, f5, gp = model.encode(view)
    rect_a = torch.tensor([[0.0, 0.0, 0.4, 0.5, 1.0]])
    rect_b = torch.tensor([[0.5, 0.4, 1.0, 0.9, 1.0]])
    state = torch.rand(1, 32)
    pred_a = model.decode_slot(f3, f5, gp, rect_a, state, torch.tensor([0]), (8, 12))
    pred_b = model.decode_slot(f3, f5, gp, rect_b, state, torch.tensor([5]), (16, 8))
    assert pred_a.shape == (1, 7, 8, 12)
    assert pred_b.shape == (1, 7, 16, 8)
