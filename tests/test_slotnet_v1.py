import torch

from models.slotnet_v1 import SlotNetV1


def test_slotnet_forward_returns_slot_sized_prediction():
    model = SlotNetV1()
    view = torch.rand(1, 3, 32, 32)
    rect = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
    state = torch.rand(1, 32)
    slot_id = torch.tensor([4])

    out = model(view, rect, state, slot_id, (16, 16), input_hw=(32, 32))

    assert out["prediction"].shape == (1, 7, 16, 16)
    assert out["crop"].shape == (1, 3, 16, 16)
    assert out["log_scale"].shape == (1, 2)

