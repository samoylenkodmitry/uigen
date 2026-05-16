import torch

from models.slotnet_v3 import SlotNetV3


def test_slotnet_v3_forward_returns_full_atlas():
    """V3 input is just the view; output is the whole 1024x1024 atlas."""
    model = SlotNetV3(base_channels=8)
    # Encoder has 5 stride-2 stages so input must be divisible by 32. Keep
    # it small for the test.
    view = torch.rand(1, 3, 64, 64)
    out = model(view)
    assert out["prediction"].shape == (1, 7, 1024, 1024)


def test_slotnet_v3_inputs_only_view():
    """Forward must accept only the view -- no slot_id, rect, state."""
    model = SlotNetV3(base_channels=8)
    out = model(torch.rand(1, 3, 32, 32))
    assert "prediction" in out
