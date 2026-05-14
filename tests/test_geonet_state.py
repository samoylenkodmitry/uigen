import torch

from models.geonet80 import STATE_ANCHORS, GeoNet80, sample_state_features


def test_state_anchor_sampling_uses_3x3_average_pooling():
    p = torch.zeros(1, 2, 5, 5)
    p[:, :, 1:4, 1:4] = 9.0
    rects = torch.zeros(1, 80, 5)
    rects[:, STATE_ANCHORS[0], :] = torch.tensor([0.4, 0.4, 0.6, 0.6, 1.0])

    sampled = sample_state_features(p, rects, jitter=False)

    assert torch.allclose(sampled[0, 0], torch.tensor([9.0, 9.0]))
    assert torch.allclose(sampled[0, 1], torch.tensor([0.0, 0.0]))


def test_geonet_forward_outputs_expected_keys_on_small_input():
    model = GeoNet80(base_channels=8, fpn_channels=16)
    image = torch.rand(1, 3, 64, 64)
    rects = torch.zeros(1, 80, 5)
    rects[:, 0, :] = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])

    out = model(image, anchor_rects=rects)

    assert out["heatmap"].shape[:2] == (1, 80)
    assert out["wh"].shape[1] == 160
    assert out["offset"].shape[1] == 160
    assert out["state"].shape == (1, 32)

