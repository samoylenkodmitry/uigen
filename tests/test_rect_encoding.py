from atlas_ai.rects import derive_eq_band_rects, encode_rect


def test_encode_rect_clips_and_normalizes_visible_rect():
    encoded = encode_rect(-10, 10, 100, 50, canvas_w=200, canvas_h=100)

    assert encoded == (0.0, 0.1, 0.5, 0.5, 1.0)


def test_encode_rect_zeroes_tiny_or_invisible_rects():
    assert encode_rect(1, 1, 2, 2, min_visible_area=4) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert encode_rect(2, 2, 1, 1) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_eq_band_derivation_is_deterministic_equal_subdivision():
    bands = derive_eq_band_rects((0.2, 0.3, 0.7, 0.9, 1.0))

    assert len(bands) == 10
    assert bands[0] == (0.2, 0.3, 0.25, 0.9, 1.0)
    assert bands[-1] == (0.65, 0.3, 0.7, 0.9, 1.0)
    assert bands == derive_eq_band_rects((0.2, 0.3, 0.7, 0.9, 1.0))

