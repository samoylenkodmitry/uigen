import numpy as np

from atlas_ai.provenance import decode_provenance, encode_provenance, provenance_to_visible_mask


def test_provenance_id_round_trips_slot_and_atlas_coordinates():
    value = encode_provenance(slot_id=12, atlas_x=640, atlas_y=512)

    assert decode_provenance(value) == (12, 640, 512)
    assert decode_provenance(0) is None


def test_visible_mask_decodes_unique_provenance_ids():
    buffer = np.array([[0, encode_provenance(1, 2, 3)], [encode_provenance(4, 5, 6), 0]], dtype=np.uint32)

    mask = provenance_to_visible_mask(buffer)

    assert mask[3, 2] == 255
    assert mask[6, 5] == 255
    assert mask[0, 0] == 0

