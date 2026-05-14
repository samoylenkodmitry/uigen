from models.atlas import magenta_policy_for_slot


def test_magenta_policy_uses_per_slot_then_default():
    policy = {"default": False, "per_slot": {"CBUTTONS": True}}

    assert magenta_policy_for_slot(policy, "CBUTTONS") is True
    assert magenta_policy_for_slot(policy, "MAIN") is False

