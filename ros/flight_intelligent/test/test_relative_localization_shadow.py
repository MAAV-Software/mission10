import pytest

from flight_intelligent.relative_localization_shadow import _peer_id_map


def test_explicit_peer_ids_decouple_identity_from_namespace_suffix():
    assert _peer_id_map(["px4_3"], [0]) == {0: "px4_3"}
    assert _peer_id_map(["px4_3", "px4_1"], [0, 1]) == {
        0: "px4_3",
        1: "px4_1",
    }


def test_legacy_peer_ids_follow_namespace_suffix():
    assert _peer_id_map(["px4_3"], [-1]) == {3: "px4_3"}


@pytest.mark.parametrize("ids", [[], [0, 1], [-2]])
def test_invalid_explicit_peer_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        _peer_id_map(["px4_3"], ids)
