import json

import pytest

from pipeline import pack_audit as module

COUNTS = {"camera": 2, "painted": 1, "in_both": 0, "left_out": 2, "invented": 1}


def box(xy, area, backed, band):
    return {"box": xy, "area": area, "band_pct": band, "support": {"backed": backed}}


def doc(status="ok"):
    left = [box([0, 0, 40, 40], 1600, True, 0.95), box([50, 0, 90, 40], 1600, True, 0.45)]
    cell = {"status": status, "counts": COUNTS, "raw_muse_counts": COUNTS, "left_out": left}
    return {"moment": "m", "n_frames": 1, "frames": {"0": {"concepts": {"person": cell}}}}


def test_pack_keeps_counts_and_flags_only_the_box_that_beats_its_row():
    packed = module.pack(doc())
    frame = packed["frames"][0]
    assert frame["camera"] == {"person": 2}
    assert frame["invented"] == 1
    assert [o["activity"] for o in frame["left_out"]] == [True, False]
    assert packed["totals"]["person"]["left_out"] == 2


def test_pack_stops_on_a_frame_without_a_verdict():
    with pytest.raises(ValueError):
        module.pack(doc("refused"))


def test_shipped_crosswalk_does_not_count_the_bollard():
    path = module.WEB / "crosswalk" / "audit.json"
    frames = json.loads(path.read_text())["frames"]
    assert [394, 220, 402, 240] not in [o["box"] for o in frames[30]["left_out"]]
    assert sum(frames[30]["camera"].values()) >= len(frames[30]["left_out"])
