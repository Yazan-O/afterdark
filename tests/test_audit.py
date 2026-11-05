import numpy as np
from PIL import Image

from pipeline import audit as module


def test_supported_event_person_missing_from_generated_is_flagged_without_camera(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    generated.mkdir()
    Image.new("RGB", (640, 480)).save(generated / "0000.png")
    grid = np.zeros((480, 640), np.float32)
    grid[100:200, 200:240] = 2
    events = tmp_path / "events.npz"
    np.savez_compressed(events, grid=grid)
    monkeypatch.setenv("MODEL_API_KEY", "test")

    def segment(path, phrase):
        if path.parent == generated:
            return []
        return [{"width": 640, "height": 480, "box": [200, 100, 239, 199], "score": 0.9, "mask": {}}]

    monkeypatch.setattr(module, "sam_segment", segment)
    result = module.audit(generated, events)
    assert result["errors"] == []
    assert result["processed"] == 1
    assert result["flagged_moments"][0]["frame"] == "0000"
    person = result["flagged_moments"][0]["candidates"][0]
    assert person["box"] == [200, 100, 240, 200]
    assert person["support"]["events"] == 8000
    assert person["support"]["backed"] is True
