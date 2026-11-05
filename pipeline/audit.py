import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from afterdark.meta import sam_segment
from afterdark.moment import H, W, Moment
from afterdark.verdict import MIN_SCORE, from_record, tally
from pipeline.env import load_key


def audit(generated, events):
    root, event_root = Path(generated), Path(events)
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not images:
        raise ValueError("Generated folder contains no images")
    if not os.environ.get("MODEL_API_KEY"):
        os.environ["MODEL_API_KEY"] = load_key()
    moment = Moment(event_root) if (event_root / "meta.json").exists() else None
    out = {"flagged_moments": [], "processed": 0, "errors": [], "candidate_source": "SAM person on 10 ms events", "meaning": "event-supported person candidates without a generated-image person match"}
    for path in images:
        try:
            if moment is not None:
                grid = moment.event_map(int(path.stem), 10)
            else:
                source = event_root / (path.stem + ".npz") if event_root.is_dir() else event_root
                with np.load(source, allow_pickle=False) as data:
                    key = path.stem if path.stem in data else "grid"
                    if key == "grid" and not event_root.is_dir() and len(images) != 1:
                        raise ValueError("A single grid requires exactly one generated image")
                    grid = data[key].copy()
            if grid.shape != (H, W) or not np.isfinite(grid).all() or (grid < 0).any():
                raise ValueError("Events must be a finite nonnegative 480 by 640 count grid")
            with tempfile.TemporaryDirectory(prefix="afterdark_audit_") as tmp:
                event_image = Path(tmp) / "events.png"
                pixels = (255 * np.clip(np.log1p(grid) / np.log(4), 0, 1)).astype(np.uint8)
                Image.fromarray(pixels).save(event_image)
                candidates = [from_record(r) for r in sam_segment(event_image, "person") if r["score"] >= MIN_SCORE]
                painted = [from_record(r) for r in sam_segment(path, "person") if r["score"] >= MIN_SCORE]
            result = tally(painted, candidates, grid, np.zeros_like(grid))
            flags = [q for q in result["camera_no_painted"] if q["support"]["backed"]]
            if flags:
                out["flagged_moments"].append({"frame": path.stem, "candidates": flags})
            out["processed"] += 1
        except Exception as error:
            out["errors"].append({"frame": path.stem, "error": type(error).__name__ + ": " + str(error)})
    return out
