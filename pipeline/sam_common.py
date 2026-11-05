from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import cv2
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import load_key

os.environ["MODEL_API_KEY"] = load_key()

from afterdark.meta import MetaAPIError, sam_segment
from afterdark.spend import SpendStop

RW, RH = 160, 120
EW, EH = 640, 480
GAMMA = 0.45
_table = np.array([(255 * (i / 255.0) ** GAMMA) for i in range(256)], dtype=np.uint8)


def brighten(img):
    return cv2.LUT(img, _table)


def cam_index(meta, k):
    t = k * 1e6 / meta["paint_fps"]
    return int(np.argmin(np.abs(np.array(meta["camera_offsets_us"], dtype=np.int64) - t)))


def cam_path(root, meta, k):
    return Path(root) / "cam" / f"{cam_index(meta, k):04d}.png"


def segment_retry(path, phrase):
    last = None
    for attempt in range(6):
        try:
            return sam_segment(path, phrase)
        except SpendStop:
            raise
        except MetaAPIError as e:
            last = e
            time.sleep(2 ** attempt)
    raise MetaAPIError(f"SAM failed after retries on {Path(path).name}: {last}")


def decode(record):
    m = record["mask"]
    marker = "~" if m["encoding"] == "lossless" else "!"
    raster = decode_mask_to_raster(SegmentationMask(encoding=m["encoding"], payload=marker + m["data"], width=m["width"], height=m["height"]))
    return np.frombuffer(raster, dtype=np.uint8).reshape(m["height"], m["width"])


def rle_encode(mask):
    flat = mask.astype(np.uint8).ravel()
    runs = []
    value, count = 0, 0
    for v in flat:
        if v == value:
            count += 1
        else:
            runs.append(count)
            value, count = int(v), 1
    runs.append(count)
    return runs


def rle_decode(runs):
    flat = np.zeros(RW * RH, np.uint8)
    pos, value = 0, 0
    for r in runs:
        if value:
            flat[pos:pos + r] = 1
        pos += r
        value = 1 - value
    return flat.reshape(RH, RW)


def to_event_record(record):
    sx, sy = EW / record["width"], EH / record["height"]
    x1, y1, x2, y2 = record["box"]
    box = [round(x1 * sx, 1), round(y1 * sy, 1), round((x2 + 1) * sx, 1), round((y2 + 1) * sy, 1)]
    full = np.zeros((record["height"], record["width"]), np.uint8)
    local = decode(record)
    h = min(local.shape[0], full.shape[0] - y1)
    w = min(local.shape[1], full.shape[1] - x1)
    full[y1:y1 + h, x1:x1 + w] = local[:h, :w]
    small = cv2.resize(full.astype(np.float32), (RW, RH), interpolation=cv2.INTER_AREA) >= 0.25
    if not small.any():
        cx, cy = int((box[0] + box[2]) / 2 * RW / EW), int((box[1] + box[3]) / 2 * RH / EH)
        small[min(cy, RH - 1), min(cx, RW - 1)] = True
    return {"box": box, "score": round(record["score"], 3), "mask_rle": rle_encode(small)}


def sweep(paths, phrase, cache_dir, prepare=None, workers=6):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(item):
        k, path = item
        out = cache_dir / f"{k:04d}.json"
        if out.exists():
            return k, json.loads(out.read_text())
        if prepare is None:
            records = segment_retry(path, phrase)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "frame.png"
                cv2.imwrite(str(p), prepare(cv2.imread(str(path))))
                records = segment_retry(p, phrase)
        people = [to_event_record(r) for r in records]
        out.write_text(json.dumps(people))
        return k, people

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, list(paths.items())))
