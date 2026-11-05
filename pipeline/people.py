import argparse
import json
from pathlib import Path

import numpy as np

from sam_common import EH, EW, RH, RW, brighten, cam_path, rle_decode, rle_encode, sweep

MAX_GAP = 3
IOU_MATCH = 0.2
DUPLICATE_IOU = 0.5


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def shifted(runs, dx, dy):
    mask = rle_decode(runs)
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    ys, xs = ys + int(round(dy * RH / EH)), xs + int(round(dx * RW / EW))
    keep = (ys >= 0) & (ys < RH) & (xs >= 0) & (xs < RW)
    out[ys[keep], xs[keep]] = 1
    return rle_encode(out)


def gap_fill(frames):
    n = len(frames)
    filled = [list(f) for f in frames]
    count = 0
    for k in range(n):
        for person in frames[k]:
            for gap in range(1, MAX_GAP + 1):
                j = k + gap + 1
                if j >= n:
                    break
                matches = [p for p in frames[j] if iou(person["box"], p["box"]) >= IOU_MATCH / gap]
                if not matches:
                    continue
                other = max(matches, key=lambda p: iou(person["box"], p["box"]))
                for m in range(k + 1, j):
                    t = (m - k) / (j - k)
                    box = [round((1 - t) * a + t * b, 1) for a, b in zip(person["box"], other["box"])]
                    if any(iou(box, p["box"]) >= IOU_MATCH for p in frames[m]):
                        continue
                    dx, dy = box[0] - person["box"][0], box[1] - person["box"][1]
                    filled[m].append({"box": box, "score": round(min(person["score"], other["score"]), 3), "mask_rle": shifted(person["mask_rle"], dx, dy)})
                    count += 1
                break
    return filled, count


def load_remove(root):
    path = Path(root) / "people_remove.json"
    if not path.exists():
        return {}
    return {(int(k), tuple(e["box"])): e["what"] for k, v in json.loads(path.read_text()).items() for e in v}


def dedupe(people):
    kept = []
    for p in sorted(people, key=lambda p: -p["score"]):
        if all(iou(p["box"], q["box"]) < DUPLICATE_IOU for q in kept):
            kept.append(p)
    return kept


def run(root):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    n = meta["n_paint"]
    paths = {k: cam_path(root, meta, k) for k in range(n)}
    found = sweep(paths, "person", root / "sam" / "person_cam", prepare=brighten)
    remove = load_remove(root)
    real = [[p for p in found[k] if (k, tuple(p["box"])) not in remove] for k in range(n)]
    removed = sum(len(found[k]) - len(real[k]) for k in range(n))
    filled, added = gap_fill(real)
    filled = [dedupe(f) for f in filled]
    out = {"frames": [{"k": k, "people": filled[k]} for k in range(n)]}
    (root / "people.json").write_text(json.dumps(out, separators=(",", ":")))
    print(root.name, "frames", n, "frames with people", sum(1 for f in filled if f), "people", sum(len(f) for f in filled), "gap-filled", added, "removed", removed)


def run_paint(root):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    n = meta["n_paint"]
    paths = {k: root / "paint" / f"{k:04d}.webp" for k in range(n) if (root / "paint" / f"{k:04d}.webp").exists()}
    found = sweep(paths, "person", root / "sam" / "person_paint")
    out = {"frames": [{"k": k, "people": found[k]} for k in sorted(found)]}
    (root / "paint_people.json").write_text(json.dumps(out, separators=(",", ":")))
    print(root.name, "paintings", len(found), "with people", sum(1 for k in found if found[k]), "people", sum(len(v) for v in found.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moments", nargs="+")
    ap.add_argument("--cam-only", action="store_true")
    a = ap.parse_args()
    for m in a.moments:
        run(m)
        if not a.cam_only:
            run_paint(m)


if __name__ == "__main__":
    main()
