import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

WEB = Path(__file__).resolve().parents[1] / "web" / "moments"
SIZES = {"paint": (1024, 768), "cam": (640, 480), "backing": (320, 240)}
FIELDS = ["seq", "t0_us", "seconds", "paint_fps", "bin_fps", "n_paint", "n_bins", "width", "height", "counts", "on", "index", "film_index", "camera_offsets_us", "packing", "n_cam", "n_paint_pub", "place", "title"]
RASTER = 160 * 120


def check(mid):
    bad = []
    out = WEB / mid
    meta = json.loads((out / "meta.json").read_text())
    for f in FIELDS:
        if f not in meta:
            bad.append(f"meta missing {f}")
    n = meta["n_paint"]
    if meta["n_paint_pub"] != n or meta["n_cam"] != n:
        bad.append("n_cam or n_paint_pub differs from n_paint")
    if len(meta["counts"]) != meta["n_bins"] or len(meta["on"]) != meta["n_bins"] or len(meta["index"]) != meta["n_bins"] + 1:
        bad.append("bin arrays do not match n_bins")
    if (out / "events.bin").stat().st_size != meta["index"][-1] * 4:
        bad.append("events.bin length does not match index")
    words = np.fromfile(out / "events.bin", dtype="<u4")
    if ((words & 1023) >= 640).any() or (((words >> 10) & 511) >= 480).any():
        bad.append("events.bin holds an event outside 640x480")
    for sub, size in SIZES.items():
        names = sorted(p.name for p in (out / sub).iterdir())
        if names != [f"{k:04d}.webp" for k in range(n)]:
            bad.append(f"{sub} frames are not exactly 0..{n - 1}")
        for name in names:
            img = cv2.imread(str(out / sub / name))
            if img is None or (img.shape[1], img.shape[0]) != size:
                bad.append(f"{sub}/{name} unreadable or wrong size")
            elif img.mean() < 0.5:
                bad.append(f"{sub}/{name} is black")
    people = json.loads((out / "people.json").read_text())
    if [f["k"] for f in people["frames"]] != list(range(n)):
        bad.append("people.json frames are not 0..n-1")
    total = 0
    for f in people["frames"]:
        for p in f["people"]:
            total += 1
            x0, y0, x1, y1 = p["box"]
            if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480):
                bad.append(f"people k={f['k']} box {p['box']}")
            if not 0 <= p["score"] <= 1:
                bad.append(f"people k={f['k']} score {p['score']}")
            runs = p["mask_rle"]
            if sum(runs) != RASTER or min(runs) < 0 or sum(runs[1::2]) == 0:
                bad.append(f"people k={f['k']} mask runs invalid")
    files = [out / n for n in ("meta.json", "events.bin", "people.json")] + [f for sub in SIZES for f in (out / sub).iterdir()]
    size = sum(f.stat().st_size for f in files)
    extras = sorted(f.name for f in out.iterdir() if f.name not in {"meta.json", "events.bin", "people.json", "agent.json", *SIZES})
    if size > 60e6:
        bad.append(f"moment is {size / 1e6:.1f} MB")
    print(mid, "frames", n, "events", len(words), "people", total, "contract MB", round(size / 1e6, 1), "extras", extras, "FAIL" if bad else "pass")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moments", nargs="*")
    a = ap.parse_args()
    index = json.loads((WEB / "index.json").read_text())["moments"]
    targets = a.moments or index
    bad = []
    if not index or index[0] != "crosswalk":
        bad.append("index.json does not open on crosswalk")
    for mid in targets:
        if mid not in index:
            bad.append(f"{mid} not in index.json")
        bad += check(mid)
    for line in bad:
        print(line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
