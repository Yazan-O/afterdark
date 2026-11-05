import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from people import load_remove
from sam_common import RW, EW, brighten, cam_path, rle_decode, sweep

HEAD_FRACTION = 0.15
MIN_HEAD = 6.0
SIGMA_FRACTION = 0.26
MIN_SIGMA = 2.0


def head_ellipse(person):
    x0, y0, x1, y1 = person["box"]
    head = max((y1 - y0) * HEAD_FRACTION, MIN_HEAD)
    cell = EW / RW
    band = rle_decode(person["mask_rle"])[int(y0 / cell):int((y0 + head) / cell) + 1]
    cols = np.nonzero(band.any(axis=0))[0]
    cx = (cols.min() + cols.max() + 1) / 2 * cell if len(cols) else (x0 + x1) / 2
    return cx, y0 + head * 0.45, head * 0.56, head * 0.66, max(head * SIGMA_FRACTION, MIN_SIGMA)


def box_ellipse(box, pad=1.25):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (x0 + x1) / 2, (y0 + y1) / 2, w / 2 * pad, h / 2 * pad, max(min(w, h) * SIGMA_FRACTION, MIN_SIGMA)


def soften(img, ellipses):
    out = img.astype(np.float32)
    for cx, cy, ax, ay, sigma in ellipses:
        reach = int(max(ax, ay) + 3 * sigma + 4)
        a, b = max(0, int(cx) - reach), min(img.shape[1], int(cx) + reach)
        c, d = max(0, int(cy) - reach), min(img.shape[0], int(cy) + reach)
        if a >= b or c >= d:
            continue
        roi = out[c:d, a:b]
        mask = np.zeros(roi.shape[:2], np.float32)
        cv2.ellipse(mask, (round((cx - a) * 8), round((cy - c) * 8)), (round(ax * 8), round(ay * 8)), 0, 0, 360, 1.0, -1, cv2.LINE_AA, 3)
        mask = cv2.GaussianBlur(mask, (0, 0), max(sigma * 0.35, 0.8))[..., None]
        out[c:d, a:b] = mask * cv2.GaussianBlur(roi, (0, 0), sigma) + (1 - mask) * roi
    return np.clip(out, 0, 255).astype(np.uint8)


def run(root, quality):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    n = meta["n_paint"]
    paths = {k: cam_path(root, meta, k) for k in range(n)}
    raw = sweep(paths, "person", root / "sam" / "person_cam", prepare=brighten)
    faces_found = sweep(paths, "face", root / "sam" / "face_cam", prepare=brighten)
    plates = sweep(paths, "license plate", root / "sam" / "plate_cam", prepare=brighten)
    people = json.loads((root / "people.json").read_text())["frames"]
    remove = load_remove(root)
    out = root / "cam_pub"
    out.mkdir(exist_ok=True)
    stats = {"heads": 0, "pictures": 0, "faces": 0, "plates": 0}
    for k in range(n):
        heads = [head_ellipse(p) for p in people[k]["people"]]
        pictures = [box_ellipse(p["box"], 1.1) for p in raw[k] if remove.get((k, tuple(p["box"])), "bollard") != "bollard"]
        faces = [box_ellipse(p["box"]) for p in faces_found[k]]
        cars = [box_ellipse(p["box"]) for p in plates[k]]
        stats["heads"] += len(heads)
        stats["pictures"] += len(pictures)
        stats["faces"] += len(faces)
        stats["plates"] += len(cars)
        img = soften(cv2.imread(str(paths[k])), heads + pictures + faces + cars)
        if not cv2.imwrite(str(out / f"{k:04d}.webp"), img, [cv2.IMWRITE_WEBP_QUALITY, quality]):
            raise SystemExit(f"could not write cam frame {k}")
    (root / "blur_report.json").write_text(json.dumps(stats))
    print(root.name, n, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moments", nargs="+")
    ap.add_argument("--quality", type=int, default=80)
    a = ap.parse_args()
    for m in a.moments:
        run(m, a.quality)


if __name__ == "__main__":
    main()
