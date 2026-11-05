import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FIT_W, FIT_H = 512, 384
STEP = 8


def gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (FIT_W, FIT_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return np.clip(np.sqrt(img) * 255, 0, 255).astype(np.uint8)


def fit_pair(dis, a, b):
    flow = dis.calc(a, b, None)
    back = dis.calc(b, a, None)
    ys, xs = np.mgrid[STEP // 2:FIT_H:STEP, STEP // 2:FIT_W:STEP]
    fx, fy = flow[ys, xs, 0], flow[ys, xs, 1]
    xe = np.clip(xs + fx, 0, FIT_W - 1).astype(int)
    ye = np.clip(ys + fy, 0, FIT_H - 1).astype(int)
    err = np.hypot(fx + back[ye, xe, 0], fy + back[ye, xe, 1])
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
    keep = (err < 1.5) & (np.hypot(gx, gy)[ys, xs] > 12)
    if keep.sum() < 40:
        return [1.0, 0.0, 0.0], 0
    src = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
    dst = src + np.stack([fx[keep], fy[keep]], 1)
    m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=4000)
    if m is None or inl is None or inl.sum() < 30:
        return [1.0, 0.0, 0.0], 0
    s = float(np.hypot(m[0, 0], m[1, 0]))
    cx, cy = FIT_W / 2, FIT_H / 2
    tx = float(m[0, 2] + (m[0, 0] - 1) * cx + m[0, 1] * cy) / FIT_W
    ty = float(m[1, 2] + m[1, 0] * cx + (m[1, 1] - 1) * cy) / FIT_H
    s = float(np.clip(s, 0.9, 1.15))
    return [s, float(np.clip(tx, -0.06, 0.06)), float(np.clip(ty, -0.06, 0.06))], int(inl.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("moment_dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--stride", type=int, default=2)
    o = p.parse_args()
    meta = json.loads((o.moment_dir / "meta.json").read_text())
    n = meta["n_paint"]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    pairs = []
    support = []
    a = gray(o.moment_dir / "paint" / "0000.webp")
    k = 0
    while k + o.stride < n:
        path = o.moment_dir / "paint" / f"{k + o.stride:04d}.webp"
        if not path.exists():
            pairs.append([1.0, 0.0, 0.0])
            support.append(0)
            k += o.stride
            continue
        b = gray(path)
        t, c = fit_pair(dis, a, b)
        pairs.append([round(v, 5) for v in t])
        support.append(c)
        a = b
        k += o.stride
    out = o.out or o.moment_dir / "drift.json"
    out.write_text(json.dumps({"stride": o.stride, "pairs": pairs}))
    arr = np.array(pairs)
    print(json.dumps({"pairs": len(pairs), "median_scale": float(np.median(arr[:, 0])), "median_abs_tx": float(np.median(np.abs(arr[:, 1]))), "median_abs_ty": float(np.median(np.abs(arr[:, 2]))), "median_support": float(np.median(support)), "no_fit": int((np.array(support) == 0).sum())}))


if __name__ == "__main__":
    main()
