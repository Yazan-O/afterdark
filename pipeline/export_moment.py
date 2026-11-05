import argparse
import glob
import json
import os

import cv2
import h5py
import hdf5plugin
import numpy as np

from dsec_util import (
    calib,
    compute_remapping,
    remap_img,
)


def window(fh, ms, off, a_us, b_us):
    a, b = a_us - off, b_us - off
    lo = int(ms[max(0, a // 1000)])
    hi = int(ms[min(b // 1000 + 1, len(ms) - 1)])
    E = fh["events"]
    t = E["t"][lo:hi].astype(np.int64)
    keep = (t >= a) & (t < b)
    return E["x"][lo:hi][keep].astype(np.int32), E["y"][lo:hi][keep].astype(np.int32), E["p"][lo:hi][keep].astype(np.int32), t[keep] + off


def count_image(x, y):
    c = np.zeros((480, 640), np.float32)
    np.add.at(c, (y, x), 1)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paint-fps", type=int, default=10)
    ap.add_argument("--bin-fps", type=int, default=30)
    ap.add_argument("--cap", type=int, default=16000)
    ap.add_argument("--film-cap", type=int, default=60000)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = f"{a.root}/{a.seq}"
    for sub in ("ev10", "backing", "cam"):
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    fh = h5py.File(f"{root}/ev/events.h5", "r")
    off = int(fh["t_offset"][()])
    ms = fh["ms_to_idx"][:]
    t1 = a.t0 + int(a.seconds * 1e6)
    n_paint = int(a.seconds * a.paint_fps)
    for k in range(n_paint):
        t = a.t0 + int(k * 1e6 / a.paint_fps)
        x, y, _, _ = window(fh, ms, off, t - 5000, t + 5000)
        v = (255 * np.clip(np.log1p(count_image(x, y)) / np.log1p(3), 0, 1)).astype(np.uint8)
        cv2.imwrite(f"{a.out}/ev10/{k:04d}.png", v)
        x, y, _, _ = window(fh, ms, off, t - 50000, t + 50000)
        s = cv2.GaussianBlur(count_image(x, y), (0, 0), 2.0)
        cv2.imwrite(f"{a.out}/backing/{k:04d}.png", (255 * np.clip(np.log1p(s) / np.log1p(6), 0, 1)).astype(np.uint8))
    ts = np.loadtxt(f"{root}/image_timestamps.txt", dtype=np.int64)
    imgs = sorted(glob.glob(f"{root}/img/*.png"))
    rm = compute_remapping(calib(a.seq), h5py.File(f"{root}/ev/rectify_map.h5", "r")["rectify_map"][:])
    i0, i1 = int(np.searchsorted(ts, a.t0)), int(np.searchsorted(ts, t1))
    cam = []
    for n, k in enumerate(range(i0, min(i1, len(imgs)))):
        cv2.imwrite(f"{a.out}/cam/{n:04d}.png", remap_img(cv2.imread(imgs[k]), rm))
        cam.append(int(ts[k] - a.t0))
    n_bins = int(a.seconds * a.bin_fps)
    counts, on = [], []
    web, film, web_idx, film_idx = [], [], [0], [0]
    bin_us = 1e6 / a.bin_fps
    rng = np.random.default_rng(0)
    for b in range(n_bins):
        lo = a.t0 + int(b * bin_us)
        x, y, p, t = window(fh, ms, off, lo, a.t0 + int((b + 1) * bin_us))
        counts.append(int(len(x)))
        on.append(int(p.sum()))
        dt = np.clip(((t - lo) / bin_us * 4095).astype(np.int64), 0, 4095)
        word = (x.astype(np.uint32) & 1023) | ((y.astype(np.uint32) & 511) << 10) | ((p.astype(np.uint32) & 1) << 19) | (dt.astype(np.uint32) << 20)
        for cap, store, idx in ((a.cap, web, web_idx), (a.film_cap, film, film_idx)):
            pick = word if len(word) <= cap else word[np.sort(rng.choice(len(word), cap, replace=False))]
            store.append(pick)
            idx.append(idx[-1] + len(pick))
    np.concatenate(web).astype("<u4").tofile(f"{a.out}/events.bin")
    np.concatenate(film).astype("<u4").tofile(f"{a.out}/events_film.bin")
    json.dump({"seq": a.seq, "t0_us": a.t0, "seconds": a.seconds, "paint_fps": a.paint_fps, "bin_fps": a.bin_fps, "n_paint": n_paint, "n_bins": n_bins, "width": 640, "height": 480, "counts": counts, "on": on, "index": web_idx, "film_index": film_idx, "camera_offsets_us": cam, "packing": "x10 y9 p1 dt12"}, open(f"{a.out}/meta.json", "w"))
    print(a.out, "paint", n_paint, "cam", len(cam), "bins", n_bins, "events per second", int(sum(counts) / a.seconds))


main()
