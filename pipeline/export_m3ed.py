import argparse
import json
import math
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np


def window(events, ms, a_us, b_us):
    lo_ms = max(0, a_us // 1000)
    hi_ms = max(0, b_us // 1000 + 1)
    lo = int(ms[min(lo_ms, len(ms) - 1)])
    hi = int(ms[hi_ms]) if hi_ms < len(ms) else len(events["t"])
    t = events["t"][lo:hi].astype(np.int64)
    keep = (t >= a_us) & (t < b_us)
    x = events["x"][lo:hi][keep].astype(np.int32)
    y = events["y"][lo:hi][keep].astype(np.int32)
    p = events["p"][lo:hi][keep].astype(np.int32)
    if np.any((x < 0) | (x >= 1280) | (y < 0) | (y >= 720)):
        raise ValueError("Event coordinates exceed the 1280x720 source resolution")
    x = x // 2
    y = y // 2 + 60
    if np.any((x < 0) | (x >= 640) | (y < 0) | (y >= 480)):
        raise ValueError("Mapped event coordinates exceed the output canvas")
    if np.any((x > 1023) | (y > 511) | ((p != 0) & (p != 1))):
        raise ValueError("Event coordinates or polarity do not fit x10 y9 p1 packing")
    return x, y, p, t[keep]


def count_image(x, y):
    counts = np.zeros((480, 640), np.float32)
    np.add.at(counts, (y, x), 1)
    return counts


def camera_remapping(events, rgb):
    efx, efy, ecx, ecy = events["calib/intrinsics"][:]
    rfx, rfy, rcx, rcy = rgb["calib/intrinsics"][:]
    intrinsics = np.array([efx, efy, ecx, ecy, rfx, rfy, rcx, rcy])
    if not np.all(np.isfinite(intrinsics)) or min(efx, efy, rfx, rfy) <= 0:
        raise ValueError("Invalid camera intrinsics")
    rotation = rgb["calib/T_to_prophesee_left"][:3, :3]
    if not np.all(np.isfinite(rotation)):
        raise ValueError("Invalid camera rotation")
    yy, xx = np.indices((360, 640), dtype=np.float64)
    native_x = (xx + 0.5) / 0.5 - 0.5
    native_y = (yy + 0.5) / 0.5 - 0.5
    rays = np.stack(((native_x - ecx) / efx, (native_y - ecy) / efy, np.ones_like(xx)))
    rays = rotation.T @ rays.reshape(3, -1)
    valid = rays[2] > 0
    map_x = np.full(rays.shape[1], -1, dtype=np.float32)
    map_y = np.full(rays.shape[1], -1, dtype=np.float32)
    map_x[valid] = rfx * rays[0, valid] / rays[2, valid] + rcx
    map_y[valid] = rfy * rays[1, valid] / rays[2, valid] + rcy
    return map_x.reshape(360, 640), map_y.reshape(360, 640)


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="car_urban_night_penno_small_loop_darker")
    ap.add_argument("--root", required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paint-fps", type=int, default=10)
    ap.add_argument("--bin-fps", type=int, default=30)
    ap.add_argument("--cap", type=int, default=16000)
    ap.add_argument("--film-cap", type=int, default=60000)
    a = ap.parse_args()
    if not math.isfinite(a.seconds) or a.seconds <= 0:
        ap.error("--seconds must be finite and positive")
    if min(a.paint_fps, a.bin_fps, a.cap, a.film_cap) <= 0:
        ap.error("Frame rates and event caps must be positive")
    n_paint = int(a.seconds * a.paint_fps)
    n_bins = int(a.seconds * a.bin_fps)
    if min(n_paint, n_bins) < 1:
        ap.error("Duration must contain at least one paint frame and event bin")
    t1 = a.t0 + int(a.seconds * 1e6)
    source = Path(a.root) / f"{a.seq}_data.h5"
    out = Path(a.out)
    with h5py.File(source, "r") as fh:
        events = fh["prophesee/left"]
        rgb = fh["ovc/rgb"]
        ms = events["ms_map_idx"][:]
        ts = fh["ovc/ts"][:].astype(np.int64)
        if len(events["t"]) == 0 or len(ms) == 0 or len(ts) < 2:
            raise ValueError("Source event or camera timeline is empty")
        if tuple(events["calib/resolution"][:]) != (1280, 720):
            raise ValueError("Expected 1280x720 event resolution")
        rgb_width, rgb_height = map(int, rgb["calib/resolution"][:])
        if rgb["data"].shape != (len(ts), rgb_height, rgb_width, 3):
            raise ValueError("RGB frames and calibration resolution do not match")
        if np.any(np.diff(ts) <= 0):
            raise ValueError("Camera timestamps must be strictly increasing")
        first = int(events["t"][0])
        end = int(events["t"][-1]) + 1
        last_paint = a.t0 + int((n_paint - 1) * 1e6 / a.paint_fps)
        if a.t0 - 50000 < first or max(t1, last_paint + 50000) > end:
            raise ValueError(f"Requested interval and paint margins exceed event range [{first}, {end}) us")
        camera_end = int(ts[-1]) + int(np.median(np.diff(ts)))
        if a.t0 < int(ts[0]) or t1 > camera_end:
            raise ValueError("Requested full duration exceeds the camera timeline")
        i0, i1 = int(np.searchsorted(ts, a.t0)), int(np.searchsorted(ts, t1))
        if i0 == i1:
            raise ValueError("Requested interval contains no camera frame")
        remap = camera_remapping(events, rgb)
        for sub in ("ev10", "backing", "cam"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        for k in range(n_paint):
            tick = a.t0 + int(k * 1e6 / a.paint_fps)
            x, y, _, _ = window(events, ms, tick - 5000, tick + 5000)
            image = (255 * np.clip(np.log1p(count_image(x, y)) / np.log1p(3), 0, 1)).astype(np.uint8)
            write_image(out / "ev10" / f"{k:04d}.png", image)
            x, y, _, _ = window(events, ms, tick - 50000, tick + 50000)
            smoothed = cv2.GaussianBlur(count_image(x, y), (0, 0), 2.0)
            image = (255 * np.clip(np.log1p(smoothed) / np.log1p(6), 0, 1)).astype(np.uint8)
            write_image(out / "backing" / f"{k:04d}.png", image)
        camera_offsets = []
        for n, k in enumerate(range(i0, i1)):
            content = cv2.remap(rgb["data"][k], *remap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            canvas[60:420] = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
            write_image(out / "cam" / f"{n:04d}.png", canvas)
            camera_offsets.append(int(ts[k] - a.t0))
        counts, on, web_idx, film_idx = [], [], [0], [0]
        bin_us = 1e6 / a.bin_fps
        rng = np.random.default_rng(0)
        with (out / "events.bin").open("wb") as web, (out / "events_film.bin").open("wb") as film:
            for b in range(n_bins):
                lo = a.t0 + int(b * bin_us)
                hi = a.t0 + int((b + 1) * bin_us)
                x, y, p, t = window(events, ms, lo, hi)
                counts.append(int(len(x)))
                on.append(int(p.sum()))
                dt = np.clip(((t - lo) / bin_us * 4095).astype(np.int64), 0, 4095)
                word = x.astype(np.uint32) | (y.astype(np.uint32) << 10) | (p.astype(np.uint32) << 19) | (dt.astype(np.uint32) << 20)
                for cap, stream, index in ((a.cap, web, web_idx), (a.film_cap, film, film_idx)):
                    pick = word if len(word) <= cap else word[np.sort(rng.choice(len(word), cap, replace=False))]
                    pick.astype("<u4").tofile(stream)
                    index.append(index[-1] + len(pick))
    meta = {
        "seq": a.seq,
        "t0_us": a.t0,
        "seconds": a.seconds,
        "paint_fps": a.paint_fps,
        "bin_fps": a.bin_fps,
        "n_paint": n_paint,
        "n_bins": n_bins,
        "width": 640,
        "height": 480,
        "counts": counts,
        "on": on,
        "index": web_idx,
        "film_index": film_idx,
        "camera_offsets_us": camera_offsets,
        "packing": "x10 y9 p1 dt12",
        "camera_alignment": "rotation only, approximate",
        "camera_distortion": "not applied",
        "dataset": "M3ED",
        "place": "Philadelphia",
        "source_resolution": [1280, 720],
        "event_transform": {"scale": 0.5, "offset": [0, 60], "content_size": [640, 360]},
        "source_file": str(source),
    }
    with (out / "meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream)
    print(a.out, "paint", n_paint, "cam", len(camera_offsets), "bins", n_bins, "events per second", int(sum(counts) / a.seconds))


if __name__ == "__main__":
    main()
