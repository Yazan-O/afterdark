import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 1024, 768
LOW_SIGMA = 48.0
EXPOSURE_RADIUS = 5
GAIN_MIN, GAIN_MAX = 0.55, 1.8
MARK_TAGS = ("plate", "badge", "text", "sign")
MARK_MAX_AREA = 0.04
MARK_SIGMA = 7.0


def read_paint(path):
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return hide_marks(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32), path)


def hide_marks(img, path):
    mask = np.zeros(img.shape[:2], np.float32)
    for tag in MARK_TAGS:
        cache = path.parent.parent / "sam" / f"{tag}_paint" / f"{path.stem}.json"
        if not cache.exists():
            continue
        for mark in json.loads(cache.read_text()):
            x0, y0, x1, y1 = [v * W / 640 for v in mark["box"]]
            w, h = x1 - x0, y1 - y0
            if w * h > MARK_MAX_AREA * W * H:
                continue
            a, b = int(max(0, x0 - max(w * 0.2, 4))), int(min(W, x1 + max(w * 0.2, 4)))
            c, d = int(max(0, y0 - max(h * 0.3, 3))), int(min(H, y1 + max(h * 0.3, 3)))
            mask[c:d, a:b] = 1.0
    if not mask.any():
        return img
    mask = np.maximum(mask, cv2.GaussianBlur(mask, (0, 0), 2.0))[..., None]
    return mask * cv2.GaussianBlur(img, (0, 0), MARK_SIGMA) + (1 - mask) * img


def luminance(img):
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]


def low(img):
    return cv2.GaussianBlur(luminance(img), (0, 0), LOW_SIGMA) + 2.0


class Flow:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.gx, self.gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    def calc(self, cur, prev):
        a = np.clip(luminance(cur), 0, 255).astype(np.uint8)
        b = np.clip(luminance(prev), 0, 255).astype(np.uint8)
        return self.dis.calc(a, b, None)

    def warp(self, img, flow, scale=1.0):
        return cv2.remap(img, self.gx + scale * flow[..., 0], self.gy + scale * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def hold_missing(frames):
    known = [k for k, f in enumerate(frames) if f is not None]
    if not known:
        raise SystemExit("no paintings to stabilise")
    held = list(frames)
    made = []
    for k, f in enumerate(frames):
        if f is None:
            before = [n for n in known if n < k]
            held[k] = frames[before[-1] if before else known[0]]
            made.append(k)
    return held, made


def exposure_lock(frames):
    lows = [low(f) for f in frames]
    out = []
    for k, f in enumerate(frames):
        a, b = max(0, k - EXPOSURE_RADIUS), min(len(frames), k + EXPOSURE_RADIUS + 1)
        gain = np.clip(np.mean(lows[a:b], axis=0) / lows[k], GAIN_MIN, GAIN_MAX)
        out.append(np.clip(f * gain[..., None], 0, 255))
    return out


def lift(img):
    return 255.0 * np.sqrt(np.clip(img, 0, 255) / 255.0)


def blend(frames, flow, low_new, high_new, sigma, low_scale, high_scale):
    state = frames[0].copy()
    out = [state.copy()]
    for k in range(1, len(frames)):
        cur = frames[k]
        motion = flow.calc(cur, frames[k - 1])
        warped = flow.warp(state, motion)
        back = flow.warp(frames[k - 1], motion)
        err_raw = np.abs(lift(back) - lift(cur)).mean(axis=2)
        err_state = np.abs(lift(warped) - lift(cur)).mean(axis=2)
        conf_low = np.exp(-cv2.GaussianBlur(err_state, (0, 0), sigma * 2) / low_scale)[..., None]
        conf_high = np.exp(-cv2.GaussianBlur(err_raw, (0, 0), 2) / high_scale)[..., None]
        cur_low = cv2.GaussianBlur(cur, (0, 0), sigma)
        warped_low = cv2.GaussianBlur(warped, (0, 0), sigma)
        weight_low = 1 - (1 - low_new) * conf_low
        weight_high = 1 - (1 - high_new) * conf_high
        state = weight_low * cur_low + (1 - weight_low) * warped_low + weight_high * (cur - cur_low) + (1 - weight_high) * (warped - warped_low)
        out.append(state.copy())
    return out


def sharpness(img):
    return float(cv2.Laplacian(luminance(img), cv2.CV_32F).var())


def frame_diff(frames):
    return float(np.mean([np.abs(luminance(a) - luminance(b)).mean() for a, b in zip(frames, frames[1:])]))


def flicker(frames, flow):
    values = []
    for a, b in zip(frames, frames[1:]):
        motion = flow.calc(b, a)
        x = cv2.GaussianBlur(lift(luminance(flow.warp(a, motion))), (0, 0), 2)
        y = cv2.GaussianBlur(lift(luminance(b)), (0, 0), 2)
        values.append(np.abs(x - y).mean())
    return float(np.mean(values))


def stabilise(root, low_new, high_new, sigma, low_scale, high_scale, quality):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    drops = set(json.loads((root / "drop.json").read_text())) if (root / "drop.json").exists() else set()
    n = meta["n_paint"]
    frames = [None if k in drops else read_paint(root / "paint" / f"{k:04d}.webp") for k in range(n)]
    flow = Flow()
    filled, made = hold_missing(frames)
    locked = exposure_lock(filled)
    final = blend(locked, flow, low_new, high_new, sigma, low_scale, high_scale)
    out = root / "stab"
    out.mkdir(exist_ok=True)
    for k, f in enumerate(final):
        ok = cv2.imwrite(str(out / f"{k:04d}.webp"), np.clip(f, 0, 255).astype(np.uint8), [cv2.IMWRITE_WEBP_QUALITY, quality])
        if not ok:
            raise SystemExit(f"could not write stab frame {k}")
    ratio = [sharpness(b) / max(sharpness(a), 1e-6) for a, b in zip(filled, final)]
    report = {"n": n, "filled": made, "dropped": sorted(drops), "params": [low_new, high_new, sigma, low_scale, high_scale], "flicker_raw": flicker(filled, flow), "flicker_locked": flicker(locked, flow), "flicker_stab": flicker(final, flow), "frame_diff_raw": frame_diff(filled), "frame_diff_stab": frame_diff(final), "sharpness_ratio_mean": float(np.mean(ratio)), "sharpness_ratio_min": float(np.min(ratio)), "sharpness_ratio_last10": float(np.mean(ratio[-10:]))}
    (root / "stab_report.json").write_text(json.dumps(report, indent=1))
    print(root.name, json.dumps(report))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moments", nargs="+")
    ap.add_argument("--low-new", type=float, default=0.1)
    ap.add_argument("--high-new", type=float, default=0.4)
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--low-scale", type=float, default=40.0)
    ap.add_argument("--high-scale", type=float, default=20.0)
    ap.add_argument("--quality", type=int, default=80)
    a = ap.parse_args()
    for m in a.moments:
        stabilise(m, a.low_new, a.high_new, a.sigma, a.low_scale, a.high_scale, a.quality)


if __name__ == "__main__":
    main()
