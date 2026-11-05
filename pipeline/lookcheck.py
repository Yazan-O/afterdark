import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np

from sam_common import MetaAPIError, SpendStop, brighten

from afterdark.meta import spark_ask
from env import load_key
from paint_moment import PROMPT, paint

QUESTION = ("This image is one painted frame of a night street. Look closely and answer in JSON only, with these keys: "
            '{"woman": true or false, "child": true or false, "face": true or false, "text": true or false, "logo": true or false, "why": "one short sentence"}. '
            "woman: any person who looks female. child: any person who looks under 18. face: any person whose facial features are visible. "
            "text: any readable words or numbers. logo: any recognisable brand mark.")
REPAINT_NOTE = " Every person is an adult man with short hair, in dark plain clothes, turned away from the viewer so no face shows. Nobody with long hair, nobody in a light coat, no bright high-visibility clothing. No lettering, numbers or lit characters on any vehicle, plate, panel or sign; number plates and destination panels are blank."
KEYS = ("woman", "child", "face", "text", "logo")


def ask(path):
    last = None
    for attempt in range(3):
        try:
            text = spark_ask([path], QUESTION, max_completion_tokens=4000)
            match = re.search(r"\{.*\}", text, re.S)
            answer = json.loads(match[0])
            return {**{k: bool(answer.get(k)) for k in KEYS}, "why": str(answer.get("why", ""))}
        except SpendStop:
            raise
        except (MetaAPIError, ValueError, TypeError) as e:
            last = e
    return {"error": str(last)[:200]}


def spark(root, only=None, workers=6):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    path = root / "lookcheck.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    todo = [k for k in range(meta["n_paint"]) if (f"{k:04d}" not in results or "error" in results[f"{k:04d}"]) or (only is not None and k in only)]
    with ThreadPoolExecutor(workers) as pool:
        for k, r in zip(todo, pool.map(lambda k: ask(root / "stab" / f"{k:04d}.webp"), todo)):
            results[f"{k:04d}"] = r
    path.write_text(json.dumps(results, indent=0))
    flagged = {k: v for k, v in results.items() if any(v.get(x) for x in KEYS)}
    errors = [k for k, v in results.items() if "error" in v]
    print(root.name, "asked", len(todo), "flags", len(flagged), "by key", {x: sum(1 for v in results.values() if v.get(x)) for x in KEYS}, "errors", errors)
    for k, v in sorted(flagged.items()):
        print(k, [x for x in KEYS if v.get(x)], v.get("why", "")[:110])


def label(img, text):
    cv2.rectangle(img, (0, 0), (46, 15), (0, 0, 0), -1)
    cv2.putText(img, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(tiles, per_row, path, quality):
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack((tiles[i:i + per_row] + [blank] * per_row)[:per_row]) for i in range(0, len(tiles), per_row)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, quality])


def sheets(root, out, per_sheet=40, quality=72):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = json.loads((root / "meta.json").read_text())["n_paint"]
    for sub, tag, change in (("stab", "paint", None), ("cam_pub", "cam", brighten)):
        for start in range(0, n, per_sheet):
            tiles = []
            for k in range(start, min(start + per_sheet, n)):
                img = cv2.imread(str(root / sub / f"{k:04d}.webp"))
                img = cv2.resize(img, (240, 180), interpolation=cv2.INTER_AREA)
                tiles.append(label(change(img) if change else img, str(k)))
            grid(tiles, 8, out / f"{root.name}_{tag}_{start:04d}.jpg", quality)
    crops = []
    people = json.loads((root / "paint_people.json").read_text())
    for f in people["frames"]:
        img = cv2.imread(str(root / "paint" / f"{f['k']:04d}.webp"))
        sx, sy = img.shape[1] / 640, img.shape[0] / 480
        for p in f["people"]:
            x0, y0, x1, y1 = p["box"]
            h = y1 - y0
            top = max(0, int((y0 - h * 0.05) * sy))
            bottom = int((y0 + max(h * 0.5, 10)) * sy)
            left, right = max(0, int((x0 - 2) * sx)), int((x1 + 2) * sx)
            crop = img[top:bottom, left:right]
            if crop.size == 0:
                continue
            scale = 120 / max(crop.shape[0], crop.shape[1])
            small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
            tile = np.zeros((120, 120, 3), np.uint8)
            tile[:small.shape[0], :small.shape[1]] = small
            crops.append(label(brighten(tile), str(f["k"])))
    for i in range(0, len(crops), 160):
        grid(crops[i:i + 160], 16, out / f"{root.name}_people_{i:04d}.jpg", quality)
    print(root.name, "people crops", len(crops))


def repaint(root, frames, style, note=False, extra=""):
    root = Path(root)
    key = load_key()
    prompt = (PROMPT + REPAINT_NOTE if note else PROMPT) + extra
    new = root / f"paint_new_{os.getpid()}"
    new.mkdir(exist_ok=True)

    def one(k):
        name = f"{k:04d}.webp"
        result = paint(key, root / "ev10" / f"{k:04d}.png", style, new / name, prompt=prompt)
        if isinstance(result, float):
            os.replace(new / name, root / "paint" / name)
            for tag in ("person", "plate", "badge", "text", "sign"):
                (root / "sam" / f"{tag}_paint" / f"{k:04d}.json").unlink(missing_ok=True)
        return k, result

    with ThreadPoolExecutor(6) as pool:
        for k, r in pool.map(one, frames):
            print(k, r if isinstance(r, float) else "failed: " + str(r)[:120])
    new.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moment")
    ap.add_argument("--spark", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--sheets")
    ap.add_argument("--repaint", default="")
    ap.add_argument("--style")
    ap.add_argument("--note", action="store_true")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x} or None
    if a.repaint:
        repaint(a.moment, [int(x) for x in a.repaint.split(",")], a.style, a.note, a.extra)
    if a.spark:
        spark(a.moment, only)
    if a.sheets:
        sheets(a.moment, a.sheets)


if __name__ == "__main__":
    main()
