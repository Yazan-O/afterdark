import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import time

import httpx

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afterdark import spend
from env import load_key

PROMPT = ("The first image is an event-camera recording from a car at night: bright pixels mark where light changed during 10 milliseconds. "
          "Paint the real scene as one frame of a night drive filmed from the driver's seat. Every object, car, lane line, tree, lamp and building must sit exactly where the first image shows it. "
          "The second image comes from a different street: it only sets the colours and the lighting mood, copy nothing from its layout. "
          "Dry asphalt, no rain, no puddles, no reflections on the road. The frame shows only the view through the glass: no dashboard, no windscreen, no bonnet. "
          "If the first image shows a person, paint an adult man in dark clothes seen from a distance with his face in shadow. No readable text, no logos, no road lettering.")
PERSON = "If the first image shows a person, paint an adult man in dark clothes seen from a distance with his face in shadow. "


def data_url(path):
    mime = "image/png" if Path(path).suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def paint(key, event_path, style_path, out_path, prompt=PROMPT, size="1536x1152", before_attempt=None, on_attempt=None, timeout=280, attempts=5):
    body = {"model": "muse-image-1.0", "prompt": prompt, "images": [{"image_url": data_url(event_path)}, {"image_url": data_url(style_path)}], "size": size, "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}}
    t = time.monotonic()
    last = "no attempt"
    for attempt in range(attempts):
        spend.check()
        if before_attempt:
            refused = before_attempt()
            if refused:
                return refused
        spend.record("muse-image-1.0", 1, note=Path(out_path).parent.parent.name + "/" + Path(out_path).name)
        started = time.monotonic()
        status = "request"
        try:
            r = httpx.post("https://api.meta.ai/v1/images/edits", headers={"Authorization": "Bearer " + key}, json=body, timeout=timeout)
        except httpx.RequestError as e:
            status = last = type(e).__name__
        else:
            status = r.status_code
            if status == 200:
                try:
                    Path(out_path).write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
                    return round(time.monotonic() - t, 1)
                except (KeyError, IndexError, TypeError, ValueError):
                    return "HTTP 200 without an image"
                finally:
                    if on_attempt:
                        on_attempt(attempt + 1, status, round(time.monotonic() - started, 3))
            last = f"HTTP {status}"
        if on_attempt:
            on_attempt(attempt + 1, status, round(time.monotonic() - started, 3))
        if isinstance(status, int):
            if re.search(r"content_policy|content management|content.?filter", r.text, re.I):
                return last + " content filter refusal"
            if re.search(r"credit|quota|insufficient|unauthori[sz]ed|authentication|invalid.?api.?key", r.text, re.I):
                return last + " permanent refusal"
            if status < 500 and status not in (408, 409, 425, 429):
                return last
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moment")
    ap.add_argument("--style", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="paint")
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    key = load_key()
    root = Path(a.moment)
    out = root / a.out
    out.mkdir(exist_ok=True)
    todo = [p for p in sorted((root / "ev10").glob("*.png"))[:: a.step] if not (out / (p.stem + ".webp")).exists()]
    with ThreadPoolExecutor(a.workers) as pool:
        done = list(pool.map(lambda p: (p.stem, paint(key, p, a.style, out / (p.stem + ".webp"))), todo))
    bad = [d for d in done if not isinstance(d[1], float)]
    good = [d[1] for d in done if isinstance(d[1], float)]
    print("painted", len(good), "failed", bad, "median seconds", sorted(good)[len(good) // 2] if good else None)


if __name__ == "__main__":
    main()
