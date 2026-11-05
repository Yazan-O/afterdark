import argparse
import json
from pathlib import Path
import shutil

import cv2

ORDER = ["crosswalk", "street", "zone30", "rider"]
TITLES = {"crosswalk": "Crosswalk", "street": "Street", "zone30": "Zone 30", "rider": "Rider"}
WEB = Path(__file__).resolve().parents[1] / "web" / "moments"
CREDIT = "DSEC dataset, CC BY-SA 4.0"


def clear(folder):
    if folder.exists():
        for f in folder.glob("*.webp"):
            f.unlink()
    folder.mkdir(parents=True, exist_ok=True)


def pack(root):
    root = Path(root)
    mid = root.name
    meta = json.loads((root / "meta.json").read_text())
    n = meta["n_paint"]
    out = WEB / mid
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("paint", "cam", "backing"):
        clear(out / sub)
    for k in range(n):
        name = f"{k:04d}.webp"
        shutil.copyfile(root / "stab" / name, out / "paint" / name)
        shutil.copyfile(root / "cam_pub" / name, out / "cam" / name)
        gray = cv2.imread(str(root / "backing" / f"{k:04d}.png"), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(out / "backing" / name), small, [cv2.IMWRITE_WEBP_QUALITY, 70]):
            raise SystemExit(f"could not write backing {name}")
    shutil.copyfile(root / "events.bin", out / "events.bin")
    shutil.copyfile(root / "people.json", out / "people.json")
    if (root / "drift.json").exists():
        shutil.copyfile(root / "drift.json", out / "drift.json")
    filled = json.loads((root / "stab_report.json").read_text())["filled"]
    meta.update({"n_cam": n, "n_paint_pub": n, "place": "Zurich", "title": TITLES[mid], "credit": CREDIT, "filled_paint": filled})
    (out / "meta.json").write_text(json.dumps(meta))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(mid, "frames", n, "total MB", round(size / 1e6, 1))
    for sub in ("paint", "cam", "backing"):
        print(" ", sub, round(sum(f.stat().st_size for f in (out / sub).glob("*")) / 1e6, 1), "MB")


def write_index():
    present = [m for m in ORDER if (WEB / m / "meta.json").exists()]
    (WEB / "index.json").write_text(json.dumps({"moments": present}))
    print("index", present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moments", nargs="+")
    a = ap.parse_args()
    for m in a.moments:
        pack(m)
    write_index()


if __name__ == "__main__":
    main()
