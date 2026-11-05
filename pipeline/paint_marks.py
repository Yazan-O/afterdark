import argparse
import json
from pathlib import Path

from sam_common import sweep

PHRASES = {"plate": "license plate", "badge": "car logo", "text": "text"}
SIGN = {"sign": "sign board"}


def run(root, phrases):
    root = Path(root)
    n = json.loads((root / "meta.json").read_text())["n_paint"]
    paths = {k: root / "paint" / f"{k:04d}.webp" for k in range(n) if (root / "paint" / f"{k:04d}.webp").exists()}
    report = {}
    for tag, phrase in phrases.items():
        found = sweep(paths, phrase, root / "sam" / f"{tag}_paint")
        report[tag] = {"frames": sum(1 for v in found.values() if v), "regions": sum(len(v) for v in found.values())}
    print(root.name, "paintings", len(paths), report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moments", nargs="+")
    ap.add_argument("--signs", action="store_true")
    a = ap.parse_args()
    for m in a.moments:
        run(m, {**PHRASES, **SIGN} if a.signs else PHRASES)


if __name__ == "__main__":
    main()
