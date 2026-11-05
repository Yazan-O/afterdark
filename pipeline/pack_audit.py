import argparse
import json
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web" / "moments"
MIN_AREA = 400
BEATS_ROW = 0.9
FIELDS = ("camera", "painted", "in_both", "left_out", "invented")


def activity(o):
    return o["area"] > MIN_AREA and o["support"]["backed"] and o["band_pct"] >= BEATS_ROW


def add(totals, concept, counts):
    row = totals.setdefault(concept, dict.fromkeys(FIELDS, 0))
    for f in FIELDS:
        row[f] += counts[f]


def pack(doc):
    frames, shipped, raw = [], {}, {}
    for k in range(doc["n_frames"]):
        out = {"camera": {}, "painted": {}, "invented": 0, "left_out": []}
        for concept, cell in doc["frames"][str(k)]["concepts"].items():
            if cell["status"] != "ok":
                raise ValueError(f"{doc['moment']} k={k} {concept}: {cell['status']}")
            c = cell["counts"]
            if c["left_out"] != len(cell["left_out"]):
                raise ValueError(f"{doc['moment']} k={k} {concept}: left_out count differs from its boxes")
            if c["camera"]:
                out["camera"][concept] = c["camera"]
            if c["painted"]:
                out["painted"][concept] = c["painted"]
            out["invented"] += c["invented"]
            out["left_out"] += [{"box": o["box"], "who": concept, "activity": bool(activity(o))} for o in cell["left_out"]]
            add(shipped, concept, c)
            add(raw, concept, cell["raw_muse_counts"])
        frames.append(out)
    return {"moment": doc["moment"], "n_frames": doc["n_frames"], "painting": "shipped smoothed", "totals": shipped, "raw_totals": raw, "frames": frames}


def check(packed, summary):
    for concept, row in summary["by_concept"].items():
        for label, totals in (("shipped", packed["totals"]), ("raw", packed["raw_totals"])):
            for f in FIELDS:
                if totals[concept][f] != row[label][f]:
                    raise ValueError(f"{packed['moment']} {concept} {label} {f}: {totals[concept][f]} against summary {row[label][f]}")
    flagged = sum(o["activity"] for fr in packed["frames"] for o in fr["left_out"])
    want = sum(row["shipped"]["over_400_beats_90pct_of_same_row_boxes"] for row in summary["by_concept"].values())
    if flagged != want:
        raise ValueError(f"{packed['moment']} activity boxes: {flagged} against summary {want}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_dir")
    ap.add_argument("--web", default=str(WEB))
    a = ap.parse_args()
    root = Path(a.audit_dir)
    summary = json.loads((root / "summary.json").read_text())
    for moment in summary:
        packed = pack(json.loads((root / f"audit_{moment}.json").read_text()))
        check(packed, summary[moment])
        out = Path(a.web) / moment / "audit.json"
        out.write_text(json.dumps(packed, separators=(",", ":")))
        print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
