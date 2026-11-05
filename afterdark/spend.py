import csv
import datetime
import os
from pathlib import Path
import threading
import time

STOP_USD = 75.0
PRICES = {
    "muse-image-1.0": {"unit": 0.01},
    "sam-3.1": {"unit": 0.0025},
    "muse-spark-1.3": {"input": 1.25e-6, "cached": 0.15e-6, "output": 4.25e-6},
    "muse-spark-1.2": {"input": 1.25e-6, "cached": 0.15e-6, "output": 4.25e-6},
    "muse-wonder-1.0": {"input": 4e-6, "output": 20e-6},
}
_lock = threading.Lock()


class SpendStop(RuntimeError):
    pass


def ledger():
    path = os.environ.get("AFTERDARK_SPEND")
    if path:
        return Path(path)
    build = Path(__file__).resolve().parents[2] / "build"
    return build / "SPEND.csv" if build.is_dir() else None


def total():
    path = ledger()
    if path is None or not path.exists():
        return 0.0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(float(r["usd"]) for r in csv.DictReader(f))


def check():
    if total() >= STOP_USD:
        raise SpendStop(f"Paid calls stopped: the ledger reached ${STOP_USD:.0f}")


def price(model, count=1, usage=None):
    p = PRICES.get(model, PRICES["muse-wonder-1.0"])
    if "unit" in p:
        return count * p["unit"]
    usage = usage or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    fresh = (usage.get("prompt_tokens") or 0) - cached
    return fresh * p["input"] + cached * p.get("cached", p["input"]) + (usage.get("completion_tokens") or 0) * p["output"]


def record(model, count=1, usage=None, note="", usd=None):
    path = ledger()
    if path is None:
        return
    cost = price(model, count, usage) if usd is None else usd
    row = [datetime.datetime.now().astimezone().isoformat(timespec="seconds"), model, count, f"{cost:.6f}", note]
    with _lock:
        for attempt in range(20):
            try:
                new = not path.exists()
                with path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["time", "model", "count", "usd", "note"])
                    w.writerow(row)
                return
            except PermissionError:
                time.sleep(0.1)
    raise SpendStop("Could not write the spend ledger")
