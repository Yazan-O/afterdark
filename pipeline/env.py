import os
from pathlib import Path


def load_key():
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"].strip()
    path = os.environ.get("AFTERDARK_ENV")
    if not path:
        raise SystemExit("Set MODEL_API_KEY or AFTERDARK_ENV to an environment-file path")
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("MUSE_CODE_API") and "=" in line:
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("MODEL_API_KEY is not set")
