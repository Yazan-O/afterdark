import re

import numpy as np

from .moment import H, RASTER, W

MIN_SCORE = 0.5
MATCH_IOU = 0.15
MATCH_CONTAIN = 0.5
MATCH_SIZE = 0.35
SUPPORT_RATIO = 1.5
RESTORED_IOU = 0.5
PATCH_MIN_ACUITY = 0.2
PATCH_MIN_RATIO = 0.5
PATCH_MAX_FILL = 0.92
PATCH_MAX_EDGE = 2.5
PATCH_MAX_LIGHT = 60.0
PATCH_MAX_GREEN = 0.02


def raster_span(box):
    sx, sy = RASTER[1] / W, RASTER[0] / H
    x0, y0, x1, y1 = box
    c0, r0 = int(x0 * sx), int(y0 * sy)
    return r0, min(RASTER[0], max(r0 + 1, int(np.ceil(y1 * sy)))), c0, min(RASTER[1], max(c0 + 1, int(np.ceil(x1 * sx))))


def decode_mask(mask, box):
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    marker = "~" if mask["encoding"] == "lossless" else "!"
    raw = decode_mask_to_raster(SegmentationMask(mask["encoding"], marker + mask["data"], mask["width"], mask["height"]))
    crop = np.frombuffer(raw, np.uint8).reshape(mask["height"], mask["width"]).astype(bool)
    r0, r1, c0, c1 = raster_span(box)
    rows = np.linspace(0, crop.shape[0] - 1, r1 - r0).round().astype(int)
    cols = np.linspace(0, crop.shape[1] - 1, c1 - c0).round().astype(int)
    grid = np.zeros(RASTER, bool)
    grid[r0:r1, c0:c1] = crop[np.ix_(rows, cols)]
    return grid


def mask_in(record):
    grid = np.zeros((record["height"], record["width"]), bool)
    x0, y0, x1, y1 = [int(v) for v in record["box"]]
    x1, y1 = min(record["width"] - 1, x1), min(record["height"] - 1, y1)
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
        m = record["mask"]
        marker = "~" if m["encoding"] == "lossless" else "!"
        raw = decode_mask_to_raster(SegmentationMask(m["encoding"], marker + m["data"], m["width"], m["height"]))
        crop = np.frombuffer(raw, np.uint8).reshape(m["height"], m["width"]).astype(bool)
        rows = np.linspace(0, crop.shape[0] - 1, y1 - y0 + 1).round().astype(int)
        cols = np.linspace(0, crop.shape[1] - 1, x1 - x0 + 1).round().astype(int)
        grid[y0:y1 + 1, x0:x1 + 1] = crop[np.ix_(rows, cols)]
    except Exception:
        grid[y0:y1 + 1, x0:x1 + 1] = True
    return grid


def box_mask(box):
    grid = np.zeros(RASTER, bool)
    r0, r1, c0, c1 = raster_span(box)
    grid[r0:r1, c0:c1] = True
    return grid


def from_record(record):
    sx, sy = W / record["width"], H / record["height"]
    x0, y0, x1, y1 = record["box"]
    box = [round(x0 * sx), round(y0 * sy), round((x1 + 1) * sx), round((y1 + 1) * sy)]
    try:
        mask = decode_mask(record["mask"], box)
    except Exception:
        mask = box_mask(box)
    return {"box": box, "score": round(record["score"], 3), "mask": mask}


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def size_ratio(a, b):
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return min(area_a, area_b) / max(area_a, area_b)


def mask_overlap(a, b):
    inter = int((a & b).sum())
    if not inter:
        return 0.0, 0.0
    return inter / int((a | b).sum()), inter / min(int(a.sum()), int(b.sum()))


def match(painted, camera, iou=MATCH_IOU, contain=MATCH_CONTAIN):
    pairs = []
    for i, p in enumerate(painted):
        for j, c in enumerate(camera):
            m_iou, m_contain = mask_overlap(p["mask"], c["mask"])
            b_iou = box_iou(p["box"], c["box"])
            if m_iou >= iou or b_iou >= iou or (m_contain >= contain and size_ratio(p["box"], c["box"]) >= MATCH_SIZE):
                pairs.append((max(m_iou, b_iou), i, j))
    pairs.sort(reverse=True)
    used_p, used_c, both = set(), set(), []
    for score, i, j in pairs:
        if i in used_p or j in used_c:
            continue
        used_p.add(i)
        used_c.add(j)
        both.append((i, j, round(float(score), 3)))
    return both, [i for i in range(len(painted)) if i not in used_p], [j for j in range(len(camera)) if j not in used_c]


def block_median(event_grid, size=16):
    rows, cols = event_grid.shape[0] // size, event_grid.shape[1] // size
    blocks = event_grid[:rows * size, :cols * size].reshape(rows, size, cols, size).sum((1, 3)) / (size * size)
    return float(np.median(blocks))


def support(box, event_grid, backing, base=None):
    x0, y0, x1, y1 = [int(v) for v in box]
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    inside = float(event_grid[y0:y1, x0:x1].sum())
    density = inside / ((x1 - x0) * (y1 - y0))
    base = block_median(event_grid) if base is None else base
    ratio = density / max(base, 0.05)
    return {
        "events": int(round(inside)),
        "per_pixel": round(density, 2),
        "vs_median": round(ratio, 2),
        "backing": round(float(backing[y0:y1, x0:x1].mean()), 1),
        "backed": ratio >= SUPPORT_RATIO,
    }


def public(obj, extra=None):
    out = {"box": obj["box"], "score": obj["score"]}
    if "support" in obj:
        out["support"] = obj["support"]
    if extra:
        out.update(extra)
    return out


def tally(painted, camera, event_grid, backing, min_score=MIN_SCORE):
    painted = [o for o in painted if o["score"] >= min_score]
    camera = [o for o in camera if o["score"] >= min_score]
    base = block_median(event_grid)
    for o in painted + camera:
        o["support"] = support(o["box"], event_grid, backing, base)
    both, only_p, only_c = match(painted, camera)
    return {
        "in_both": [{"painted": public(painted[i]), "camera": public(camera[j]), "overlap": s} for i, j, s in both],
        "painted_no_camera": [public(painted[i]) for i in only_p],
        "camera_no_painted": [public(camera[j]) for j in only_c],
        "counts": {
            "painted": len(painted), "camera": len(camera), "in_both": len(both),
            "painted_no_camera": len(only_p), "camera_no_painted": len(only_c),
            "left_out_with_events": sum(1 for j in only_c if camera[j]["support"]["backed"]),
            "invented_no_events": sum(1 for i in only_p if not painted[i]["support"]["backed"]),
        },
    }


def luma(rgb):
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def blurred(gray, sigma):
    reach = int(3 * sigma + .5)
    kernel = np.exp(-.5 * (np.arange(-reach, reach + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    pad = np.pad(np.asarray(gray, np.float32), reach, mode="edge")
    wide = np.apply_along_axis(lambda row: np.convolve(row, kernel, "valid"), 1, pad)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, "valid"), 0, wide)


def acuity(gray, where):
    if where.sum() < 8:
        return 0.0
    detail = gray - blurred(gray, 1.5)
    return float(detail[where].std() / (gray[where].std() + 2.0))


def patch_check(before, painted, soft, hard, concept):
    gray, base = luma(np.asarray(painted, np.float32)), luma(np.asarray(before, np.float32))
    ring = (blurred(soft, 8) > .02) & ~(soft > .02)
    ys, xs = np.nonzero(hard)
    fill = float(hard.sum() / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))) if hard.any() else 0.0
    edge = blurred(hard.astype(np.float32), 1)
    edge = (edge > .15) & (edge < .85)
    grad = np.hypot(*np.gradient(gray))
    grad_base = np.hypot(*np.gradient(base))
    r, g, b = [np.asarray(painted, np.float32)[..., i] for i in range(3)]
    a_in, a_ring = acuity(gray, hard), acuity(base, ring)
    out = {
        "acuity": round(a_in, 3), "acuity_ring": round(a_ring, 3), "acuity_ratio": round(a_in / max(a_ring, .01), 2),
        "edge": round(float(grad[edge].mean()) if edge.any() else 0.0, 2), "edge_ring": round(float(np.percentile(grad_base[ring], 90)) if ring.any() else 0.0, 2),
        "fill": round(fill, 3), "green": round(float(((g > 1.5 * r) & (g > 1.5 * b) & (g > 90))[hard].mean()) if hard.any() else 0.0, 3),
        "light_in": round(float(gray[hard].mean()) if hard.any() else 0.0, 1), "light_ring": round(float(base[ring].mean()) if ring.any() else 0.0, 1),
    }
    out["edge_ratio"] = round(out["edge"] / max(out["edge_ring"], 2.0), 2)
    why = []
    if out["acuity"] < PATCH_MIN_ACUITY or out["acuity_ratio"] < PATCH_MIN_RATIO:
        why.append("blurred")
    if out["fill"] > PATCH_MAX_FILL or out["edge_ratio"] > PATCH_MAX_EDGE or out["light_in"] > out["light_ring"] + PATCH_MAX_LIGHT:
        why.append("pasted on")
    if out["green"] > PATCH_MAX_GREEN:
        why.append("box mark left in")
    out.update(ok=not why, why=", ".join(why))
    return out


def restored(before, after, within=None):
    near = (lambda b: box_iou(b, within) > 0) if within else (lambda b: True)
    now = [o["box"] for o in after["camera_no_painted"]]
    back = [o for o in before["camera_no_painted"] if near(o["box"]) and not any(box_iou(o["box"], n) >= RESTORED_IOU for n in now)]
    invented = after["counts"]["painted_no_camera"] - before["counts"]["painted_no_camera"]
    if within:
        had = [o["box"] for o in before["painted_no_camera"]]
        invented = sum(1 for o in after["painted_no_camera"] if near(o["box"]) and not any(box_iou(o["box"], h) >= RESTORED_IOU for h in had))
    was = [o["box"] for o in before["camera_no_painted"]]
    lost = [b for b in now if near(b) and not any(box_iou(b, w) >= RESTORED_IOU for w in was)]
    return {
        "restored": [o["box"] for o in back],
        "restored_with_events": sum(1 for o in back if o.get("support", {}).get("backed")),
        "left_out_change": after["counts"]["camera_no_painted"] - before["counts"]["camera_no_painted"],
        "invented_change": invented,
        "lost": lost,
        "helped": bool(back) and invented <= 0 and not lost,
    }


NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "hundred": 100, "thousand": 1000}
NAMES = re.compile(r"(?i)\b(sam[ -]?3\.1|muse[ -]spark[ -]1\.3|muse[ -]image[ -]1\.0|ev10)\b")


def numbers_in(value):
    found = set()
    if isinstance(value, dict):
        for v in value.values():
            found |= numbers_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            found |= numbers_in(v)
    elif isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        found.add(float(value))
    elif isinstance(value, str):
        found |= {float(t.replace(",", "")) for t in NUMBER.findall(value)}
    return found


def known(text, pool):
    value = float(text.replace(",", ""))
    if value in pool:
        return True
    places = len(text.split(".")[1]) if "." in text else 0
    return any(round(p, places) == value for p in pool)


def audit_text(text, pool):
    problems = []
    clean = NAMES.sub("", text)
    for sentence in re.split(r"(?<=[.!?])\s+", clean.strip()):
        if not sentence.strip():
            continue
        tokens = NUMBER.findall(sentence)
        lowered = re.findall(r"[a-z]+", sentence.lower())
        spelled = [w for w in lowered if w in WORDS]
        if not tokens:
            problems.append(f"no tool number in: {sentence[:70]}")
        for t in tokens:
            if not known(t, pool):
                problems.append(f"{t} is not in any tool result")
        for w in spelled:
            problems.append(f"write '{w}' as the digit a tool returned")
    return problems
