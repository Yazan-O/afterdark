import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import spend
from .meta import MetaAPIError, sam_segment, spark_ask
from .moment import H, Moment, MomentError, VIEWS, W, activity_grid, crop_box, png_bytes
from .verdict import MIN_SCORE, audit_text, block_median, box_iou, from_record, mask_in, numbers_in, patch_check, public, restored, support, tally

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
WEB = Path(__file__).resolve().parents[1] / "web"
MAX_STEPS = 14
GRACE_STEPS = 2
MAX_REPAINTS = 3
KEEP_IMAGES = 4
WIDE_MS = (300, 1000)
MAX_REASON_REFUSALS = 3
INSPECT_STEPS = 8
INSPECT_REPAINTS = 2
INSPECT_REPAINT_BY = 95
LOOKED_IOU = 0.3
FIELD_NAME = re.compile(r"\b[a-z]+_[a-z_]+\b|\bk\s*=\s*\d")
SENSOR_WORD = re.compile(r"(?i)\b(camera|events?)\b")
BANNED_NOTE = re.compile(r"(?i)\b(wom[ae]n|lady|ladies|girls?|boys?|child(ren)?|kids?|bab(y|ies)|logos?|brand\w*|text|letter\w*|words?)\b")
SPARK_URL = "https://api.meta.ai/v1"
KINDS = ("person", "car", "tram", "cyclist")
PEOPLE = ("person", "pedestrian", "man")
ALSO_ASK = {"tram": "train"}
SPREAD = (0.125, 0.375, 0.625, 0.875)
CROP_ZOOM = 1.8
CROP_MIN = 64
CROP_PX = 1024
CROP_WINDOW_MS = 300
PATCH_MATCH = 0.15
PATCH_GROW = 0.04
FACING_MIN_PERSON = 60


class AgentError(RuntimeError):
    pass


class ToolError(ValueError):
    pass


def pipeline():
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import paint_moment
    from env import load_key
    return paint_moment, load_key


def ensure_key():
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"]
    _, load_key = pipeline()
    os.environ["MODEL_API_KEY"] = load_key()
    return os.environ["MODEL_API_KEY"]


def _reason():
    return {"type": "string", "description": "One short sentence in your own words: which sensor or view you chose (events or camera) and why."}


def _tool(name, description, props, required):
    props = {**props, "reason": _reason()}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": [*required, "reason"]}}}


_K = {"type": "integer", "description": "Frame index: one per 100 ms."}
_VIEW = {"type": "string", "enum": list(VIEWS)}
_WINDOW = {"type": "integer", "description": "Event time window in ms, centred on k. 10 is what the painter saw; wider gathers more."}
_CONCEPT = {"type": "string", "description": "One noun phrase, e.g. person, car, tram, cyclist."}

TOOLS = [
    _tool("scan", "Summary of the whole drive: event activity per second on a coarse grid, camera brightness, which frames have paintings, where people masks are. Returns one contact sheet of per-second event maps and one sheet pairing the camera (left) with the painting (right) for each second.", {}, []),
    _tool("look", "See one view yourself: events over a window you choose, smoothed event map (backing), the camera (brightened, it is dark as recorded), or the painting. Optional crop at x, y, r in 640x480 coordinates. Also returns the real event count inside the crop. An events look at 300 to 1000 ms also reports how strongly the events hold each object left out at this frame.", {"k": _K, "view": _VIEW, "window_ms": _WINDOW, "x": {"type": "integer"}, "y": {"type": "integer"}, "r": {"type": "integer"}}, ["k", "view"]),
    _tool("find", "Run SAM 3.1 for one concept on one view, with the real event count inside each box over the window.", {"k": _K, "view": _VIEW, "concept": _CONCEPT, "window_ms": _WINDOW}, ["k", "view", "concept"]),
    _tool("compare", "Arithmetic check of the painting against the sensors for one concept: SAM on the painting and on the camera, matched by overlap, with event support for every box. Returns in_both, painted_no_camera, camera_no_painted, counts. This is the verdict; you do not give verdicts.", {"k": _K, "concept": _CONCEPT}, ["k", "concept"]),
    _tool("repaint", "Camera-guided repair: Muse Image paints the left-out object into a zoomed crop of the painting marked with the camera's box, only that patch goes back into the frame, and compare re-runs to measure whether it helped. Needs a compare on k that left something out, then an events look at 300 to 1000 ms covering it.", {"k": _K, "note": {"type": "string", "description": "Under 200 characters: how the object stands and faces, read from the camera (toward the viewer, away from the viewer, or side-on), e.g. a car in the lane, its rear toward the viewer."}, "concept": {"type": "string", "description": "Which compared concept the painter should restore. Default: every compared concept."}}, ["k", "note"]),
    _tool("report", "End the run. Write plain sentences for a street team. Every sentence of finding and invented must quote numbers copied from tool results; a sentence with none, with another number, or with a tool field name is refused. The whole-drive list is added for you.", {"finding": {"type": "string", "description": "What was caught across the drive, and which sensor backs each item."}, "view": {"type": "string", "enum": ["events", "camera", "events+camera"]}, "invented": {"type": "string", "description": "What the painting invented, or left out, with numbers."}}, ["finding", "view", "invented"]),
]

INSPECT_TOOLS = [
    _tool("compare", "Arithmetic check of the held painting against the sensors: SAM on the painting and on the camera, matched by overlap, with event activity inside every box. Returns by_concept with counts, camera_no_painted and painted_no_camera. This is the verdict; you do not give verdicts.", {"concepts": {"type": "array", "items": {"type": "string"}, "description": "Noun phrases to check, e.g. person, car, tram, cyclist."}}, ["concepts"]),
    TOOLS[1],
    TOOLS[4],
    _tool("report", "End the inspection when more checks would not change the result. Ending with no repaint is a good outcome when the events do not hold the object.", {}, []),
]

INSPECT_SYSTEM = """You inspect one frame of a night drive that a viewer is holding. Muse Image painted it from 10 milliseconds of an event camera alone, so the painting is a guess. You hold the recorded camera, dark but real, and the event stream, which you can re-accumulate over any window. Work on the held frame only. You have 8 steps, and the viewer is waiting, so each turn should move straight to the next decision.

1. Call compare once, with person, car, tram and cyclist.
2. If compare lists objects under camera_no_painted, it also names aim_first: a left-out person whenever there is one, because people come first, and otherwise the largest object. In one turn call look twice on it: view events at window_ms 800, and view camera. Use the x, y and r it gives.
3. The repair is camera-guided: the camera's box says where, so left_out_here is shown to the viewer but does not decide. A person in the dark may be only a few dim pixels; count it as shown unless the crop plainly shows something else. If the camera look shows that object, call repaint with that concept and a note under 15 words saying how it stands and faces, read from the camera. For a person, say only which way the body walks (left, right, toward or away from the viewer), never the face, clothes or who it is. If the camera look shows no such object, call report.
4. If the repaint did not help and steps remain, try once more on the same object with a clearer note. Otherwise call report.

Arithmetic decides whether anything came back. Ending with no repaint is right only when the camera does not show the object. Never ask for a woman, a child, readable text or a logo."""

SYSTEM = """You inspect a night drive. Muse Image painted each frame from 10 milliseconds of an event camera and nothing else, so the painting is a guess with a sensor underneath it. You hold two real sensors: the event stream, which you can re-accumulate over any window, and the recorded camera, which is dark but real. A team will act on your list, so it must say who was on the street, when, from which sensor, and what the painting got wrong.

You choose the checks and you have 14 steps, so use as few as the job allows. You never decide what is there: SAM finds objects and arithmetic counts them. Trust compare over your own eyes, and trust your own eyes only to decide where to spend the next check.

How to work:
- Every reason names the sensor you chose, the camera or the events, and why that sensor can settle this question. The camera shows what was in front of the car when light allowed. The events show what changed; a 10 ms window shows little, a 300 to 1000 ms window turns a moving object into a sharp outline while the static world smears. Use both sensors.
- Make one call at a time whenever the next call depends on the result. Calls sent together when they depend on each other are not run.
- Each turn tells you how many steps you have used. Keep 4 steps for one repair: the wide look, the camera look, the repaint, and the report.
- Spend checks where the drive changes, not on every frame. Jump across the drive; do not walk frame by frame. Before the first repair, compare people at 4 or more moments spread from the start to the end of the drive, so the list covers the whole drive.
- Compare people and also whatever else you see in the camera that could matter on a street: vehicles, trams, cyclists.
- When compare lists an object under camera_no_painted, that omission is confirmed. Pick a left-out person first, otherwise the largest one nearest the car, and do these in order: look at its box on the events at 300 to 1000 ms; if the events hold it, look at its box on the camera and read how the object faces; then order a repaint with a short note saying what stands where and how it faces, as the camera shows it. The repaint re-runs compare and the arithmetic says whether it came back. Do not report before at least one confirmed omission has been acted on, unless compare finds none.
- Never ask for a woman, a child, readable text or a logo in a note.
- Finish with report when more checks would not change the list.

How to write the report:
- Plain sentences a colleague would say aloud, for example: "At [T] seconds the camera holds [N] cars and the painting shows none of them." Say seconds, not frame numbers. Never use tool field names.
- Every sentence quotes numbers from tool results, and no other numbers. Write counts as digits.
- An event count belongs to one box: give it for that one object, never for a group. Say an object came back only where the repaint result says helped or restored.
- Say no more than the numbers support: never that this prevents collisions, never that one sensor is better than the other."""


CROP_PROMPT = ("The first image is a close-up cut from a painting of a night street. The second image is an event-camera recording of the same close-up: bright grain marks where light changed. "
               "A {who} stands inside the thin green rectangle, and the painting left it out. Paint {paint} inside the green rectangle, filling it from top to bottom, {ground}, in the same night light and painted style. How it stands, read from the camera: {note} "
               "Remove the green rectangle. Change nothing outside it. Add no other person or object. No readable text, no logos, no lettering, no numbers.")


def crop_prompt(concept, note):
    person = concept in ("person", "pedestrian", "man")
    paint = "one adult man in dark clothes with his face in shadow" if person else f"one {concept} with no sign and no marks on it"
    return CROP_PROMPT.format(who=concept, paint=paint, ground="feet on the ground" if person else "on the road", note=note)


def crop_area(box):
    x0, y0, x1, y1 = box
    side = int(min(H, max(CROP_MIN, CROP_ZOOM * max(x1 - x0, y1 - y0))))
    a = int(min(max(0, (x0 + x1 - side) / 2), W - side))
    b = int(min(max(0, (y0 + y1 - side) / 2), H - side))
    return a, b, a + side, b + side


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def aim_rank(who, box):
    return (who not in PEOPLE, -box_area(box))


FACING_QUESTION = ("The first image is a crop of a recorded night camera, brightened. The second is the same spot in a painting. Each should show a {concept}. Judge each image on its own, from the {concept} only. Answer in JSON only with these keys: "
                   '{{"camera_view": "toward" or "away" or "side" or "unclear", "camera_front_at": "left" or "right" or "unclear", "painted_view": "toward" or "away" or "side" or "unclear", "painted_front_at": "left" or "right" or "unclear", "why": "one short sentence"}}. '
                   "view toward: its front end is the near end and the body recedes behind it. view away: its rear end is the near end. view side: its long side runs across the picture, roughly level. front_at: which side of the image its front end points to; unclear if it points at or away from the viewer.")


def facing_check(camera, painted, concept):
    last = None
    for attempt in range(2):
        try:
            text = spark_ask([camera, painted], FACING_QUESTION.format(concept=concept), max_completion_tokens=2000, reasoning_effort="low")
            answer = json.loads(re.search(r"\{.*\}", text, re.S)[0])
            break
        except spend.SpendStop:
            raise
        except (MetaAPIError, ValueError, TypeError) as e:
            last = e
    else:
        return {"bad": ["unchecked"], "why": str(last)[:120]}
    a, b = str(answer.get("camera_view")), str(answer.get("painted_view"))
    c, d = str(answer.get("camera_front_at")), str(answer.get("painted_front_at"))
    if concept in PEOPLE:
        a, b = ("along" if v in ("toward", "away") else v for v in (a, b))
    known = {"toward", "away", "side", "along"}
    wrong = (a in known and b in known and a != b) or {c, d} == {"left", "right"}
    return {"bad": ["facing"] if wrong else [], "answer": answer}


def rules_check(path, box):
    pipeline()
    import lookcheck
    image = Image.open(path).convert("RGB")
    sx, sy = image.width / W, image.height / H
    shown = Path(path).with_suffix(".shown.png")
    image.crop((round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy))).save(shown)
    with ThreadPoolExecutor(2) as pool:
        answers = list(pool.map(lookcheck.ask, (shown, path)))
    if any("error" in a for a in answers):
        return ["unchecked"]
    return [k for k in lookcheck.KEYS if any(a[k] for a in answers)]


class Route:
    def __init__(self, name, base_url, model, served_on, key=None, extra=None, guided=False):
        self.name, self.base_url, self.model, self.served_on, self.key, self.extra, self.guided = name, base_url.rstrip("/"), model, served_on, key, extra or {}, guided
        self.paid = "api.meta.ai" in self.base_url
        self.timeout = 280

    def chat(self, messages, tools):
        body = {"model": self.model, "messages": messages, "tools": tools, "max_tokens": 8000, **self.extra}
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        if self.paid:
            spend.check()
        last = "no attempt"
        for attempt in range(4):
            try:
                r = httpx.post(self.base_url + "/chat/completions", headers=headers, json=body, timeout=600 if not self.paid else self.timeout)
            except httpx.RequestError as e:
                last = type(e).__name__
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                    message = data["choices"][0]["message"]
                    message["finish_reason"] = data["choices"][0].get("finish_reason")
                    message["usage"] = data.get("usage")
                except (ValueError, KeyError, IndexError, TypeError):
                    raise AgentError("The model route returned an unreadable answer") from None
                if self.paid:
                    spend.record(self.model, 1, data.get("usage"), note="agent")
                return message
            last = f"HTTP {r.status_code}"
            if r.status_code not in (408, 429) and r.status_code < 500:
                break
            try:
                wait = min(float(r.headers.get("retry-after", 0)), 65.0)
            except ValueError:
                wait = 0.0
            time.sleep(max(2 ** attempt, wait))
        raise AgentError(f"The model route failed: {last}")


def spark_route():
    return Route("spark", SPARK_URL, "muse-spark-1.3", "Meta API", key=ensure_key())


def glimmer_route():
    base = os.environ.get("GLIMMER_URL", "http://127.0.0.1:8091/v1").rstrip("/")
    try:
        ids = httpx.get(base + "/models", timeout=20).json()["data"]
    except Exception:
        raise AgentError("Glimmer does not answer at GLIMMER_URL") from None
    extra = json.loads(os.environ.get("GLIMMER_EXTRA", "{}"))
    return Route("glimmer", base, os.environ.get("GLIMMER_MODEL", ids[0]["id"]), os.environ.get("GLIMMER_SERVED_ON", "amadeus H200 NVL, llama.cpp commit bee23322, weights sha256 prefix 4cc57c0f"), extra=extra, guided=True)


def get_route(name):
    if name == "spark":
        return spark_route()
    if name == "glimmer":
        return glimmer_route()
    raise AgentError("route must be spark or glimmer")


def data_url(image_bytes, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode()


def fit_for_model(image, longest=768):
    if max(image.size) > longest:
        s = longest / max(image.size)
        return image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)
    if max(image.size) < 384:
        s = 384 / max(image.size)
        return image.resize((round(image.width * s), round(image.height * s)), Image.NEAREST)
    return image


def _int(value, name, low=None, high=None, default=None):
    if value is None:
        if default is None:
            raise ToolError(f"{name} is required")
        return default
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a number") from None
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _concept(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 40:
        raise ToolError("concept must be one short noun phrase")
    if BANNED_NOTE.search(value):
        raise ToolError("concept may not name a woman, a child, text or a logo; say a person or a vehicle")
    return value.strip().lower()


class Tools:
    def __init__(self, moment, live=False, paint_guard=None, web=WEB, painter=None, segment=None, repaint_cap=None, rules=None, facing=None):
        self.m = moment
        self.live = live
        self.paint_guard = paint_guard
        self.web = Path(web)
        self.painter = painter
        self.segment = segment or sam_segment
        self.repaint_cap = repaint_cap if repaint_cap is not None else (1 if live else MAX_REPAINTS)
        self.tmp = Path(tempfile.mkdtemp(prefix="afterdark_"))
        self.sam = {}
        self.compared = {}
        self.repaints = []
        self.sightings = []
        self.invented = []
        self.frames = []
        self.latest = {}
        self.versions = {}
        self.calls = {"sam": 0, "paint": 0}
        self.pool = set()
        self.tried = set()
        self.camera_crops = []
        self.only_k = None
        self.started = time.time()
        self.refused = None
        self.rules = rules
        self.facing = facing
        self.withheld = []

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def touch(self, k):
        if k not in self.frames:
            self.frames.append(k)

    def base(self, k):
        k = self.m.check_k(k)
        if self.only_k is not None and k != self.only_k:
            raise ToolError(f"this run inspects k={self.only_k} only")
        return {"k": k, "t_s": round(self.m.t_ms(k) / 1000, 1)}

    def sam_on(self, k, view, concept, window_ms=100, version=None):
        if view == "painting" and version is None:
            version = self.latest.get(k)
        key = (k, view, concept, window_ms if view == "events" else 0, version)
        if key in self.sam:
            return self.sam[key]
        if view == "painting":
            path = self.versions.get((k, version)) or self.m.painting_path(k, version)
        else:
            path = self.tmp / f"{view}_{k}_{window_ms}.png"
            if not path.exists():
                self.m.view_image(k, view, window_ms).save(path)
        self.calls["sam"] += 1
        objects = [from_record(r) for r in self.segment(path, concept)]
        if concept in ALSO_ASK:
            for o in map(from_record, self.segment(path, ALSO_ASK[concept])):
                if not any(box_iou(o["box"], b["box"]) >= .5 for b in objects):
                    objects.append(o)
        self.sam[key] = objects
        return objects

    def current_painting(self, k):
        path = self.versions.get((k, self.latest.get(k)))
        return Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS) if path else self.m.painting(k)

    def scan(self):
        m = self.m
        seconds = []
        for s in range(max(1, int(m.seconds))):
            k = min(s * m.paint_fps + m.paint_fps // 2, m.n_paint - 1)
            a, b = s * m.bin_fps, min(len(m.counts), (s + 1) * m.bin_fps)
            events = int(m.counts[a:b].sum())
            row = {
                "second": s, "k": k, "events": events,
                "on_share": round(float(m.on[a:b].sum()) / max(1, events), 2),
                "camera_mean": round(float(m.camera_raw(k).mean()), 1),
                "grid": activity_grid(m.backing(k)),
            }
            people = [p for f, ps in m.people.items() if s * m.paint_fps <= f < (s + 1) * m.paint_fps for p in ps]
            if m.people:
                row["people_boxes"] = len(people)
            seconds.append(row)
        result = {
            "seconds": m.seconds, "n_frames": m.n_paint, "painted": m.painted(),
            "grid_cols": 8, "grid_rows": 6, "per_second": seconds,
        }
        if m.people:
            first = {f: [p["box"] for p in ps] for f, ps in sorted(m.people.items())}
            result["people_frames"] = len(first)
            result["people_first"] = dict(list(first.items())[:8])
        return result, [("Per-second event maps across the drive.", fit_for_model(m.contact_sheet())), ("Each second: camera on the left, painting on the right.", fit_for_model(m.pair_sheet(), 1280))]

    def look(self, k, view, window_ms=None, x=None, y=None, r=None):
        result = self.base(k)
        k = result["k"]
        if view not in VIEWS:
            raise ToolError(f"view must be one of {', '.join(VIEWS)}")
        window_ms = _int(window_ms, "window_ms", 1, 3000, 100)
        image = self.current_painting(k) if view == "painting" else self.m.view_image(k, view, window_ms)
        box = (0, 0, W, H)
        if x is not None and y is not None:
            box = crop_box(_int(x, "x"), _int(y, "y"), _int(r, "r", 8, 320, 80))
            result.update(x=int(x), y=int(y), r=_int(r, "r", 8, 320, 80))
            image = image.crop(box)
            if view == "camera":
                self.camera_crops.append((k, box))
        grid = self.m.event_map(k, window_ms)
        sup = support(box, grid, self.m.backing(k))
        result.update(view=view, window_ms=window_ms, events=sup["events"], per_pixel=sup["per_pixel"], vs_median=sup["vs_median"])
        if view == "camera":
            raw = self.m.camera_raw(k)[box[1]:box[3], box[0]:box[2]]
            result["camera_raw_mean"] = round(float(raw.mean()), 1)
        if view == "events" and WIDE_MS[0] <= window_ms <= WIDE_MS[1]:
            held = self._wide_look(k, window_ms, box, grid)
            if held:
                result["left_out_here"] = held
        self.touch(k)
        return result, [(f"{view} at k={k}", fit_for_model(image))]

    def _wide_look(self, k, window_ms, box, grid):
        held = []
        backing = self.m.backing(k)
        for s in self.sightings:
            if s["k"] != k or s["painted"] is not False or s.get("restored_by") or box_iou(box, s["box"]) <= 0:
                continue
            sup = support(s["box"], grid, backing)
            s.update(wide_ms=window_ms, wide_events=sup["events"], wide_vs_median=sup["vs_median"], wide_held=sup["backed"])
            if sup["backed"] and "events" not in s["from"]:
                s["from"].append("events")
            held.append({"who": s["who"], "box": s["box"], "events": sup["events"], "vs_median": sup["vs_median"], "held": sup["backed"]})
        return held

    def open_items(self):
        left = [s for s in self.sightings if s["painted"] is False and not s.get("restored_by")]
        left.sort(key=lambda s: aim_rank(s["who"], s["box"]))
        lines = []
        for s in left[:3]:
            step = "repaint it" if s.get("wide_held") else "the events did not hold it" if s.get("wide_ms") else "look at its box on the events at 300 to 1000 ms"
            lines.append(f"k={s['k']} {s['who']} at {s['box']}: {step}")
        return lines

    def repair_step(self, s):
        x0, y0, x1, y1 = s["box"]
        spot = f"x={(x0 + x1) // 2}, y={(y0 + y1) // 2}, r={max(x1 - x0, y1 - y0) // 2 + 12}"
        name = f"the {s['who']} at {s['box']} on k={s['k']}"
        if not s.get("wide_ms"):
            return f"look at {name} on the events, window_ms 800, {spot}."
        if not any(k == s["k"] and box_iou(box, s["box"]) > 0 for k, box in self.camera_crops):
            return f"look at {name} on the camera, {spot}, and read how it faces."
        return f"repaint {name}, concept {s['who']}, with a note on what stands where and how it faces."

    def plan(self, held, left):
        anchor = held if held is not None else self.m.n_paint // 2
        missing = [c for c in KINDS if (anchor, c) not in self.compared]
        if missing:
            return f"compare {missing[0]} at k={anchor}. Still to compare there: {', '.join(missing)}. No repair until these are done."
        items = [s for s in self.sightings if s["painted"] is False and not s.get("restored_by") and (s["k"], s["who"]) not in self.tried]
        items.sort(key=lambda s: aim_rank(s["who"], s["box"]))
        here = [s for s in items if s["k"] == anchor]
        room = left >= 4 and len(self.repaints) < self.repaint_cap
        if here and room and not self.repaints:
            return self.repair_step(here[0])
        if items and room and left == 4 and not any(r["helped"] for r in self.repaints):
            return self.repair_step(items[0])
        seen = [c for c in KINDS if self.compared[(anchor, c)]["counts"]["camera"]]
        seen += [c for c in KINDS if c not in seen and self.compared[(anchor, c)]["counts"]["painted"]]
        for kind in seen:
            spots = [round(self.m.n_paint * f) for f in SPREAD]
            spots = [k for k in spots if abs(k - anchor) >= self.m.n_paint // 10 and (k, kind) not in self.compared]
            if spots and left >= 2:
                return f"compare {kind} at k={spots[0]}. Still to compare across the drive: {', '.join(f'k={k}' for k in spots)}."
        if items and room:
            return self.repair_step(items[0])
        return "report, since more checks would not change the list."

    def find(self, k, view, concept, window_ms=None):
        result = self.base(k)
        k = result["k"]
        concept = _concept(concept)
        if view not in VIEWS:
            raise ToolError(f"view must be one of {', '.join(VIEWS)}")
        window_ms = _int(window_ms, "window_ms", 1, 3000, 100)
        objects = self.sam_on(k, view, concept, window_ms)
        grid = self.m.event_map(k, window_ms)
        backing = self.m.backing(k)
        base = block_median(grid)
        found = []
        for o in objects:
            found.append({"box": o["box"], "score": o["score"], "support": support(o["box"], grid, backing, base)})
        result.update(view=view, concept=concept, window_ms=window_ms, n=len(found), objects=found)
        self.touch(k)
        if view in ("camera", "events", "backing"):
            for f in found:
                if f["score"] >= 0.5:
                    self._sight(concept, k, f, "events" if view == "backing" else view, None)
        return result, []

    def _sight(self, concept, k, obj, sensor, painted):
        for s in self.sightings:
            if s["who"] == concept and s["k"] == k and box_iou(s["box"], obj["box"]) >= 0.3:
                if sensor not in s["from"]:
                    s["from"].append(sensor)
                if painted is not None:
                    s["painted"] = painted
                return
        self.sightings.append({
            "who": concept, "k": k, "t_s": round(self.m.t_ms(k) / 1000, 1), "box": obj["box"], "score": obj["score"],
            "from": [sensor], "events": obj["support"]["events"], "vs_median": obj["support"]["vs_median"], "painted": painted,
        })

    def _tally(self, k, concept, version=None):
        painted = self.sam_on(k, "painting", concept, 100, version)
        camera = self.sam_on(k, "camera", concept)
        grid = self.m.event_map(k, 100)
        return tally([dict(o) for o in painted], [dict(o) for o in camera], grid, self.m.backing(k))

    def compare(self, k, concept):
        result = self.base(k)
        k = result["k"]
        concept = _concept(concept)
        verdict = self._tally(k, concept)
        self.compared[(k, concept)] = verdict
        self.touch(k)
        for pair in verdict["in_both"]:
            self._sight(concept, k, pair["camera"], "camera", True)
        for obj in verdict["camera_no_painted"]:
            self._sight(concept, k, obj, "camera", False)
            if obj["support"]["backed"]:
                self._sight(concept, k, obj, "events", False)
        for obj in verdict["painted_no_camera"]:
            entry = {"who": concept, "k": k, "t_s": result["t_s"], "box": obj["box"], "score": obj["score"], "events": obj["support"]["events"], "vs_median": obj["support"]["vs_median"], "events_backed": obj["support"]["backed"]}
            if not any(e["who"] == concept and e["k"] == k and box_iou(e["box"], obj["box"]) >= 0.3 for e in self.invented):
                self.invented.append(entry)
        result.update(concept=concept, window_ms=100, **verdict)
        return result, []

    def warm(self, k, concepts, version=None):
        path = self.tmp / f"camera_{k}_100.png"
        if not path.exists():
            self.m.view_image(k, "camera", 100).save(path)
        jobs = [(k, "painting", c, 100, version) for c in concepts]
        if version is None:
            jobs += [(k, "camera", c) for c in concepts]
        with ThreadPoolExecutor(8) as pool:
            for job in [pool.submit(self.sam_on, *j) for j in jobs]:
                job.exception()

    def compare_many(self, concepts, k=None):
        k = self.only_k if k is None else k
        result = self.base(k)
        k = result["k"]
        if isinstance(concepts, str):
            concepts = concepts.split(",")
        if not isinstance(concepts, list) or not concepts:
            raise ToolError("concepts must be a list of noun phrases")
        concepts = list(dict.fromkeys(_concept(c) for c in concepts))[:8]
        self.warm(k, concepts)
        rows = {}
        for c in concepts:
            one = self.compare(k, c)[0]
            rows[c] = {"counts": one["counts"], "camera_no_painted": one["camera_no_painted"], "painted_no_camera": one["painted_no_camera"]}
        result.update(window_ms=100, by_concept=rows)
        aim = self.first_aim(k)
        if aim:
            result["aim_first"] = aim
        return result, []

    def first_aim(self, k):
        left = [(c, o) for (f, c), v in self.compared.items() if f == k for o in v["camera_no_painted"]]
        if not left:
            return None
        who, obj = min(left, key=lambda p: (p[0] not in PEOPLE, not p[1]["support"]["backed"], -box_area(p[1]["box"])))
        x0, y0, x1, y1 = obj["box"]
        return {"who": who, "box": obj["box"], "x": (x0 + x1) // 2, "y": (y0 + y1) // 2, "r": max(x1 - x0, y1 - y0) // 2 + 12}

    def finish(self):
        return self.summary(), []

    def settled(self):
        k = self.only_k
        if any(r["helped"] for r in self.repaints):
            return True
        here = [v for (f, _), v in self.compared.items() if f == k]
        return bool(here) and not any(v["camera_no_painted"] for v in here)

    def end_state(self):
        k = self.only_k
        left = [s for s in self.sightings if s["k"] == k and s["painted"] is False and not s.get("restored_by")]
        good = next((r for r in self.repaints if r["helped"]), None)
        state = "verified" if good else "unchanged" if self.repaints or self.refused or not left else "evidence"
        return {
            "state": state, "k": k, "left_out": len(left), "who": good["aimed_at"]["who"] if good else None,
            "repaints": len(self.repaints), "refused": self.refused, "withheld": self.withheld,
            "looked": sum(1 for s in left if s.get("wide_ms")), "held": sum(1 for s in left if s.get("wide_held")),
        }

    def _looked(self, k, target):
        passed, looked = [], False
        for concept, v in target.items():
            for obj in v["camera_no_painted"]:
                for s in self.sightings:
                    if s["k"] != k or s["who"] != concept or box_iou(s["box"], obj["box"]) < LOOKED_IOU or not s.get("wide_ms"):
                        continue
                    looked = True
                    if s.get("wide_held") or self.only_k is not None:
                        passed.append((concept, obj["box"], s["wide_ms"]))
                    break
        passed.sort(key=lambda p: aim_rank(p[0], p[1]))
        return passed, looked

    def native(self, k):
        path = self.versions.get((k, self.latest.get(k))) or self.m.painting_path(k)
        return Image.open(path).convert("RGB")

    def paint_patch(self, k, concept, box, note, window, version, out):
        canvas = self.native(k)
        scale = canvas.width / W
        area = crop_area(box)
        full = tuple(round(v * scale) for v in area)
        size = (full[2] - full[0], full[3] - full[1])
        zoom = CROP_PX / (area[2] - area[0])
        crop = canvas.crop(full).resize((CROP_PX, CROP_PX), Image.LANCZOS)
        rect = [(box[0] - area[0]) * zoom, (box[1] - area[1]) * zoom, (box[2] - area[0]) * zoom, (box[3] - area[1]) * zoom]
        ImageDraw.Draw(crop).rectangle(rect, outline=(0, 255, 0), width=3)
        first, second, raw = self.tmp / f"crop_{k}.{version}.png", self.tmp / f"crop_events_{k}.{version}.png", self.tmp / f"patch_{k}.{version}.webp"
        crop.save(first)
        self.m.render_events(k, window).crop(area).resize((CROP_PX, CROP_PX), Image.BILINEAR).save(second)
        self.calls["paint"] += 1
        painter = self.painter or pipeline()[0].paint
        done = painter(ensure_key(), first, second, raw, prompt=crop_prompt(concept, note), size=f"{CROP_PX}x{CROP_PX}")
        if not isinstance(done, float):
            self.refused = str(done)[:80]
            raise ToolError(f"Muse Image did not paint this frame: {str(done)[:80]}")
        small = Image.open(raw).convert("RGB").resize(size, Image.LANCZOS)
        phrases = [concept] + ([ALSO_ASK[concept]] if concept in ALSO_ASK else [])
        best, overlap = None, 0.0
        for phrase in phrases:
            self.calls["sam"] += 1
            for r in self.segment(raw, phrase):
                fit = box_iou([v * r["width"] / CROP_PX for v in rect], r["box"]) if r["score"] >= MIN_SCORE else 0.0
                if fit > overlap:
                    best, overlap = r, fit
        if best is None or overlap < PATCH_MATCH:
            return {"ok": False, "why": "no object in the patch"}, None
        hard = Image.fromarray(mask_in(best).astype(np.uint8) * 255).resize(size, Image.NEAREST)
        tall = max(1, (best["box"][3] - best["box"][1]) * size[1] / best["height"])
        grow = max(1, round(tall * PATCH_GROW))
        soft = hard.filter(ImageFilter.MaxFilter(2 * grow + 1)).filter(ImageFilter.GaussianBlur(grow))
        check = patch_check(np.asarray(canvas.crop(full)), np.asarray(small), np.asarray(soft, np.float32) / 255, np.asarray(hard) > 127, concept)
        canvas.paste(small, full[:2], soft)
        canvas.save(out, "WEBP", quality=92)
        ys, xs = np.nonzero(np.asarray(soft) > 8)
        check["box"] = [round(v, 1) for v in (area[0] + xs.min() / scale, area[1] + ys.min() / scale, area[0] + (xs.max() + 1) / scale, area[1] + (ys.max() + 1) / scale)]
        check["score"] = round(best["score"], 3)
        return check, out

    def face_gate(self, k, concept, box, out, version):
        if concept in PEOPLE and box[3] - box[1] < FACING_MIN_PERSON:
            return {"bad": [], "skipped": "too small to show which way it faces"}
        area = crop_area(box)
        camera, painted = self.tmp / f"facing_cam_{k}.{version}.png", self.tmp / f"facing_paint_{k}.{version}.png"
        self.m.view_image(k, "camera", 100).crop(area).resize((768, 768), Image.LANCZOS).save(camera)
        Image.open(out).convert("RGB").resize((W, H), Image.LANCZOS).crop(area).resize((768, 768), Image.LANCZOS).save(painted)
        return self.facing(camera, painted, concept)

    def repaint(self, k, note, evidence_window_ms=None, concept=None):
        result = self.base(k)
        k = result["k"]
        if not isinstance(note, str) or not note.strip() or len(note) > 200:
            raise ToolError("note must be one sentence under 200 characters")
        if BANNED_NOTE.search(note):
            raise ToolError("note may not ask for a woman, a child, text or a logo; say a person or a vehicle")
        if self.only_k is not None:
            if time.time() - self.started > INSPECT_REPAINT_BY:
                raise ToolError("no time left for a repaint in this run; call report")
            for c in KINDS:
                if (k, c) not in self.compared:
                    self.compare(k, c)
        before = {c: v for (f, c), v in self.compared.items() if f == k}
        if not before:
            raise ToolError(f"run compare on k={k} before ordering a repaint")
        target = before
        if concept is not None:
            concept = _concept(concept)
            if concept not in before:
                raise ToolError(f"run compare on k={k} for {concept} before ordering its repaint")
            target = {concept: before[concept]}
        if self.only_k is not None and (k, "person") not in self.tried and before.get("person", {}).get("camera_no_painted"):
            if concept not in (None, "person"):
                raise ToolError(f"a person is left out at k={k}, and people come first: look at aim_first, then repaint concept person")
            target = {"person": before["person"]}
        if not any(v["camera_no_painted"] for v in target.values()):
            raise ToolError(f"compare left nothing out at k={k}, so there is nothing to restore")
        passed, looked = self._looked(k, target)
        if not passed:
            if looked:
                raise ToolError(f"the events did not hold the left-out object you looked at on k={k}, so there is not enough evidence for a repaint")
            raise ToolError(f"look at the left-out box on the events at {WIDE_MS[0]} to {WIDE_MS[1]} ms on k={k} before ordering a repaint")
        if len(self.repaints) >= self.repaint_cap:
            raise ToolError(f"this run may repaint {self.repaint_cap} times")
        if self.paint_guard:
            refused = self.paint_guard()
            if refused:
                raise ToolError(refused)
        aim_concept, aim_box, _ = passed[0]
        window = _int(evidence_window_ms, "evidence_window_ms", 10, 3000, CROP_WINDOW_MS)
        self.tried.add((k, aim_concept))
        folder = (self.tmp if self.live else self.m.root / "paint" / "repaint")
        folder.mkdir(parents=True, exist_ok=True)
        number = 1
        while (folder / f"{k:04d}.r{number}.webp").exists():
            number += 1
        version = f"r{number}"
        out = folder / f"{k:04d}.{version}.webp"
        started = time.time()
        patch, _ = self.paint_patch(k, aim_concept, aim_box, note.strip(), window, version, out)
        result.update(note=note.strip(), paint_seconds=round(time.time() - started, 1), aimed_at={"who": aim_concept, "box": aim_box}, guided_by="camera", evidence_window_ms=window, patch=patch)
        if not patch["ok"]:
            result.update(helped=False, restored=0, restored_with_events=0, changes={}, unshown=patch["why"])
            self.repaints.append(result)
            return result, []
        result["version"] = version
        self.versions[(k, version)] = out
        with ThreadPoolExecutor(2) as pool:
            gate = pool.submit(self.rules, out, patch["box"]) if self.rules else None
            facing = pool.submit(self.face_gate, k, aim_concept, aim_box, out, version) if self.facing else None
            if self.only_k is not None:
                self.warm(k, list(before), version)
            broken = gate.result() if gate else []
            if facing:
                result["facing"] = facing.result()
                broken = [*broken, *result["facing"]["bad"]]
        verdicts = {c: self._tally(k, c, version) for c in before}
        if not self.live:
            image = publish_repaint(self.m, self.web, out)
            if image:
                result["image"] = image
        changes = {}
        for c, after in verdicts.items():
            changes[c] = {"before": before[c]["counts"], "after": after["counts"], "whole_frame_invented_change": after["counts"]["painted_no_camera"] - before[c]["counts"]["painted_no_camera"], **restored(before[c], after, patch["box"])}
        result["changes"] = changes
        aim_back = any(box_iou(aim_box, b) >= 0.5 for b in changes[aim_concept]["restored"])
        with_events = sum(v["restored_with_events"] for v in changes.values())
        result["helped"] = aim_back and with_events > 0 and not broken and all(v["invented_change"] <= 0 and not v["lost"] for v in changes.values())
        if broken:
            result["withheld"] = broken
            self.withheld += broken
        result["restored"] = sum(len(v["restored"]) for v in changes.values())
        result["restored_with_events"] = sum(v["restored_with_events"] for v in changes.values())
        if result["helped"]:
            self.latest[k] = version
            for c, after in verdicts.items():
                self.compared[(k, c)] = after
        for c, v in changes.items():
            for s in self.sightings:
                if s["k"] == k and s["who"] == c and s["painted"] is False and any(box_iou(s["box"], b) >= 0.5 for b in v["restored"]):
                    s["restored_by"] = version
        self.repaints.append(result)
        return result, [(f"The repaint of k={k}", fit_for_model(Image.open(out).convert("RGB").resize((W, H), Image.LANCZOS)))]

    def summary(self):
        left = [s for s in self.sightings if s["painted"] is False]
        return {
            "frames_checked": len(self.frames), "frames": sorted(self.frames),
            "found": len(self.sightings), "left_out": len(left),
            "invented": len(self.invented),
            "invented_no_events": sum(1 for e in self.invented if not e["events_backed"]),
            "repaints": len(self.repaints), "repaints_helped": sum(1 for r in self.repaints if r["helped"]),
            "restored": sum(r["restored"] for r in self.repaints),
            "sam_calls": self.calls["sam"], "paint_calls": self.calls["paint"],
        }

    def drive_list(self):
        rows = []
        for s in self.sightings:
            painting = {True: "painted", False: "left out", None: "not compared"}[s["painted"]]
            if s.get("restored_by"):
                painting = f"left out, brought back by repaint {s['restored_by']}"
            rows.append({"who": s["who"], "t_s": s["t_s"], "k": s["k"], "box": s["box"], "from": sorted(s["from"]), "events": s["events"], "painting": painting})
        for e in self.invented:
            rows.append({"who": e["who"], "t_s": e["t_s"], "k": e["k"], "box": e["box"], "from": [], "events": e["events"], "painting": "invented"})
        rows.sort(key=lambda r: (r["t_s"], r["who"], r["box"]))
        return rows

    def drive_lines(self):
        lines = []
        for r in self.drive_list():
            where = f"{r['t_s']:.1f} s, {r['who']} at {r['box']}"
            if r["painting"] == "invented":
                lines.append(f"{where}: only in the painting, the camera has no match; {r['events']} events in that box over 100 ms.")
            else:
                lines.append(f"{where}: from the {' and the '.join(r['from'])}, {r['events']} events over 100 ms; painting: {r['painting']}.")
        return lines

    def auto_report(self):
        s = self.summary()
        finding = f"Checked {s['frames_checked']} frames and found {s['found']} objects, {s['left_out']} of them left out of the painting."
        invented = f"The painting showed {s['invented']} objects with no camera match."
        return {"finding": finding, "view": "events+camera", "invented": invented}

    def report(self, finding, view, invented, auto=False):
        problems = []
        for text in (finding, invented):
            if not isinstance(text, str) or not text.strip():
                raise ToolError("finding and invented must be sentences with numbers from tool results")
            problems += audit_text(text, self.pool)
            if FIELD_NAME.search(text):
                problems.append("say it in plain words, without tool field names or k=")
        if problems and not auto:
            raise ToolError("Refused: " + "; ".join(problems[:6]))
        return {
            "finding": finding.strip(), "view": view, "invented": invented.strip(), "auto": auto,
            "verdict": self.summary(), "sightings": self.sightings, "invented_list": self.invented, "drive_list": self.drive_list(), "lines": self.drive_lines(),
        }, []

    def learn(self, result):
        clean = {k: v for k, v in result.items() if k not in ("grid", "per_second")}
        self.pool |= numbers_in(clean)
        for row in result.get("per_second", []):
            self.pool |= numbers_in({k: v for k, v in row.items() if k != "grid"})


def publish_repaint(moment, web, path):
    folder = Path(web) / "moments" / moment.id
    if not folder.is_dir():
        return None
    target = folder / "repaint"
    target.mkdir(exist_ok=True)
    Image.open(path).convert("RGB").resize((1024, 768), Image.LANCZOS).save(target / path.name, "WEBP", quality=80)
    return "repaint/" + path.name


def parse_calls(message):
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        calls.append({"id": call.get("id") or f"call_{len(calls)}", "name": str(fn.get("name")).split(".")[-1], "arguments": fn.get("arguments")})
    if calls:
        return calls
    text = message.get("content")
    if not isinstance(text, str):
        return []
    for block in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>|```(?:json|tool_call)?\s*(\{.*?\})\s*```", text, re.S):
        raw = block[0] or block[1]
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("name"):
            calls.append({"id": f"text_{len(calls)}", "name": data["name"], "arguments": data.get("arguments") or data.get("parameters") or {}, "text": True})
    return calls


def parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        raise ToolError("arguments were not valid JSON") from None
    if not isinstance(data, dict):
        raise ToolError("arguments must be an object")
    return data


def run_together(calls):
    if len(calls) > 1 and all(c["name"] == "look" for c in calls[:2]):
        return calls[:2]
    return calls[:1]


def first_sentence(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return re.split(r"(?<=[.!?])\s", text.strip())[0][:200]


def prune_images(messages, keep=KEEP_IMAGES):
    seen = 0
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        fresh = []
        for part in reversed(content):
            if part.get("type") == "image_url":
                seen += 1
                if seen > keep:
                    continue
            fresh.append(part)
        message["content"] = list(reversed(fresh))


def live_image(path):
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((1024, 768), Image.LANCZOS).save(buf, "WEBP", quality=80)
    return data_url(buf.getvalue(), "image/webp")


def run_agent(moment, route, on_step=None, max_steps=MAX_STEPS, live=False, paint_guard=None, web=WEB, should_stop=None, tools=None, held=None, inspect=False, on_start=None):
    m = moment if isinstance(moment, Moment) else Moment(moment)
    kit = tools or Tools(m, live=live, paint_guard=paint_guard, web=web)
    trace = {
        "model": route.model, "served_on": route.served_on,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "moment": m.id, "steps": [],
    }
    if inspect:
        kit.only_k = m.check_k(held)
        kit.started = time.time()
    messages = [
        {"role": "system", "content": INSPECT_SYSTEM if inspect else SYSTEM},
        {"role": "user", "content": f"Moment {m.id}: a viewer holds frame k={held}, at {m.t_ms(held) / 1000:.1f} seconds. Settle that frame."} if inspect else {"role": "user", "content": f"Moment {m.id}: {m.seconds:g} seconds of driving at night, frames k=0 to {m.n_paint - 1}, one per 100 ms. " + (f"A viewer has held the drive at k={held}. Check that moment first against the sensors, one compare for every kind of object you can see there, then the rest of the drive. The viewer is looking at that moment, so your first repair goes to the largest omission at it." if held is not None else "Start where you judge best.")},
    ]
    if held is not None:
        trace["held"] = held
    guided = getattr(route, "guided", False)
    if guided:
        trace["guided"] = True
    handlers = {"scan": kit.scan, "look": kit.look, "find": kit.find, "compare": kit.compare, "repaint": kit.repaint, "report": kit.report}
    menu, grace = TOOLS, GRACE_STEPS
    if inspect:
        handlers = {"compare": kit.compare_many, "look": kit.look, "repaint": kit.repaint, "report": kit.finish}
        menu, grace = INSPECT_TOOLS, 0
    finished = False
    nudged = 0
    refusals = 0

    def emit(tool, args, reason, result, seconds, model_seconds):
        step = {"i": len(trace["steps"]), "tool": tool, "args": args, "reason": reason, "result": result, "seconds": round(seconds, 1), "model_seconds": round(model_seconds, 1)}
        trace["steps"].append(step)
        if on_step:
            on_step(step)
        return step

    try:
        while not finished and len(trace["steps"]) < max_steps + grace:
            if should_stop and should_stop():
                trace["stopped"] = "viewer left"
                break
            last_call = len(trace["steps"]) >= max_steps - 1
            offered = [t for t in menu if t["function"]["name"] == "report"] if last_call or nudged else menu
            if inspect and time.time() - kit.started > INSPECT_REPAINT_BY:
                offered = [t for t in menu if t["function"]["name"] == "report"]
            used = f"Steps used {len(trace['steps'])} of {max_steps}."
            if last_call:
                messages.append({"role": "user", "content": used + " This is your last step: call report now, in plain sentences, quoting numbers from tool results only."})
            else:
                waiting = kit.open_items()
                if guided:
                    messages.append({"role": "user", "content": used + " Next: " + kit.plan(held, max_steps - len(trace["steps"]))})
                else:
                    messages.append({"role": "user", "content": used + (" Omissions not yet repaired, largest first: " + "; ".join(waiting) + "." if waiting else "")})
            prune_images(messages)
            t0 = time.time()
            message = route.chat(messages, offered)
            model_seconds = time.time() - t0
            calls = parse_calls(message)
            if not calls:
                nudged += 1
                if nudged > 2:
                    break
                messages.append({"role": "assistant", "content": message.get("content") or ""})
                messages.append({"role": "user", "content": "Do not answer in prose. Call report now, in plain sentences, quoting numbers from tool results only."})
                trace.setdefault("prose", []).append({"after_step": len(trace["steps"]), "text": (message.get("content") or "")[:400], "finish_reason": message.get("finish_reason")})
                prune_images(messages, keep=0)
                continue
            if calls[0].get("text"):
                messages.append({"role": "assistant", "content": message.get("content") or ""})
            else:
                messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])}} for c in calls
                ]})
            images = []
            replies = []
            run_now = run_together(calls)
            for call in calls[len(run_now):]:
                replies.append((call, json.dumps({"not_run": "One call at a time when the next depends on the result. Decide this call again after the result above."})))
            for n, call in enumerate(run_now):
                t1 = time.time()
                args, reason, images_here = {}, "", []
                try:
                    peek = parse_args(call["arguments"]).get("reason") or first_sentence(message.get("content"))
                except ToolError:
                    peek = ""
                if not inspect and peek and call["name"] != "report" and not SENSOR_WORD.search(peek) and refusals < MAX_REASON_REFUSALS:
                    refusals += 1
                    replies.append((call, json.dumps({"not_run": "Your reason must name the sensor you chose, the camera or the events, and why."})))
                    continue
                try:
                    args = parse_args(call["arguments"])
                    reason = args.pop("reason", "") or first_sentence(message.get("content"))
                    handler = handlers.get(call["name"])
                    if handler is None:
                        raise ToolError(f"no tool named {call['name']}")
                    if not reason and not inspect:
                        raise ToolError("every call needs a reason")
                    allowed = {k: v for k, v in args.items() if v is not None}
                    if on_start:
                        on_start({"start": call["name"], "args": args})
                    result, images_here = handler(**allowed)
                    kit.learn(result)
                    finished = call["name"] == "report" or (inspect and kit.settled())
                except TypeError as e:
                    result = {"error": f"bad arguments for {call['name']}: {str(e)[:100]}"}
                except (ToolError, MomentError, MetaAPIError, ValueError) as e:
                    result = {"error": str(e)[:300]}
                emit(call["name"], args, reason, result, time.time() - t1 + (model_seconds if n == 0 else 0), model_seconds if n == 0 else 0)
                replies.append((call, json.dumps(result)))
                images += images_here
                if finished:
                    break
            if finished:
                break
            replies.sort(key=lambda pair: calls.index(pair[0]))
            for call, text in replies:
                if call.get("text"):
                    messages.append({"role": "user", "content": f"Result of {call['name']}: {text}"})
                else:
                    messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": text})
            if images:
                parts = []
                for label, image in images:
                    parts.append({"type": "text", "text": label})
                    parts.append({"type": "image_url", "image_url": {"url": data_url(png_bytes(image))}})
                messages.append({"role": "user", "content": parts})
    except spend.SpendStop as e:
        trace["stopped"] = str(e)
    except AgentError as e:
        trace["stopped"] = str(e)
    if inspect:
        trace["end"] = kit.end_state()
    elif not finished and "stopped" not in trace or (not finished and trace.get("stopped") and trace["stopped"] != "viewer left"):
        auto = kit.auto_report()
        result, _ = kit.report(auto["finding"], auto["view"], auto["invented"], auto=True)
        emit("report", {}, "The run ended before the model reported, so this report is arithmetic only.", result, 0, 0)
    if tools is None:
        kit.close()
    trace["seconds"] = round(sum(s["seconds"] for s in trace["steps"]), 1)
    return trace


def write_trace(trace, moment, route_name, web=WEB):
    m = moment if isinstance(moment, Moment) else Moment(moment)
    text = json.dumps(trace, indent=1)
    paths = [m.root]
    folder = Path(web) / "moments" / m.id
    if folder.is_dir():
        paths.append(folder)
    written = []
    for base in paths:
        (base / f"agent.{route_name}.json").write_text(text, encoding="utf-8")
        main = base / "agent.json"
        if route_name == "spark" and main.exists() and "glimmer" in json.loads(main.read_text(encoding="utf-8")).get("model", ""):
            continue
        main.write_text(text, encoding="utf-8")
        written.append(str(main))
    return written
