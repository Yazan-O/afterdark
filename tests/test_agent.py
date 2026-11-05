import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageFilter
import pytest

from afterdark import agent, server, spend, verdict
from afterdark.moment import H, Moment, MomentError, W


def word(x, y, p=1, dt=100):
    return x | (y << 10) | (p << 19) | (dt << 20)


@pytest.fixture(autouse=True)
def plain_patches(monkeypatch):
    monkeypatch.setattr(agent, "patch_check", lambda *args: {"ok": True, "why": ""})


@pytest.fixture
def moment(tmp_path):
    root = tmp_path / "moments" / "demo"
    for name in ("ev10", "backing", "cam", "paint"):
        (root / name).mkdir(parents=True)
    n_paint, n_bins = 6, 18
    events = []
    index = [0]
    for b in range(n_bins):
        events += [word(100 + i % 40, 200 + i % 40) for i in range(200)]
        index.append(len(events))
    np.asarray(events, "<u4").tofile(root / "events_film.bin")
    meta = {"seq": "t", "t0_us": 0, "seconds": 0.6, "paint_fps": 10, "bin_fps": 30, "n_paint": n_paint, "n_bins": n_bins, "width": W, "height": H, "counts": [400] * n_bins, "on": [400] * n_bins, "index": index, "film_index": index, "camera_offsets_us": [i * 50000 for i in range(12)], "packing": "x10 y9 p1 dt12"}
    (root / "meta.json").write_text(json.dumps(meta))
    for k in range(n_paint):
        Image.new("L", (W, H), 20).save(root / "ev10" / f"{k:04d}.png")
        Image.new("L", (W, H), 40).save(root / "backing" / f"{k:04d}.png")
    for c in range(12):
        Image.new("RGB", (W, H), (10, 10, 10)).save(root / "cam" / f"{c:04d}.png")
    Image.new("RGB", (1792, 1344), (30, 30, 30)).save(root / "paint" / "0002.webp")
    Image.new("RGB", (8, 6), (1, 2, 3)).save(tmp_path / "moments" / "style.png")
    return Moment(root)


def record(box, score=0.9):
    return {"object_id": 0, "box": box, "width": W, "height": H, "score": score, "mask": {"encoding": "bad", "width": 1, "height": 1, "data": "x"}}


def textured():
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (128, 128, 3), np.uint8))


def fake_segment(answers):
    def segment(path, concept):
        name = str(path)
        if "patch_" in name:
            return [record(b) for b in answers.get("patch", [[0, 0, W - 1, H - 1]])]
        if name.endswith(".webp"):
            kind = "repaint" if ".r" in name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] else "painting"
        else:
            kind = "camera" if "camera" in name else "events"
        return [record(b) for b in answers.get(kind, [])]
    return segment


def obj(box, score=0.9):
    return {"box": box, "score": score, "mask": verdict.box_mask(box)}


def test_compare_arithmetic_on_masks():
    a = np.zeros((240, 320), bool)
    b = np.zeros((240, 320), bool)
    a[10:50, 10:50] = True
    b[20:60, 20:60] = True
    iou, contain = verdict.mask_overlap(a, b)
    assert iou == pytest.approx(900 / (1600 + 1600 - 900))
    assert contain == pytest.approx(900 / 1600)
    grid = np.zeros((H, W), np.float32)
    grid[100:200, 100:200] = 10
    backing = np.full((H, W), 50.0)
    painted = [obj([100, 100, 200, 200]), obj([500, 400, 560, 460])]
    camera = [obj([110, 110, 210, 210]), obj([300, 100, 360, 160])]
    v = verdict.tally(painted, camera, grid, backing)
    assert v["counts"]["in_both"] == 1
    assert v["counts"]["painted_no_camera"] == 1
    assert v["counts"]["camera_no_painted"] == 1
    assert v["painted_no_camera"][0]["box"] == [500, 400, 560, 460]
    assert v["camera_no_painted"][0]["box"] == [300, 100, 360, 160]
    assert all("mask" not in o for o in v["painted_no_camera"] + v["camera_no_painted"])


def test_tiny_object_inside_big_box_does_not_match():
    big = obj([300, 200, 400, 300])
    tiny = obj([340, 240, 350, 250])
    both, only_p, only_c = verdict.match([tiny], [big])
    assert both == [] and only_p == [0] and only_c == [0]


def test_low_scores_are_dropped():
    grid = np.ones((H, W), np.float32)
    v = verdict.tally([obj([0, 0, 10, 10], 0.2)], [], grid, np.zeros((H, W)))
    assert v["counts"]["painted"] == 0


def verdict_of(left_out, invented=0):
    return {"camera_no_painted": left_out, "counts": {"camera_no_painted": len(left_out), "painted_no_camera": invented}}


def test_restored_needs_a_gain_and_no_new_invention():
    box = [10, 10, 50, 50]
    weak = {"box": box, "support": {"backed": False}}
    before = verdict_of([weak])
    result = verdict.restored(before, verdict_of([]))
    assert result["helped"] is True and result["restored"] == [box]
    assert result["restored_with_events"] == 0
    assert verdict.restored(before, verdict_of([], invented=2))["helped"] is False
    assert verdict.restored(before, verdict_of([weak]))["helped"] is False


def test_restored_reports_event_support_apart_from_helped():
    box = [10, 10, 50, 50]
    other = [200, 200, 240, 240]
    backed = {"box": box, "support": {"backed": True}}
    before = verdict_of([backed, {"box": other, "support": {"backed": False}}], invented=1)
    result = verdict.restored(before, verdict_of([{"box": other}], invented=1))
    assert result == {"restored": [box], "restored_with_events": 1, "left_out_change": -1, "invented_change": 0, "lost": [], "helped": True}


def person_scene():
    rng = np.random.default_rng(1)
    scene = rng.normal(40, 12, (200, 200, 3)).clip(0, 255)
    hard = np.zeros((200, 200), bool)
    hard[60:160, 85:115] = True
    hard[60:80, 85:94] = hard[60:80, 106:115] = False
    painted = scene.copy()
    painted[hard] = rng.normal(15, 12, (hard.sum(), 3)).clip(0, 255)
    return scene.astype(np.uint8), painted.astype(np.uint8), hard


def test_patch_check_passes_a_sharp_object_and_stops_blur_stickers_and_marks():
    before, painted, hard = person_scene()
    soft = verdict.blurred(hard.astype(np.float32), 2)
    assert verdict.patch_check(before, painted, soft, hard, "person")["ok"] is True
    smeared = np.asarray(Image.fromarray(painted).filter(ImageFilter.GaussianBlur(3)))
    assert verdict.patch_check(before, smeared, soft, hard, "person")["why"] == "blurred"
    box = np.zeros((200, 200), bool)
    box[60:160, 85:115] = True
    sticker = painted.copy()
    sticker[box] = np.random.default_rng(2).normal(170, 12, (box.sum(), 3)).clip(0, 255).astype(np.uint8)
    assert "pasted on" in verdict.patch_check(before, sticker, verdict.blurred(box.astype(np.float32), 2), box, "person")["why"]
    marked = painted.copy()
    marked[hard] = (0, 220, 0)
    assert "box mark left in" in verdict.patch_check(before, marked, soft, hard, "person")["why"]


def test_a_repaint_that_loses_a_matched_object_did_not_help():
    box, other = [10, 10, 50, 50], [200, 200, 240, 240]
    result = verdict.restored(verdict_of([{"box": box}]), verdict_of([{"box": other}]))
    assert result["restored"] == [box] and result["lost"] == [other] and result["helped"] is False


def test_masks_stay_inside_their_box_and_far_objects_do_not_match(monkeypatch):
    import meta_sam_parser
    monkeypatch.setattr(meta_sam_parser, "decode_mask_to_raster", lambda mask: bytes([1] * (mask.width * mask.height)))
    mask = {"encoding": "one_bit", "width": 10, "height": 20, "data": "x"}
    left = verdict.from_record({"box": [20, 200, 29, 219], "width": W, "height": H, "score": 0.9, "mask": mask})
    right = verdict.from_record({"box": [500, 200, 509, 219], "width": W, "height": H, "score": 0.9, "mask": mask})
    assert left["mask"].any() and not left["mask"][:, 40:].any()
    assert verdict.mask_overlap(left["mask"], right["mask"]) == (0.0, 0.0)
    both, only_p, only_c = verdict.match([left], [right])
    assert both == [] and only_p == [0] and only_c == [0]


def test_report_audit_numbers():
    pool = verdict.numbers_in({"events": 8244, "ratio": 2.517, "k": 30})
    assert verdict.audit_text("At frame 30 the car held 8,244 events.", pool) == []
    assert verdict.audit_text("The ratio was 2.52 at frame 30.", pool) == []
    assert verdict.audit_text("The car was there.", pool)
    assert verdict.audit_text("It held 999 events.", pool)
    assert verdict.audit_text("Two cars at 30.", pool)
    assert verdict.audit_text("SAM 3.1 found 8244 events.", pool) == []


def test_tool_arguments(moment):
    kit = agent.Tools(moment, segment=fake_segment({}))
    with pytest.raises(MomentError):
        kit.look(99, "events")
    with pytest.raises(agent.ToolError):
        kit.look(1, "radar")
    with pytest.raises(MomentError):
        kit.look(1, "painting")
    with pytest.raises(agent.ToolError):
        kit.find(1, "camera", "")
    with pytest.raises(agent.ToolError):
        kit.repaint(2, "a car ahead")
    result, images = kit.look("1", "events", "5000", 120, 220, 30)
    assert result["window_ms"] == 3000 and result["events"] > 0 and result["r"] == 30
    assert images[0][1].size[0] >= 384
    kit.close()


def test_events_window_uses_true_counts(moment):
    x, _, _, w = moment.events(2, 100)
    assert len(x) > 0 and w[0] == pytest.approx(2.0)
    assert moment.event_map(2, 100).sum() == pytest.approx(w.sum())


def test_scan_reads_real_activity(moment):
    kit = agent.Tools(moment, segment=fake_segment({}))
    result, images = kit.scan()
    assert result["painted"] == [2]
    assert result["per_second"][0]["events"] == 400 * 18
    assert len(images) == 2
    kit.close()


def test_note_rules_and_repaint(moment, monkeypatch, tmp_path):
    calls = []

    def painter(key, first, second, out, prompt, size=None):
        calls.append((str(first), prompt))
        textured().save(out, "WEBP")
        return 1.5

    monkeypatch.setenv("MODEL_API_KEY", "test")
    answers = {"painting": [], "camera": [[100, 200, 140, 240]], "repaint": [[100, 200, 140, 240]]}
    web = tmp_path / "web"
    (web / "moments" / "demo").mkdir(parents=True)
    kit = agent.Tools(moment, segment=fake_segment(answers), painter=painter, web=web, repaint_cap=1)
    before, _ = kit.compare(2, "car")
    assert before["counts"]["camera_no_painted"] == 1
    with pytest.raises(agent.ToolError):
        kit.repaint(2, "a woman at the kerb")
    with pytest.raises(agent.ToolError, match="300 to 1000"):
        kit.repaint(2, "a car ahead")
    assert calls == []
    wide, _ = kit.look(2, "events", 500, 120, 220, 40)
    assert wide["left_out_here"][0]["box"] == [100, 200, 141, 241] and wide["left_out_here"][0]["events"] > 0
    result, images = kit.repaint(2, "a car ahead")
    assert result["evidence_window_ms"] == 300 and result["aimed_at"]["box"] == [100, 200, 141, 241]
    assert result["helped"] is True and result["restored"] == 1
    assert result["image"] == "repaint/0002.r1.webp"
    assert (web / "moments" / "demo" / "repaint" / "0002.r1.webp").exists()
    assert (moment.root / "paint" / "repaint" / "0002.r1.webp").exists()
    assert moment.painted() == [2]
    assert "green rectangle" in calls[0][1]
    assert kit.drive_list()[0]["painting"] == "left out, brought back by repaint r1"
    assert "crop_2" in calls[0][0]
    with pytest.raises(agent.ToolError):
        kit.repaint(2, "a car ahead")
    kit.close()


def test_a_repaint_that_helped_is_what_the_next_check_sees(moment, monkeypatch, tmp_path):
    def painter(key, first, second, out, prompt, size=None):
        textured().save(out, "WEBP")
        return 1.5

    monkeypatch.setenv("MODEL_API_KEY", "test")
    answers = {"painting": [], "camera": [[100, 200, 140, 240]], "repaint": [[100, 200, 140, 240]]}
    kit = agent.Tools(moment, segment=fake_segment(answers), painter=painter, web=tmp_path / "none", repaint_cap=2)
    kit.compare(2, "car")
    kit.look(2, "events", 500)
    assert kit.repaint(2, "a car ahead")[0]["helped"] is True
    assert kit.latest[2] == "r1"
    assert kit.compare(2, "car")[0]["counts"]["camera_no_painted"] == 0
    assert kit.open_items() == []
    with pytest.raises(agent.ToolError, match="nothing to restore"):
        kit.repaint(2, "a car ahead")
    kit.close()


def test_a_repaint_with_no_events_behind_it_is_not_helped(moment, monkeypatch, tmp_path):
    def painter(key, first, second, out, prompt, size=None):
        textured().save(out, "WEBP")
        return 1.5

    monkeypatch.setenv("MODEL_API_KEY", "test")
    answers = {"painting": [], "camera": [[100, 200, 140, 240]], "repaint": [[100, 200, 140, 240]]}
    kit = agent.Tools(moment, segment=fake_segment(answers), painter=painter, web=tmp_path / "none", repaint_cap=2)
    kit.compare(2, "car")
    kit.look(2, "events", 500)
    real = agent.restored
    monkeypatch.setattr(agent, "restored", lambda a, b, within=None: {**real(a, b, within), "restored_with_events": 0})
    result = kit.repaint(2, "a car ahead")[0]
    assert result["restored"] == 1 and result["helped"] is False
    assert 2 not in kit.latest


def test_a_repaint_that_failed_is_dropped(moment, monkeypatch, tmp_path):
    def painter(key, first, second, out, prompt, size=None):
        textured().save(out, "WEBP")
        return 1.5

    monkeypatch.setenv("MODEL_API_KEY", "test")
    answers = {"painting": [], "camera": [[100, 200, 140, 240]], "repaint": [[400, 300, 500, 400]]}
    kit = agent.Tools(moment, segment=fake_segment(answers), painter=painter, web=tmp_path / "none", repaint_cap=2)
    kit.compare(2, "car")
    kit.look(2, "events", 500)
    result = kit.repaint(2, "a car ahead")[0]
    assert result["helped"] is False and result["changes"]["car"]["whole_frame_invented_change"] == 1
    assert 2 not in kit.latest
    assert kit.compare(2, "car")[0]["counts"]["camera_no_painted"] == 1
    kit.close()


def test_repaint_is_bound_to_the_object_the_events_held(moment, monkeypatch, tmp_path):
    def painter(key, first, second, out, prompt, size=None):
        textured().save(out, "WEBP")
        return 1.5

    monkeypatch.setenv("MODEL_API_KEY", "test")
    answers = {"painting": [], "camera": [[100, 200, 140, 240], [400, 300, 460, 400]], "repaint": [[100, 200, 140, 240]]}
    kit = agent.Tools(moment, segment=fake_segment(answers), painter=painter, web=tmp_path / "none", repaint_cap=2)
    kit.compare(2, "car")
    kit.look(2, "events", 500, 430, 350, 40)
    with pytest.raises(agent.ToolError, match="not enough evidence"):
        kit.repaint(2, "a car ahead")
    kit.look(2, "events", 500, 120, 220, 25)
    result = kit.repaint(2, "a car ahead")[0]
    assert result["aimed_at"]["box"] == [100, 200, 141, 241] and result["helped"] is True
    kit.close()


def test_repaint_needs_no_style_or_painter_input_when_it_edits(moment, monkeypatch, tmp_path):
    def painter(key, first, second, out, prompt, size=None):
        textured().save(out, "WEBP")
        return 1.5

    monkeypatch.setenv("MODEL_API_KEY", "test")
    (moment.root / "ev10" / "0002.png").unlink()
    (moment.root.parent / "style.png").unlink()
    answers = {"painting": [], "camera": [[100, 200, 140, 240]], "repaint": [[100, 200, 140, 240]]}
    kit = agent.Tools(moment, segment=fake_segment(answers), painter=painter, web=tmp_path / "none")
    kit.compare(2, "car")
    kit.look(2, "events", 500)
    assert kit.repaint(2, "a car ahead")[0]["helped"] is True
    kit.close()


def test_a_held_concept_cannot_name_a_banned_subject(moment):
    kit = agent.Tools(moment, segment=fake_segment({}))
    with pytest.raises(agent.ToolError):
        kit.compare(2, "a child")
    kit.close()


def test_paint_stops_at_a_content_filter_and_keeps_the_person_rule(monkeypatch, tmp_path):
    import httpx
    paint_moment, _ = agent.pipeline()
    sent = []
    answers = [(500, "busy"), (500, '{"error":{"code":"content_policy_violation"}}')]

    def post(url, headers, json, timeout):
        sent.append(json["prompt"])
        status, text = answers[len(sent) - 1]
        return httpx.Response(status, text=text)

    monkeypatch.setattr(paint_moment.httpx, "post", post)
    monkeypatch.setattr(paint_moment.time, "sleep", lambda s: None)
    monkeypatch.setenv("AFTERDARK_SPEND", str(tmp_path / "spend.csv"))
    image = tmp_path / "a.png"
    Image.new("L", (8, 8)).save(image)
    done = paint_moment.paint("k", image, image, tmp_path / "o" / "x.webp")
    assert "content filter" in done and len(sent) == 2
    assert all("adult man" in p and "face in shadow" in p for p in sent)


class FakeRoute:
    model = "fake-model"
    served_on = "test"
    name = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append((json.loads(json.dumps(messages)), [t["function"]["name"] for t in tools]))
        name, args = self.script.pop(0)
        return {"content": "", "tool_calls": [{"id": f"c{len(self.seen)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_trace_format_and_report_audit(moment):
    route = FakeRoute([
        ("scan", {"reason": "Events first to see where the drive changes."}),
        ("compare", {"k": 2, "concept": "car", "reason": "Camera against painting at this frame."}),
        ("report", {"finding": "A claim with 12345 events.", "view": "camera", "invented": "At k 2 nothing.", "reason": "Done."}),
        ("report", {"finding": "At k 2 the camera held 1 car.", "view": "camera", "invented": "At k 2 the painting showed 0 cars.", "reason": "Done."}),
    ])
    kit = agent.Tools(moment, segment=fake_segment({"camera": [[100, 200, 140, 240]]}))
    steps = []
    trace = agent.run_agent(moment, route, on_step=steps.append, tools=kit)
    assert [s["tool"] for s in trace["steps"]] == ["scan", "compare", "report", "report"]
    assert "error" in trace["steps"][2]["result"]
    final = trace["steps"][3]["result"]
    assert final["verdict"]["left_out"] == 1 and final["auto"] is False
    assert any("left out" in line for line in final["lines"])
    for key in ("model", "served_on", "started", "moment", "steps"):
        assert key in trace
    for i, step in enumerate(trace["steps"]):
        assert step["i"] == i and step["reason"] and isinstance(step["seconds"], float)
        assert "reason" not in step["args"]
    text = json.dumps(trace)
    assert "base64" not in text and "mask" not in text
    assert steps == trace["steps"]
    assert any(m.get("role") == "tool" for m in route.seen[1][0])
    assert any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]) for m in route.seen[1][0])
    kit.close()


class BatchRoute(FakeRoute):
    def chat(self, messages, tools):
        self.seen.append((json.loads(json.dumps(messages)), [t["function"]["name"] for t in tools]))
        calls = self.script.pop(0)
        return {"content": "", "tool_calls": [{"id": f"c{len(self.seen)}_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}} for n, (name, args) in enumerate(calls)]}


def test_one_call_at_a_time_and_steps_shown(moment):
    compare = {"k": 2, "concept": "car", "reason": "Camera against painting."}
    report = {"finding": "At 0.2 seconds the camera held 1 car.", "view": "camera", "invented": "The painting showed 0 cars at 0.2 seconds.", "reason": "Done."}
    route = BatchRoute([[("compare", compare), ("compare", {**compare, "concept": "person"})], [("report", report)]])
    kit = agent.Tools(moment, segment=fake_segment({"camera": [[100, 200, 140, 240]]}))
    trace = agent.run_agent(moment, route, tools=kit)
    assert [s["tool"] for s in trace["steps"]] == ["compare", "report"]
    assert any("Steps used 0 of 14" in str(m.get("content")) for m in route.seen[0][0])
    assert any("Steps used 1 of 14" in str(m.get("content")) and "k=2 car" in str(m.get("content")) for m in route.seen[1][0])
    replies = [m for m in route.seen[1][0] if m.get("role") == "tool"]
    assert len(replies) == 2 and "not_run" in replies[1]["content"]
    kit.close()


def test_held_frame_is_told_and_recorded(moment):
    route = FakeRoute([("report", {"finding": "At 0.2 seconds nothing was found.", "view": "camera", "invented": "The painting showed 0 objects at 0.2 seconds.", "reason": "Done."})])
    kit = agent.Tools(moment, segment=fake_segment({}))
    kit.learn({"t": 0.2, "n": 0})
    trace = agent.run_agent(moment, route, tools=kit, held=2)
    assert trace["held"] == 2 and "held the drive at k=2" in str(route.seen[0][0])
    kit.close()


def test_reason_must_name_a_sensor(moment):
    route = FakeRoute([
        ("look", {"k": 1, "view": "events", "reason": "Because it seems useful."}),
        ("look", {"k": 1, "view": "events", "reason": "The events can settle this."}),
        ("report", {"finding": "At 0.1 seconds nothing was found.", "view": "events", "invented": "The painting showed 0 objects.", "reason": "Done."}),
    ])
    kit = agent.Tools(moment, segment=fake_segment({}))
    trace = agent.run_agent(moment, route, tools=kit)
    assert [s["tool"] for s in trace["steps"]] == ["look", "report"]
    assert "not_run" in json.dumps(route.seen[1][0])
    kit.close()


def test_report_refuses_field_names(moment):
    kit = agent.Tools(moment, segment=fake_segment({}))
    kit.learn({"n": 3})
    with pytest.raises(agent.ToolError, match="plain words"):
        kit.report("The camera_no_painted count was 3.", "camera", "The painting showed 3 objects.")
    kit.close()


def test_step_cap_ends_with_arithmetic_report(moment):
    route = FakeRoute([("look", {"k": 1, "view": "events", "reason": "Events at this frame."})] * 8)
    kit = agent.Tools(moment, segment=fake_segment({}))
    trace = agent.run_agent(moment, route, max_steps=3, tools=kit)
    assert route.seen[2][1] == ["report"]
    assert trace["steps"][-1]["tool"] == "report" and trace["steps"][-1]["result"]["auto"] is True
    kit.close()


def test_spend_stop_ends_the_run(moment):
    class Broke(FakeRoute):
        def chat(self, messages, tools):
            raise spend.SpendStop("stopped")
    kit = agent.Tools(moment, segment=fake_segment({}))
    trace = agent.run_agent(moment, Broke([]), tools=kit)
    assert trace["stopped"] == "stopped"
    kit.close()


def test_text_tool_calls_are_parsed():
    calls = agent.parse_calls({"content": 'ok <tool_call>{"name": "look", "arguments": {"k": 1}}</tool_call>'})
    assert calls[0]["name"] == "look" and calls[0]["arguments"] == {"k": 1}
    assert agent.parse_calls({"content": "no calls here"}) == []
    with pytest.raises(agent.ToolError):
        agent.parse_args("{not json")


def test_plan_orders_kinds_then_repair_then_spread(moment):
    kit = agent.Tools(moment, segment=fake_segment({}))
    assert kit.plan(1, 14).startswith("compare person at k=1")
    for c in agent.KINDS:
        kit.compared[(1, c)] = {"camera_no_painted": [], "counts": {"camera": 1 if c == "car" else 0, "painted": 0}}
    kit.sightings.append({"who": "tram", "k": 1, "t_s": 0.1, "box": [10, 10, 60, 60], "score": 0.9, "from": ["camera"], "events": 5, "vs_median": 2, "painted": False})
    assert kit.plan(1, 14).startswith("look at the tram")
    kit.sightings[0]["wide_ms"] = 800
    assert "on the camera" in kit.plan(1, 14)
    kit.camera_crops.append((1, (0, 0, 100, 100)))
    assert kit.plan(1, 14).startswith("repaint the tram")
    assert kit.plan(1, 1).startswith("report")
    kit.tried.add((1, "tram"))
    assert kit.plan(1, 14).startswith("compare car at k=")
    kit.close()


def test_old_images_are_pruned():
    part = {"type": "image_url", "image_url": {"url": "x"}}
    messages = [{"role": "user", "content": [part, part, part]}, {"role": "user", "content": [part, part, part]}]
    agent.prune_images(messages, keep=4)
    assert sum(1 for m in messages for p in m["content"] if p["type"] == "image_url") == 4


def test_trace_written_without_replacing_glimmer(moment, tmp_path):
    web = tmp_path / "web"
    (web / "moments" / "demo").mkdir(parents=True)
    agent.write_trace({"model": "muse-glimmer-30b", "steps": []}, moment, "glimmer", web=web)
    agent.write_trace({"model": "muse-spark-1.3", "steps": []}, moment, "spark", web=web)
    assert json.loads((moment.root / "agent.json").read_text())["model"] == "muse-glimmer-30b"
    assert json.loads((web / "moments" / "demo" / "agent.spark.json").read_text())["model"] == "muse-spark-1.3"


def test_caps():
    now = [1000000.0]
    caps = server.Caps(2, 3, clock=lambda: now[0])
    assert caps.acquire("a") is None
    assert caps.acquire("a")
    caps.release("a")
    assert caps.acquire("a") is None
    caps.release("a")
    assert "address" in caps.acquire("a")
    assert caps.acquire("b") is None
    caps.release("b")
    assert "Today" in caps.acquire("c")
    assert caps.charge("c")
    now[0] += 86400
    assert caps.acquire("a") is None


@pytest.fixture
def live(moment, tmp_path):
    painted = []

    def painter(w, m, k):
        painted.append((m, k))
        if len(painted) > 1:
            raise spend.SpendStop("stop")
        return {"image": "data:image/webp;base64,AAAA", "seconds": 1.0, "model": "muse-image-1.0"}

    def runner(root, step, guard, should_stop, w, held=None):
        step({"i": 0, "tool": "scan", "args": {}, "reason": "r", "result": {"held": held}, "seconds": 0.1})
        return {"steps": [], "end": {"state": "unchanged"}} if held is not None else {"steps": []}

    srv = server.make_server(port=0, web=tmp_path, painter=painter, agent_runner=runner)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(base, body):
    req = Request(base + "/api/paint", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urlopen(req, timeout=10))
    except HTTPError as e:
        return json.load(e)


def test_server_routes(live):
    assert set(json.load(urlopen(live + "/api/status", timeout=10))) == {"live", "models"}
    assert post(live, {"moment": "demo", "k": 2})["image"].startswith("data:image/webp")
    fallback = post(live, {"moment": "demo", "k": 2})
    assert "credit" in fallback["error"] and fallback["saved"] == "moments/demo/paint/0002.webp"
    assert "Only frames" in post(live, {"moment": "demo", "k": 3})["error"]
    assert "error" in post(live, {"moment": "../x", "k": 3})
    text = urlopen(live + "/api/agent?moment=demo", timeout=10).read().decode()
    rows = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
    assert rows[0]["tool"] == "scan" and rows[-1]["done"] is True
    text = urlopen(live + "/api/agent?moment=demo&k=2", timeout=10).read().decode()
    rows = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
    assert rows[0]["result"]["held"] == 2 and rows[1]["end"]["state"] == "unchanged" and rows[-1]["done"] is True
    for bad in ("99", "-1", "2.5", "x"):
        with pytest.raises(HTTPError) as e:
            urlopen(live + f"/api/agent?moment=demo&k={bad}", timeout=10)
        assert e.value.code == 400
    assert json.load(urlopen(live + "/moments/demo/meta.json", timeout=10))["n_paint"] == 6
    with pytest.raises(HTTPError):
        urlopen(live + "/moments/", timeout=10)


def test_paint_turns_an_unexpected_error_into_the_saved_painting(tmp_path):
    folder = tmp_path / "moments" / "demo" / "paint"
    folder.mkdir(parents=True)
    (folder / "0002.webp").write_bytes(b"RIFF")

    def painter(w, m, k):
        raise KeyError("data")

    srv = server.make_server(port=0, web=tmp_path, painter=painter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        reply = post(f"http://127.0.0.1:{srv.server_address[1]}", {"moment": "demo", "k": 2})
    finally:
        srv.shutdown()
    assert reply["error"] and reply["saved"] == "moments/demo/paint/0002.webp"


def test_inspect_stays_on_the_held_frame_and_ends_when_it_helped(moment, monkeypatch, tmp_path):
    def painter(key, first, second, out, prompt, size=None):
        textured().save(out, "WEBP")
        return 1.5

    monkeypatch.setenv("MODEL_API_KEY", "test")
    answers = {"painting": [], "camera": [[100, 200, 140, 240]], "repaint": [[100, 200, 140, 240]]}
    only_car = fake_segment(answers)
    kit = agent.Tools(moment, segment=lambda path, concept: only_car(path, concept) if concept == "car" or "patch_" in str(path) else [], painter=painter, web=tmp_path / "none", repaint_cap=2)
    route = FakeRoute([
        ("look", {"k": 1, "view": "events"}),
        ("compare", {"concepts": ["car"]}),
        ("look", {"k": 2, "view": "events", "window_ms": 800, "x": 120, "y": 220, "r": 25}),
        ("repaint", {"k": 2, "note": "a car ahead", "concept": "car"}),
    ])
    begun = []
    trace = agent.run_agent(moment, route, tools=kit, held=2, inspect=True, max_steps=8, on_start=begun.append)
    assert "only" in trace["steps"][0]["result"]["error"]
    assert trace["steps"][-1]["tool"] == "repaint" and trace["end"]["state"] == "verified" and trace["end"]["who"] == "car"
    assert [b["start"] for b in begun] == ["look", "compare", "look", "repaint"]
    assert {t for _, names in route.seen for t in names} == {"compare", "look", "repaint", "report"}
    kit.close()


def test_inspect_aims_at_a_left_out_person_before_a_larger_car(moment, monkeypatch, tmp_path):
    def painter(key, first, second, out, prompt, size=None):
        textured().save(out, "WEBP")
        return 1.5

    def segment(path, concept):
        name = str(path)
        if "patch_" in name:
            return [record([0, 0, W - 1, H - 1])]
        if concept == "car":
            return [record([300, 150, 600, 450])] if "camera" in name else []
        if concept == "person":
            return [record([100, 200, 140, 240])] if "camera" in name or ".r" in name.replace("\\", "/").rsplit("/", 1)[-1] else []
        return []

    monkeypatch.setenv("MODEL_API_KEY", "test")
    kit = agent.Tools(moment, segment=segment, painter=painter, web=tmp_path / "none", repaint_cap=2, facing=lambda camera, painted, concept: {"bad": []})
    route = FakeRoute([
        ("compare", {"concepts": ["person", "car"]}),
        ("look", {"k": 2, "view": "events", "window_ms": 800, "x": 450, "y": 300, "r": 150}),
        ("repaint", {"k": 2, "note": "a car ahead", "concept": "car"}),
        ("look", {"k": 2, "view": "events", "window_ms": 800, "x": 120, "y": 220, "r": 25}),
        ("repaint", {"k": 2, "note": "a person walking left"}),
    ])
    trace = agent.run_agent(moment, route, tools=kit, held=2, inspect=True, max_steps=8)
    assert trace["steps"][0]["result"]["aim_first"]["who"] == "person"
    assert "people come first" in trace["steps"][2]["result"]["error"]
    assert trace["steps"][-1]["result"]["aimed_at"]["who"] == "person"
    assert trace["end"]["state"] == "verified" and trace["end"]["who"] == "person"
    kit.close()


def test_a_change_outside_the_patch_is_not_charged_to_the_repaint():
    far, here = {"box": [500, 300, 560, 360]}, {"box": [100, 100, 140, 200]}
    before = {"camera_no_painted": [here], "painted_no_camera": [], "counts": {"painted_no_camera": 0, "camera_no_painted": 1}}
    noise = {"camera_no_painted": [], "painted_no_camera": [far], "counts": {"painted_no_camera": 1, "camera_no_painted": 0}}
    inside = {"camera_no_painted": [], "painted_no_camera": [{"box": [90, 90, 150, 210]}], "counts": {"painted_no_camera": 1, "camera_no_painted": 0}}
    assert verdict.restored(before, noise)["invented_change"] == 1
    assert verdict.restored(before, noise, [90, 90, 150, 210])["invented_change"] == 0
    assert verdict.restored(before, inside, [90, 90, 150, 210])["invented_change"] == 1


def test_facing_check_withholds_a_side_on_tram_the_camera_shows_front_on(monkeypatch):
    def answer(view):
        return json.dumps({"camera_view": "toward", "camera_front_at": "left", "painted_view": view, "painted_front_at": "unclear"})
    monkeypatch.setattr(agent, "spark_ask", lambda paths, question, max_completion_tokens=None, reasoning_effort=None: answer("side"))
    assert agent.facing_check("a", "b", "tram")["bad"] == ["facing"]
    monkeypatch.setattr(agent, "spark_ask", lambda paths, question, max_completion_tokens=None, reasoning_effort=None: answer("toward"))
    assert agent.facing_check("a", "b", "tram")["bad"] == []
    monkeypatch.setattr(agent, "spark_ask", lambda paths, question, max_completion_tokens=None, reasoning_effort=None: answer("away"))
    assert agent.facing_check("a", "b", "person")["bad"] == []


def test_inspect_abstains_when_the_events_do_not_hold_it(moment, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test")
    answers = {"painting": [], "camera": [[400, 300, 460, 400]]}
    kit = agent.Tools(moment, segment=fake_segment(answers))
    route = FakeRoute([
        ("compare", {"concepts": ["car"]}),
        ("look", {"k": 2, "view": "events", "window_ms": 800, "x": 430, "y": 350, "r": 40}),
        ("report", {}),
    ])
    trace = agent.run_agent(moment, route, tools=kit, held=2, inspect=True, max_steps=8)
    assert trace["end"]["state"] == "evidence" and trace["end"]["repaints"] == 0
    kit.close()


def test_inspect_with_nothing_left_out_ends_unchanged(moment, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test")
    kit = agent.Tools(moment, segment=fake_segment({}))
    trace = agent.run_agent(moment, FakeRoute([("compare", {"concepts": ["car"]})]), tools=kit, held=2, inspect=True, max_steps=8)
    assert len(trace["steps"]) == 1 and trace["end"]["state"] == "unchanged"
    kit.close()


def held_server(tmp_path, runner, monkeypatch, deadline=30):
    monkeypatch.setattr(server, "BEAT_S", 0.2)
    monkeypatch.setattr(server, "HELD_DEADLINE_S", deadline)
    folder = tmp_path / "moments" / "demo"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"n_paint": 6}), encoding="utf-8")
    srv = server.make_server(port=0, web=tmp_path, agent_runner=runner)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/api/agent?moment=demo&k=2"


def test_held_run_beats_through_a_hung_call_and_ends_at_its_deadline(tmp_path, monkeypatch):
    hang = threading.Event()
    srv, url = held_server(tmp_path, lambda *a: hang.wait(20) and {}, monkeypatch, deadline=2)
    rows = [json.loads(line[6:]) for line in urlopen(url, timeout=10).read().decode().splitlines() if line.startswith("data: ")]
    assert sum("beat" in r for r in rows) >= 2
    assert "ran out of time" in rows[-2]["error"] and rows[-1]["done"] is True
    assert not srv.agent_caps.busy
    hang.set()
    srv.shutdown()


def test_leaving_a_held_run_frees_the_slot_and_stops_the_worker(tmp_path, monkeypatch):
    stopped = threading.Event()

    def runner(root, step, guard, should_stop, web, held):
        for _ in range(200):
            if should_stop():
                stopped.set()
                assert guard() == "The viewer left."
                return {}
            time.sleep(0.05)
        return {}

    srv, url = held_server(tmp_path, runner, monkeypatch)
    conn = urlopen(url, timeout=10)
    conn.readline()
    assert srv.agent_caps.busy
    conn.fp.raw._sock.close()
    conn.close()
    assert stopped.wait(5)
    for _ in range(100):
        if not srv.agent_caps.busy:
            break
        time.sleep(0.05)
    assert not srv.agent_caps.busy
    srv.shutdown()


def test_default_caps_allow_twenty_presses_with_a_bounded_total():
    srv = server.make_server(port=0)
    caps = srv.agent_caps
    for _ in range(20):
        assert caps.acquire("a") is None
        caps.release("a")
    assert "address" in caps.acquire("a")
    assert caps.total == 100 and srv.paint_caps.per_address >= 2 * caps.per_address
    srv.server_close()
