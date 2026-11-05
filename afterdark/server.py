import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import select
import socket
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse

from . import meta, spend

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["muse-image-1.0", "muse-spark-1.3", "sam-3.1"]
MOMENT_ID = re.compile(r"^[a-z0-9_-]{1,40}$")
OUT_OF_CREDIT = "The painter is out of credit for now, so this is the saved painting."
UNREACHABLE = "Could not reach the painter, so this is the saved painting."
BEAT_S = 5
HELD_DEADLINE_S = 145
SNIFF = {b"RIFF": "image/webp", b"\x89PNG": "image/png", b"\xff\xd8\xff": "image/jpeg"}


class Caps:
    def __init__(self, per_address=30, total=200, clock=time.time):
        self.per_address, self.total, self.clock = per_address, total, clock
        self.lock = threading.Lock()
        self.day = None
        self.counts = {}
        self.sum = 0
        self.busy = set()

    def _roll(self):
        now = time.gmtime(self.clock())
        day = (now.tm_year, now.tm_yday)
        if day != self.day:
            self.day, self.counts, self.sum = day, {}, 0

    def _refuse(self, address):
        if self.sum >= self.total:
            return "Today's limit for live paintings is used up."
        if self.counts.get(address, 0) >= self.per_address:
            return "This address has used its live paintings for today."
        return None

    def _charge(self, address):
        self.counts[address] = self.counts.get(address, 0) + 1
        self.sum += 1

    def acquire(self, address, charge=True):
        with self.lock:
            self._roll()
            if address in self.busy:
                return "One at a time: the last one is still running."
            refused = self._refuse(address)
            if refused:
                return refused
            if charge:
                self._charge(address)
            self.busy.add(address)
            return None

    def charge(self, address):
        with self.lock:
            self._roll()
            refused = self._refuse(address)
            if refused:
                return refused
            self._charge(address)
            return None

    def release(self, address):
        with self.lock:
            self.busy.discard(address)


def pipeline():
    folder = str(ROOT / "pipeline")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import paint_moment
    from env import load_key
    return paint_moment, load_key


def have_key():
    try:
        return bool(pipeline()[1]())
    except (SystemExit, OSError):
        return False


def sniff(data):
    for head, mime in SNIFF.items():
        if data.startswith(head):
            return mime
    return "image/webp"


def source_root():
    value = os.environ.get("AFTERDARK_SOURCE")
    return Path(value) if value else None


def input_path(web, moment, k):
    options = [web / "moments" / moment / "ev10" / f"{k:04d}.png"]
    if source_root():
        options.append(source_root() / moment / "ev10" / f"{k:04d}.png")
    return next((p for p in options if p.exists()), None)


def style_path(web):
    options = []
    if os.environ.get("AFTERDARK_STYLE"):
        options.append(Path(os.environ["AFTERDARK_STYLE"]))
    options.append(web / "moments" / "style.png")
    if source_root():
        options.append(source_root() / "style.png")
    return next((p for p in options if p.exists()), None)


def moment_root(web, moment):
    if source_root() and (source_root() / moment / "meta.json").exists():
        return source_root() / moment
    return web / "moments" / moment


def log_paint_attempt(attempt, status, seconds):
    print(json.dumps({"event": "paint_attempt", "attempt": attempt, "status": status, "seconds": seconds}), file=sys.stderr, flush=True)


def paint_frame(web, moment, k, before_attempt=None):
    paint_moment, load_key = pipeline()
    event, style = input_path(web, moment, k), style_path(web)
    if event is None or style is None:
        raise FileNotFoundError
    with tempfile.TemporaryDirectory(prefix="afterdark_") as tmp:
        out = Path(tmp) / "paint.webp"
        seconds = paint_moment.paint(load_key(), event, style, out, before_attempt=before_attempt, on_attempt=log_paint_attempt)
        if not isinstance(seconds, float):
            raise ConnectionError(str(seconds))
        data = out.read_bytes()
    return {"image": f"data:{sniff(data)};base64," + base64.b64encode(data).decode(), "seconds": seconds, "model": "muse-image-1.0"}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".webp": "image/webp", ".json": "application/json", ".bin": "application/octet-stream", ".js": "text/javascript", ".wasm": "application/wasm"}
    protocol_version = "HTTP/1.1"
    server_version = "afterdark"
    sys_version = ""

    def log_message(self, *args):
        pass

    def list_directory(self, path):
        self.send_error(404)
        return None

    def address(self):
        return self.client_address[0]

    def reply(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            return self.status()
        if url.path == "/api/agent":
            query = parse_qs(url.query)
            return self.agent(query.get("moment", [""])[0], query.get("k", [None])[0])
        if url.path.startswith("/api/"):
            return self.reply({"error": "No such route."}, 404)
        if any(part.startswith(".") for part in url.path.split("/")):
            return self.send_error(404)
        return super().do_GET()

    def status(self):
        live = have_key()
        try:
            spend.check()
        except spend.SpendStop:
            live = False
        self.reply({"live": live, "models": MODELS})

    def do_POST(self):
        if urlparse(self.path).path != "/api/paint":
            return self.reply({"error": "No such route."}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > 4096:
            self.close_connection = True
            return self.reply({"error": "That request is not one this server takes."}, 400)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            moment, k = body["moment"], body["k"]
            if not isinstance(moment, str) or not MOMENT_ID.fullmatch(moment) or type(k) is not int or not 0 <= k <= 9999:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return self.reply({"error": "Send a moment and a frame number."}, 400)
        self.paint(moment, k)

    def paint(self, moment, k):
        web = self.server.web
        saved = web / "moments" / moment / "paint" / f"{k:04d}.webp"
        if k < 0 or not saved.exists():
            return self.reply({"error": "Only frames shipped with the page can be painted."}, 404)
        fallback = {"saved": f"moments/{moment}/paint/{k:04d}.webp"}
        address = self.address()
        default_painter = self.server.painter is paint_frame
        refused = self.server.paint_caps.acquire(address, charge=not default_painter)
        if refused:
            return self.reply({"error": refused, **fallback}, 429)
        try:
            if default_painter:
                result = self.server.painter(web, moment, k, before_attempt=lambda: self.server.paint_caps.charge(address))
            else:
                result = self.server.painter(web, moment, k)
            self.reply(result)
        except spend.SpendStop:
            self.reply({"error": OUT_OF_CREDIT, **fallback})
        except ConnectionError as e:
            line = OUT_OF_CREDIT if re.search(r"HTTP 40[23]|credit|quota|insufficient", str(e), re.I) else UNREACHABLE
            self.reply({"error": line, **fallback})
        except (FileNotFoundError, SystemExit, OSError):
            self.reply({"error": UNREACHABLE, **fallback})
        except Exception:
            self.reply({"error": UNREACHABLE, **fallback})
        finally:
            self.server.paint_caps.release(address)

    def sse(self, body):
        self.wfile.write(b"data: " + json.dumps(body).encode() + b"\n\n")
        self.wfile.flush()

    def agent(self, moment, k=None):
        web = self.server.web
        if not MOMENT_ID.fullmatch(moment) or not (web / "moments" / moment / "meta.json").exists():
            return self.reply({"error": "No such moment."}, 404)
        held = None
        if k is not None:
            if not re.fullmatch(r"[0-9]{1,4}", k) or int(k) >= json.loads((web / "moments" / moment / "meta.json").read_text(encoding="utf-8"))["n_paint"]:
                return self.reply({"error": "Send a frame number from this drive."}, 400)
            held = int(k)
        address = self.address()
        refused = self.server.agent_caps.acquire(address)
        if refused:
            return self.reply({"error": refused}, 429)
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        left = threading.Event()
        outbox = queue.Queue()
        ending = {"done": True}
        if (web / "moments" / moment / "agent.json").exists():
            ending["recorded"] = f"moments/{moment}/agent.json"

        def guard():
            return "The viewer left." if left.is_set() else self.server.paint_caps.charge(address)

        def work():
            try:
                root = web / "moments" / moment if held is not None else moment_root(web, moment)
                trace = self.server.agent_runner(root, outbox.put, guard, left.is_set, web, held)
                if trace.get("end") and not trace.get("stopped"):
                    outbox.put({"end": trace["end"], "seconds": trace.get("seconds"), "model": trace.get("model")})
                if trace.get("stopped") and trace["stopped"] != "viewer left":
                    outbox.put({"error": "The live agent stopped early, so the recorded run stands."})
            except spend.SpendStop:
                outbox.put({"error": "The live agent is out of credit for now, so the recorded run stands."})
            except Exception:
                outbox.put({"error": "Could not run the live agent, so the recorded run stands."})
            finally:
                outbox.put(ending)

        threading.Thread(target=work, daemon=True).start()
        started = beat = time.monotonic()
        try:
            while not self.viewer_gone():
                try:
                    body = outbox.get(timeout=0.5)
                except queue.Empty:
                    body = None
                now = time.monotonic()
                if body is not None:
                    self.sse(body)
                    if body is ending:
                        break
                if held is None:
                    continue
                if now - started > HELD_DEADLINE_S:
                    self.sse({"error": "The live check ran out of time, so the painting stands."})
                    self.sse(ending)
                    break
                if now - beat >= BEAT_S:
                    beat = now
                    self.sse({"beat": round(now - started)})
        except OSError:
            pass
        finally:
            left.set()
            self.server.agent_caps.release(address)

    def viewer_gone(self):
        try:
            ready = select.select([self.connection], [], [], 0)[0]
            return bool(ready) and not self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            return True


def run_live_agent(root, step, guard, should_stop, web, held=None):
    from .agent import INSPECT_REPAINTS, INSPECT_STEPS, Tools, facing_check, get_route, live_image, rules_check, run_agent
    from .moment import Moment
    paint_moment, _ = pipeline()
    if held is not None:
        meta.REQUEST_TIMEOUT = 30.0
    bounds = {"timeout": 50, "attempts": 2} if held is not None else {}
    painter = lambda *args, **kwargs: paint_moment.paint(*args, before_attempt=guard, on_attempt=log_paint_attempt, **bounds, **kwargs)
    kit = Tools(Moment(root), live=True, web=web, painter=painter, repaint_cap=INSPECT_REPAINTS if held is not None else None, rules=rules_check if held is not None else None, facing=facing_check if held is not None else None)

    def send(body):
        result = body.get("result") or {}
        version = result.get("version") if body.get("tool") == "repaint" and not result.get("withheld") else None
        step({**body, "image": live_image(kit.versions[(held, version)])} if version else body)

    try:
        if held is None:
            return run_agent(root, get_route("spark"), on_step=step, live=True, web=web, should_stop=should_stop, tools=kit)
        route = get_route("spark")
        route.extra = {"reasoning_effort": "low"}
        route.timeout = 40
        return run_agent(root, route, on_step=send, on_start=step, live=True, web=web, should_stop=should_stop, tools=kit, held=held, inspect=True, max_steps=INSPECT_STEPS)
    finally:
        kit.close()


def make_server(host="127.0.0.1", port=8080, web=None, painter=paint_frame, agent_runner=run_live_agent, paint_caps=None, agent_caps=None):
    web = Path(web) if web else ROOT / "web"

    class Bound(Handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web), **kwargs)

    server = ThreadingHTTPServer((host, port), Bound)
    server.daemon_threads = True
    server.web = web
    server.painter = painter
    server.agent_runner = agent_runner
    server.paint_caps = paint_caps or Caps(40, 200)
    server.agent_caps = agent_caps or Caps(20, 100)
    return server


def serve(host="127.0.0.1", port=8080, web=None):
    server = make_server(host, port, web)
    print(f"Afterdark on http://{host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
