import argparse, base64, hashlib, json, re, subprocess, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import websocket

import sound

FILM = Path(__file__).resolve().parent
WEB = FILM.parent / "web"
ROOT = FILM.parents[2]
RUN = ROOT / "_runs" / "2026-11-04_hack_film"
SRC = ROOT / "_runs" / "2026-11-03_hack_build_paint" / "moments"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
FF = r"C:\Tools\ffmpeg\bin\ffmpeg.exe"
SQUARE_X = 100
SQUARE_CSS = f"""
#caption{{left:{SQUARE_X + 40}px;right:{1920 - SQUARE_X - 1080 + 40}px;max-width:1000px;bottom:112px}}
#low{{left:{SQUARE_X + 40}px;right:{1920 - SQUARE_X - 1080 + 40}px;bottom:22px;flex-wrap:wrap}}
#stamp{{font-size:22px}}
#keycap{{left:{SQUARE_X + 40}px;bottom:238px}}
"""


class H(SimpleHTTPRequestHandler):
    meta = {}

    def translate_path(self, path):
        p = path.split("?")[0].lstrip("/")
        f = WEB / p
        if f.is_file():
            return str(f)
        parts = p.split("/")
        if len(parts) >= 3 and parts[0] == "moments":
            mid, rest = parts[1], parts[2:]
            s = SRC / mid
            if rest == ["meta.json"]:
                return str(self.meta_for(mid))
            if rest[0] == "paint":
                return str(s / "stab" / rest[1])
            if rest[0] == "cam":
                return str(s / "cam_pub" / rest[1])
            if rest[0] == "backing":
                return str(s / "backing" / rest[1].replace(".webp", ".png"))
            return str(s / "/".join(rest))
        return str(f)

    def meta_for(self, mid):
        out = RUN / "overlay_meta"
        out.mkdir(parents=True, exist_ok=True)
        q = out / f"{mid}.json"
        m = json.loads((SRC / mid / "meta.json").read_text())
        m.update({"place": "Zurich", "title": mid.title(), "credit": "DSEC dataset, CC BY-SA 4.0", "n_cam": m["n_paint"], "n_paint_pub": m["n_paint"]})
        q.write_text(json.dumps(m))
        return q

    def log_message(self, *a):
        pass


def serve():
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s.server_address[1]


class Cdp:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, max_size=None, timeout=300)
        self.i = 0

    def call(self, m, **p):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": m, "params": p}))
        while True:
            r = json.loads(self.ws.recv())
            if r.get("id") == self.i:
                if "error" in r:
                    raise RuntimeError(r["error"])
                return r["result"]

    def ev(self, js):
        r = self.call("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:900])
        return r["result"].get("value")


def encoder(path, fps, w, h, crf, preset):
    return subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps), "-i", "-",
                             "-vf", f"scale={w}:{h}:in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p",
                             "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-x264-params", "aq-mode=3:aq-strength=1.0:deblock=-1,-1:keyint=60",
                             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-an", str(path)], stdin=subprocess.PIPE)


def render_shot(shot, fps, port, cport, out, square, crf, preset, limit):
    driver = (FILM / "driver.js").read_text(encoding="utf-8")
    prof = tempfile.mkdtemp(prefix="chr_")
    ch = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={cport}", f"--user-data-dir={prof}", "--use-angle=d3d11", "--enable-webgl", "--ignore-gpu-blocklist", "--force_high_performance_gpu", "--no-first-run", "--hide-scrollbars", "--window-size=1920,1080", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{cport}/json"))
                t = [x for x in tabs if x["type"] == "page"][0]
                break
            except Exception:
                time.sleep(.5)
        c = Cdp(t["webSocketDebuggerUrl"])
        c.call("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
        c.call("Page.enable")
        c.call("Runtime.enable")
        mid = shot.get("moment", "crosswalk")
        c.call("Page.navigate", url=f"http://127.0.0.1:{port}/v5.html?render=1&moment={mid}&cap=film&t={shot.get('t0', 0)}&exposed=1")
        for _ in range(150):
            try:
                if c.ev("typeof window.__ready") == "object":
                    break
            except Exception:
                pass
            time.sleep(.2)
        c.ev(driver)
        extra = SQUARE_CSS if square else ""
        info = c.ev(f"window.__film.prep({json.dumps(shot)},{json.dumps(extra)}).then(r=>JSON.stringify(r))")
        n = int(round(shot["dur"] * fps))
        if limit:
            n = min(n, limit)
        clip = {"x": SQUARE_X if square else 0, "y": 0, "width": 1080 if square else 1920, "height": 1080, "scale": 1}
        enc = encoder(out, fps, clip["width"], 1080, crf, preset)
        static = None
        r = None
        flat = shot.get("kind") in ("card", "still")
        poster = int(round(shot.get("poster", 0) * fps))
        for i in range(n):
            if i and i <= poster:
                enc.stdin.write(static)
                continue
            r = c.ev(f"window.__film.step({i}).then(r=>JSON.stringify(r))")
            if flat and static is not None and not json.loads(r).get("dirty"):
                enc.stdin.write(static)
                continue
            s = c.call("Page.captureScreenshot", format="jpeg", quality=95, clip=clip)
            static = base64.b64decode(s["data"])
            enc.stdin.write(static)
        enc.stdin.close()
        enc.wait()
        print(shot["id"], n, "frames", round(time.time() - t0, 1), "s", info, r, flush=True)
    finally:
        ch.terminate()


def render_clip(shot, fps, out, crf, preset):
    src = ROOT / shot["clip"]
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(shot["from"]), "-t", str(shot["dur"]), "-i", str(src), "-vf", f"fps={fps},scale=1920:1080,format=yuv420p",
                    "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-x264-params", "aq-mode=3:aq-strength=1.0:deblock=-1,-1:keyint=60",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-an", str(out)], check=True)
    print(shot["id"], "clip", shot["from"], shot["dur"], flush=True)


def key_for(shot, fps, square, crf):
    h = hashlib.sha1()
    h.update(json.dumps(shot, sort_keys=True).encode())
    h.update(str((fps, square, crf)).encode())
    h.update((FILM / "driver.js").read_bytes())
    if shot.get("clip"):
        h.update(str((ROOT / shot["clip"]).stat().st_mtime).encode())
        return h.hexdigest()[:12]
    h.update((WEB / "app.js").read_bytes())
    h.update((WEB / "fx.js").read_bytes())
    h.update((WEB / "style.css").read_bytes())
    if shot.get("kind") != "card" and shot.get("moment"):
        m = WEB / "moments" / shot["moment"] / "meta.json"
        h.update(str(m.stat().st_mtime if m.exists() else 0).encode())
    return h.hexdigest()[:12]


def concat(parts, out):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    lst.unlink()


def seam(video, out, fps, dur, crf):
    still = out.with_name(out.stem + "_first.png")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(video), "-frames:v", "1", "-update", "1", str(still)], check=True)
    xf = 0.5
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(video), "-loop", "1", "-framerate", str(fps), "-t", str(xf + 0.2), "-i", str(still),
                    "-filter_complex", f"[1:v]format=yuv420p,fps={fps},settb=1/{fps}[s];[0:v]fps={fps},settb=1/{fps}[a];[a][s]xfade=transition=fade:duration={xf}:offset={dur - xf},format=yuv420p,trim=duration={dur}[v]",
                    "-map", "[v]", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-x264-params", "aq-mode=3:deblock=-1,-1",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-an", str(out)], check=True)
    still.unlink()


def mux(video, wav, out):
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(out)], check=True)


def ship(video, wav, out, mb):
    for crf in (19, 21, 23, 25):
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                        "-x264-params", "aq-mode=3:aq-strength=1.1:deblock=-1,-1:keyint=60:psy-rd=1.0,0.15", "-pix_fmt", "yuv420p",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
        print("ship crf", crf, round(out.stat().st_size / 1e6, 1), "MB", flush=True)
        if out.stat().st_size < mb * 1e6:
            return
    raise SystemExit(f"over {mb} MB at crf 25")


def build(name, a, cuts):
    shots = cuts[name]
    if a.only:
        shots = [s for s in shots if s["id"] in a.only.split(",")]
    fps = cuts["fps"]
    tag = name + ("_sq" if a.square else "")
    d = RUN / "shots" / tag
    d.mkdir(parents=True, exist_ok=True)
    port = serve()
    jobs, parts = [], []
    for i, s in enumerate(shots):
        k = key_for(s, fps, a.square, a.crf)
        p = d / f"{s['id']}_{k}{'_lim' + str(a.limit) if a.limit else ''}.mp4"
        parts.append(p)
        for old in d.glob(f"{s['id']}_*.mp4"):
            if old != p:
                old.unlink()
        if not p.exists():
            jobs.append((s, p))
    print(tag, "shots", len(shots), "to render", len(jobs), flush=True)

    def work(ix_job):
        ix, (s, p) = ix_job
        tmp = p.with_name(p.stem + ".tmp.mp4")
        if s.get("clip"):
            render_clip(s, fps, tmp, a.crf, a.preset)
        else:
            render_shot(s, fps, port, 9340 + ix, tmp, a.square, a.crf, a.preset, a.limit)
        tmp.replace(p)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(work, enumerate(jobs)))
    if a.only or a.limit:
        print("partial, shots in", d)
        return None
    silent = RUN / f"_{tag}_v.mp4"
    concat(parts, silent)
    if name == "loop":
        s2 = RUN / f"_{tag}_seam.mp4"
        seam(silent, s2, fps, sum(s["dur"] for s in shots), a.crf)
        silent.unlink()
        silent = s2
    wav = RUN / f"afterdark_{name}.wav"
    if not wav.exists() or a.sound or wav.stat().st_mtime < (FILM / "cuts.json").stat().st_mtime or wav.stat().st_mtime < (FILM / "sound.py").stat().st_mtime:
        sound.render(cuts, name, wav, RUN / f"sound_{name}")
    final = RUN / {"film": "afterdark_pass2.mp4", "film_sq": "afterdark_pass2_sq.mp4", "loop": "afterdark_loop_1920.mp4", "loop_sq": "afterdark_loop_1080.mp4"}[tag]
    if name == "film":
        ship(silent, wav, final, a.mb)
    else:
        mux(silent, wav, final)
    silent.unlink()
    print("wrote", final, round(final.stat().st_size / 1e6, 1), "MB", flush=True)
    return final


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cut", default="film", choices=["film", "loop"])
    a.add_argument("--only", default="")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--jobs", type=int, default=2)
    a.add_argument("--crf", type=int, default=17)
    a.add_argument("--preset", default="fast")
    a.add_argument("--square", action="store_true")
    a.add_argument("--sound", action="store_true")
    a.add_argument("--mb", type=float, default=195)
    a = a.parse_args()
    cuts = json.loads((FILM / "cuts.json").read_text(encoding="utf-8"))
    RUN.mkdir(parents=True, exist_ok=True)
    build(a.cut, a, cuts)


if __name__ == "__main__":
    main()
