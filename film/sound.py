import json, re, subprocess, wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfiltfilt
from scipy.stats import pearsonr, spearmanr

FILM = Path(__file__).resolve().parent
WEB = FILM.parent / "web" / "moments"
SRC = FILM.parents[2] / "_runs" / "2026-11-03_hack_build_paint" / "moments"
FF = r"C:\Tools\ffmpeg\bin\ffmpeg.exe"
SR = 48000
W0 = 640.0
MAX_GRAINS_S = 1500.0
LUFS = -16.0
CEIL = 10 ** (-2.8 / 20)


def moment_file(mid, name):
    p = WEB / mid / name
    return p if p.exists() else SRC / mid / name


class Moment:
    def __init__(self, mid):
        self.meta = json.loads(moment_file(mid, "meta.json").read_text())
        self.counts = np.array(self.meta["counts"], dtype=float)
        self.idx = np.array(self.meta["film_index"], dtype=np.int64)
        self.ev = np.fromfile(moment_file(mid, "events_film.bin"), dtype="<u4")
        self.people = {}
        pj = moment_file(mid, "people.json")
        if pj.exists():
            for fr in json.loads(pj.read_text()).get("frames", []):
                if fr.get("people"):
                    self.people[fr["k"]] = self.mask(fr)

    @staticmethod
    def mask(fr):
        m = np.zeros(19200, dtype=bool)
        for p in fr["people"]:
            i, v = 0, 0
            for run in p.get("mask_rle", []):
                if v:
                    m[i:i + run] = True
                i += run
                v = 1 - v
        return m.reshape(120, 160)

    def bin_events(self, b):
        b = int(np.clip(b, 0, len(self.counts) - 1))
        e = self.ev[self.idx[b]:self.idx[b + 1]]
        return e & 1023, (e >> 10) & 511, (e >> 19) & 1, (e >> 20) / 4096.0


def rate_at(path, ft):
    a = 0.0
    for d, r in path:
        if ft < a + d - 1e-9:
            return r
        a += d
    return path[-1][1]


def lerp(kv, x):
    if x <= kv[0][0]:
        return kv[0][1]
    for (a, u), (b, v) in zip(kv, kv[1:]):
        if x < b:
            return u + (v - u) * (x - a) / (b - a)
    return kv[-1][1]


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def view_at(shot, ft):
    v = 0
    for c in shot.get("keys", []):
        if c["at"] <= ft < c["to"]:
            v = 1 if c["key"] == "space" else 2
    return v


def grain_bank(rng):
    bank = []
    for i in range(16):
        n = 40 + 4 * (i % 8)
        g = rng.standard_normal(n) * np.exp(-np.arange(n) / (6 + (i % 8)))
        if i % 2:
            g = np.diff(g, prepend=0)
        bank.append(g / np.max(np.abs(g)))
    return bank


def click(down):
    n = int(SR * .05)
    t = np.arange(n) / SR
    f = 1900 if down else 2600
    rng = np.random.default_rng(7 if down else 8)
    d = .5 * rng.uniform(-1, 1, n) * np.exp(-t / .0016) + .5 * np.sin(6.283 * f * t) * np.exp(-t / .004)
    late = t > .009
    d[late] += .3 * np.sin(6.283 * 320 * t[late]) * np.exp(-(t[late] - .009) / .006)
    return d


def tick():
    n = int(SR * .03)
    t = np.arange(n) / SR
    return .55 * np.sin(6.283 * 1180 * t) * np.exp(-t / .0035) + .25 * np.sin(6.283 * 540 * t) * np.exp(-t / .006)


def add(buf, at, g, left, right):
    if at < 0 or at >= buf.shape[0]:
        return
    n = min(len(g), buf.shape[0] - at)
    buf[at:at + n, 0] += g[:n] * left
    buf[at:at + n, 1] += g[:n] * right


def pan_gains(x):
    p = np.clip(x / W0 * 1.6 - .3, 0, 1)
    return np.cos(p * 1.5708), np.sin(p * 1.5708)


def synth(cuts, name):
    fps = cuts["fps"]
    shots = cuts[name]
    total = sum(s["dur"] for s in shots)
    buf = np.zeros((int(round(total * SR)) + SR // 10, 2))
    rng = np.random.default_rng(20261104)
    bank = grain_bank(rng)
    mids = sorted({s["moment"] for s in shots if s.get("kind") != "card"})
    mom = {m: Moment(m) for m in mids}
    lo = min(mom[m].counts.min() for m in mids)
    hi = max(mom[m].counts.max() for m in mids)
    level = lambda c: float((np.log(c) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    nframes = int(round(total * fps))
    hiss_env = np.zeros((nframes + 1, 2))
    rec = []
    click_d, click_u, tk = click(True), click(False), tick()
    ofs = 0.0
    for s in shots:
        n = int(round(s["dur"] * fps))
        f0 = int(round(ofs * fps))
        if s.get("kind") == "card":
            ofs += s["dur"]
            continue
        m = mom[s["moment"]]
        t = s["t0"]
        fade = s.get("fade_out")
        for i in range(n):
            ft = i / fps
            r = rate_at(s["path"], ft)
            view = view_at(s, ft)
            dev = float(sstep(.75, 4.1, lerp(s["sheet"], ft))) if s.get("sheet") else 1.0
            t1 = t + r / fps
            b = int(t * 30)
            b = min(b, len(m.counts) - 1)
            c = m.counts[b]
            v = level(c)
            g = (1 - .5 * dev) * (0.2 + .8 * v)
            g *= 0.0 if view == 1 else .12 if view == 2 else 1.0
            if r == 0:
                g *= .3
            g *= lerp(s["swell"], ft) if s.get("swell") else 1.0
            if fade:
                g *= min(1.0, max(0.0, (s["dur"] - ft) / fade))
            xs, ys, ps, us = m.bin_events(b)
            mean_x = float(xs.mean()) if len(xs) else W0 / 2
            start = int((ofs + ft) * SR)
            span = SR / fps
            want = MAX_GRAINS_S * v ** 2.2 / fps
            if g > 0 and len(xs):
                count = rng.poisson(want)
                pick = rng.integers(0, len(xs), count)
                for j in pick:
                    x = .55 * mean_x + .45 * float(xs[j])
                    L, R = pan_gains(x)
                    amp = g * (.3 + .7 * rng.random()) * .5
                    add(buf, start + int(rng.random() * span), bank[int(ps[j]) * 8 + int(rng.integers(0, 8))] * amp, L, R)
            ft_mask = ft >= s.get("mask_from", 0) and view != 1
            kk = int(t * 10)
            if ft_mask and kk in m.people and len(xs) and r > 0:
                mk = m.people[kk]
                inside = mk[np.minimum(ys // 4, 119), np.minimum(xs // 4, 159)]
                ix = np.flatnonzero(inside)
                if len(ix):
                    sx = float(xs[ix].mean())
                    L, R = pan_gains(sx)
                    L, R = (L ** 2, R ** 2)
                    frac = len(ix) / len(xs)
                    cnt = rng.poisson(frac * c / fps * .012)
                    for _ in range(cnt):
                        fr = 2300 + 400 * rng.random()
                        ln = 90
                        tt = np.arange(ln) / SR
                        gr = np.sin(6.283 * fr * tt) * np.exp(-tt / .0011)
                        add(buf, start + int(rng.random() * span), gr * .07 * (.12 if view == 2 else 1.0), L, R)
            L, R = pan_gains(mean_x)
            hiss_env[f0 + i] = (g * v ** 2 * .032 * L, g * v ** 2 * .032 * R)
            rec.append((ofs + ft, c, r, view, dev, s["id"], g))
            t = t1
        for c in s.get("keys", []):
            add(buf, int((ofs + c["at"]) * SR), click_d * .55, .9, .9)
            add(buf, int((ofs + c["to"]) * SR), click_u * .55, .9, .9)
        for c in s.get("clicks", []):
            add(buf, int((ofs + c) * SR), click_d * .55, .9, .9)
        for c in s.get("ticks", []):
            add(buf, int((ofs + c) * SR), tk * .4, .8, .8)
        ofs += s["dur"]
    noise = rng.standard_normal(buf.shape)
    noise = lfilter([1, -.92], [1], noise, axis=0)
    tt = np.arange(buf.shape[0]) / SR
    for ch in (0, 1):
        env = np.interp(tt, np.arange(nframes + 1) / fps, hiss_env[:, ch])
        buf[:, ch] += noise[:, ch] * env
    return buf, rec, total


def write_wav(path, buf):
    pcm = (np.clip(buf, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def lufs(path):
    r = subprocess.run([FF, "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r)[-1])


def check(buf, rec, total, cuts, name, out):
    win = 1.0
    rows = []
    rec = np.array([(a, b, c, d, e, g) for a, b, c, d, e, _, g in rec], dtype=float)
    for w0 in np.arange(0, total - win, win):
        q = rec[(rec[:, 0] >= w0) & (rec[:, 0] < w0 + win)]
        if len(q) < 25 or (q[:, 2] <= 0).any() or (q[:, 3] != 0).any():
            continue
        a, b = int(w0 * SR), int((w0 + win) * SR)
        rms = float(np.sqrt(np.mean(buf[a:b] ** 2)))
        rows.append((w0, q[:, 1].mean(), rms, q[:, 4].mean()))
    rows = np.array(rows)
    p = pearsonr(rows[:, 1], rows[:, 2])[0]
    s = spearmanr(rows[:, 1], rows[:, 2])[0]
    full = rows[rows[:, 3] > .99]
    pf = pearsonr(full[:, 1], full[:, 2])[0] if len(full) > 3 else float("nan")
    print(f"sound check {name}: seconds={len(rows)} pearson_counts_vs_rms={p:.3f} spearman={s:.3f} developed_only_n={len(full)} pearson={pf:.3f} lufs={lufs(out.with_suffix('.wav')):.1f} peak={np.abs(buf).max():.3f}", flush=True)
    fig, ax = plt.subplots(3, 1, figsize=(14, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    st = max(1, buf.shape[0] // 40000)
    tt = np.arange(buf.shape[0])[::st] / SR
    ax[0].plot(tt, buf[::st, 0], lw=.25, color="#1f6f8b")
    ax[0].plot(tt, buf[::st, 1] - 1.4, lw=.25, color="#b04a2f")
    ax[0].set_ylabel("L, R")
    ax[1].plot(rec[:, 0], rec[:, 1] / 1000, lw=.8, color="#333")
    ax[1].set_ylabel("events per bin (k)")
    ax[2].plot(rows[:, 0], 20 * np.log10(rows[:, 2] + 1e-6), ".", ms=3, color="#333")
    ax[2].set_ylabel("RMS dB (moving, no key)")
    ax[2].set_xlabel("film seconds")
    o = 0.0
    for sh in cuts[name]:
        for a_ in ax:
            a_.axvline(o, color="#aaa", lw=.5)
        ax[0].text(o + .2, .92, sh["id"], fontsize=7)
        o += sh["dur"]
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=90)
    plt.close(fig)


def limit(buf):
    n = int(.004 * SR) | 1
    up = np.abs(resample_poly(buf, 4, 1, axis=0)).max(axis=1)
    m = buf.shape[0]
    up = up[:m * 4 - m * 4 % 4].reshape(-1, 4).max(axis=1)
    need = np.minimum(1.0, CEIL / np.maximum(up, 1e-9))
    w = np.hanning(n + 2)[1:-1]
    gain = fftconvolve(minimum_filter1d(need, n), w / w.sum(), mode="same")
    return buf * np.minimum(gain, 1.0)[:, None]


def render(cuts, name, wav, stem):
    buf, rec, total = synth(cuts, name)
    buf = sosfiltfilt(butter(4, 15000, fs=SR, output="sos"), buf, axis=0)
    tmp = Path(wav).with_suffix(".tmp.wav")
    for _ in range(3):
        write_wav(tmp, np.clip(buf, -1, 1))
        buf = limit(buf * 10 ** ((LUFS - lufs(tmp)) / 20))
    tmp.unlink()
    write_wav(wav, buf)
    stem = Path(stem)
    write_wav(stem.with_suffix(".wav"), buf)
    check(buf, rec, total, cuts, name, stem)
    stem.with_suffix(".wav").unlink() if stem.with_suffix(".wav") != Path(wav) else None


if __name__ == "__main__":
    import sys
    c = json.loads((FILM / "cuts.json").read_text(encoding="utf-8"))
    n = sys.argv[1] if len(sys.argv) > 1 else "film"
    run = FILM.parents[2] / "_runs" / "2026-11-04_hack_film"
    run.mkdir(parents=True, exist_ok=True)
    render(c, n, run / f"afterdark_{n}.wav", run / f"sound_{n}")
