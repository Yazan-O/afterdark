'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const COL = { cam: [216, 180, 106], ev: [134, 191, 178] };
const EV_TINT = '#BAC7CE';
const FADE_MS = 700, MASK_TAU_MS = 110, EV_TAU_MS = 70, MIX_TAU_MS = 320, COUNT_MS = 250, MW = 160, MH = 120;
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

const S = { id: null, pack: null, img: { cam: [], ev: [] }, t: 0, playing: !REDUCED, n: -1, layout: 'split', mix: 0.5, mixTo: 0.5, token: 0, last: 0, countAt: 0, dur: 0 };

const evBuf = document.createElement('canvas');
evBuf.width = 640; evBuf.height = 480;
const evG = evBuf.getContext('2d');

function decode(runs, into) {
  let pos = 0, on = 0;
  for (const r of runs) {
    if (on) into.fill(1, pos, Math.min(pos + r, into.length));
    pos += r; on ^= 1;
  }
}

class Pane {
  constructor(el, kinds) {
    this.el = el; this.c = el.querySelector('canvas'); this.g = this.c.getContext('2d');
    this.kinds = kinds;
    this.prev = document.createElement('canvas'); this.pg = this.prev.getContext('2d');
    this.fadeAt = -1e9;
    this.acc = {}; this.tgt = {};
    for (const k of kinds) { this.acc[k] = new Float32Array(MW * MH); this.tgt[k] = new Float32Array(MW * MH); }
    this.m = document.createElement('canvas'); this.m.width = MW; this.m.height = MH;
    this.mg = this.m.getContext('2d'); this.data = this.mg.createImageData(MW, MH);
  }
  visible() { return this.c.offsetParent !== null; }
  size() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(this.c.clientWidth * dpr), h = Math.round(w * 3 / 4);
    if (w && (this.c.width !== w || this.c.height !== h)) { this.c.width = w; this.c.height = h; this.prev.width = w; this.prev.height = h; return true; }
    return false;
  }
  snapshot(now) {
    if (!this.c.width) return;
    this.pg.clearRect(0, 0, this.prev.width, this.prev.height);
    this.pg.drawImage(this.c, 0, 0);
    this.fadeAt = now;
  }
  target(n, snap) {
    for (const k of this.kinds) {
      const t = this.tgt[k]; t.fill(0);
      for (const p of (S.pack.people[k][n] || [])) decode(p.mask_rle, t);
      if (snap) this.acc[k].set(t);
    }
  }
  ease(dt) {
    const a = 1 - Math.exp(-dt / MASK_TAU_MS);
    let moving = false;
    for (const k of this.kinds) {
      const acc = this.acc[k], t = this.tgt[k];
      for (let i = 0; i < acc.length; i++) {
        const d = t[i] - acc[i];
        if (d > 0.002 || d < -0.002) { acc[i] += d * a; moving = true; } else acc[i] = t[i];
      }
    }
    return moving;
  }
  masks() {
    const g = this.g, W = this.c.width, H = this.c.height, px = this.data.data;
    for (const k of this.kinds) {
      const acc = this.acc[k], [r, gg, b] = COL[k];
      for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) {
        const i = y * MW + x, v = acc[i], o = i * 4;
        let lo = v;
        if (v > 0) {
          lo = Math.min(x > 0 ? acc[i - 1] : 0, x < MW - 1 ? acc[i + 1] : 0, y > 0 ? acc[i - MW] : 0, y < MH - 1 ? acc[i + MW] : 0);
        }
        const edge = Math.max(0, v - lo);
        px[o] = r; px[o + 1] = gg; px[o + 2] = b;
        px[o + 3] = Math.min(0.92, 0.26 * v + 1.15 * edge) * 255;
      }
      this.mg.putImageData(this.data, 0, 0);
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.filter = `blur(${(W / 640 * 1.6).toFixed(2)}px)`;
      g.drawImage(this.m, 0, 0, W, H);
      g.filter = 'none';
    }
  }
  fade(now) {
    const f = 1 - (now - this.fadeAt) / FADE_MS;
    if (f <= 0) return false;
    const g = this.g;
    g.globalAlpha = f * f * (3 - 2 * f);
    g.drawImage(this.prev, 0, 0);
    g.globalAlpha = 1;
    return true;
  }
}

function tintedEvents(n) {
  const im = S.img.ev[n];
  evG.globalCompositeOperation = 'source-over';
  evG.fillStyle = '#000'; evG.fillRect(0, 0, 640, 480);
  if (im && im.complete && im.naturalWidth) evG.drawImage(im, 0, 0);
  evG.globalCompositeOperation = 'multiply';
  evG.fillStyle = EV_TINT; evG.fillRect(0, 0, 640, 480);
  evG.globalCompositeOperation = 'source-over';
  return evBuf;
}

function smoothEvents(pane, n, dt, jump) {
  if (!pane.ev) { pane.ev = document.createElement('canvas'); pane.ev.width = 640; pane.ev.height = 480; pane.eg = pane.ev.getContext('2d'); jump = true; }
  pane.eg.globalAlpha = jump ? 1 : 1 - Math.exp(-dt / EV_TAU_MS);
  pane.eg.drawImage(tintedEvents(n), 0, 0);
  pane.eg.globalAlpha = 1;
  return pane.ev;
}

function drawImages(pane, n, dt, jump) {
  const g = pane.g, W = pane.c.width, H = pane.c.height;
  g.fillStyle = '#0A0B0C'; g.fillRect(0, 0, W, H);
  const cam = S.img.cam[n];
  const camOk = cam && cam.complete && cam.naturalWidth;
  if (pane.kind === 'cam' && camOk) g.drawImage(cam, 0, 0, W, H);
  if (pane.kind === 'ev') g.drawImage(smoothEvents(pane, n, dt, jump), 0, 0, W, H);
  if (pane.kind === 'mix') {
    const ev = smoothEvents(pane, n, dt, jump);
    if (camOk) g.drawImage(cam, 0, 0, W, H);
    g.globalAlpha = S.mix; g.drawImage(ev, 0, 0, W, H); g.globalAlpha = 1;
  }
}

const panes = {};
for (const el of $$('.pane')) {
  const kind = el.dataset.pane;
  panes[kind] = new Pane(el, kind === 'mix' ? ['cam', 'ev'] : [kind]);
  panes[kind].kind = kind;
}
const shown = () => Object.values(panes).filter(p => p.visible());

function frameAt(t) {
  const ts = S.pack.t_us, us = t * 1e6;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (ts[mid] <= us) lo = mid; else hi = mid - 1; }
  return lo;
}

const TL = { c: $('#timeline') };
TL.g = TL.c.getContext('2d');
function drawTimeline() {
  if (!S.pack) return;
  const dpr = Math.min(devicePixelRatio || 1, 2), c = TL.c;
  const W = Math.round(c.clientWidth * dpr), H = Math.round(c.clientHeight * dpr);
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const g = TL.g, n = S.pack.n, P = S.pack.people;
  g.clearRect(0, 0, W, H);
  const max = Math.max(1, ...P.cam.map(f => f.length), ...P.ev.map(f => f.length));
  const half = H / 2, bw = W / n;
  g.fillStyle = 'rgba(232,230,225,0.05)'; g.fillRect(0, 0, W, half - dpr); g.fillRect(0, half + dpr, W, half - dpr);
  for (let i = 0; i < n; i++) {
    const x = i * bw;
    const a = P.cam[i].length / max, b = P.ev[i].length / max;
    if (a) { g.fillStyle = 'rgba(216,180,106,0.75)'; g.fillRect(x, half - dpr - a * (half - 4 * dpr), Math.max(1, bw - 0.5), a * (half - 4 * dpr)); }
    if (b) { g.fillStyle = 'rgba(134,191,178,0.75)'; g.fillRect(x, half + dpr, Math.max(1, bw - 0.5), b * (half - 4 * dpr)); }
  }
  const px = (S.t / S.dur) * W;
  g.fillStyle = '#E8E6E1'; g.fillRect(Math.round(px) - dpr, 0, 2 * dpr, H);
}

function clock() {
  $('#clock').textContent = `${S.t.toFixed(2)} s / ${S.dur.toFixed(2)} s`;
}

function seek(t, now) {
  const before = S.n;
  S.t = Math.max(0, Math.min(S.dur, t));
  const n = frameAt(S.t);
  if (Math.abs(n - before) > 2) for (const p of shown()) p.snapshot(now);
  render(now, Math.abs(n - before) > 2, true);
}

let fading = false, easing = false;
function render(now, jump, force) {
  const n = frameAt(S.t), changed = n !== S.n;
  const dt = Math.min(100, now - S.last || 16); S.last = now;
  let resized = false;
  for (const p of shown()) resized = p.size() || resized;
  if (changed) { for (const p of Object.values(panes)) p.target(n, jump); S.n = n; S.changedAt = now; }
  easing = now - (S.changedAt || 0) < EV_TAU_MS * 5;
  for (const p of shown()) easing = p.ease(dt) || easing;
  const dm = S.mixTo - S.mix;
  if (Math.abs(dm) > 0.002) { S.mix += dm * (1 - Math.exp(-dt / MIX_TAU_MS)); easing = true; } else S.mix = S.mixTo;
  if (!(changed || easing || fading || resized || force)) return;
  fading = false;
  for (const p of shown()) {
    drawImages(p, n, dt, jump);
    p.masks();
    fading = p.fade(now) || fading;
  }
  drawTimeline(); clock();
}

function tick(now) {
  if (S.pack && S.playing) {
    const dt = Math.min(100, now - (S.playAt || now)) / 1000;
    S.playAt = now;
    let t = S.t + dt;
    if (t >= S.dur) {
      for (const p of shown()) p.snapshot(now);
      S.t = 0; S.n = -1;
      render(now, true, true);
    } else { S.t = t; render(now, false, false); }
  } else if (S.pack) { S.playAt = now; render(now, false, false); }
  requestAnimationFrame(tick);
}

function loadImage(src) {
  return new Promise(res => { const im = new Image(); im.decoding = 'async'; im.onload = () => res(im); im.onerror = () => res(im); im.src = src; });
}

async function loadClip(id) {
  const tok = ++S.token;
  const pack = await (await fetch(`v6/${id}/pack.json`)).json();
  if (tok !== S.token) return;
  const pad = i => String(i).padStart(4, '0');
  const cam = [], ev = [];
  const first = 24;
  const jobs = [];
  for (let i = 0; i < pack.n; i++) {
    jobs.push(loadImage(`v6/${id}/cam/${pad(i)}.webp`).then(im => { cam[i] = im; }));
    jobs.push(loadImage(`v6/${id}/ev/${pad(i)}.webp`).then(im => { ev[i] = im; }));
  }
  $('#clock').textContent = 'loading';
  await Promise.all(jobs.slice(0, first * 2));
  await Promise.all(jobs);
  if (tok !== S.token) return;
  const now = performance.now();
  if (S.pack) for (const p of shown()) p.snapshot(now);
  S.id = id; S.pack = pack; S.img = { cam, ev };
  S.dur = (pack.t_us[pack.n - 1] + 50000) / 1e6;
  S.t = 0; S.n = -1;
  render(now, true, true);
}

$$('[data-layout]').forEach(b => b.addEventListener('click', () => {
  const to = b.dataset.layout;
  if (to === S.layout) return;
  $$('[data-layout]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.layout === to)));
  const stage = $('#stage');
  stage.classList.add('dim');
  setTimeout(() => {
    S.layout = to; stage.dataset.layout = to;
    render(performance.now(), true, true);
    stage.classList.remove('dim');
  }, REDUCED ? 0 : 350);
}));

$('#mix').addEventListener('input', e => { S.mixTo = e.target.value / 100; });

function setPlaying(on) {
  S.playing = on;
  const b = $('#play');
  b.textContent = on ? 'Pause' : 'Play';
  b.setAttribute('aria-label', on ? 'Pause' : 'Play');
}
$('#play').addEventListener('click', () => setPlaying(!S.playing));

(() => {
  let drag = false, was = false;
  const at = e => { const r = TL.c.getBoundingClientRect(); return ((e.clientX - r.left) / r.width) * S.dur; };
  TL.c.addEventListener('pointerdown', e => { if (!S.pack) return; drag = true; was = S.playing; setPlaying(false); TL.c.setPointerCapture(e.pointerId); seek(at(e), performance.now()); });
  TL.c.addEventListener('pointermove', e => { if (drag) seek(at(e), performance.now()); });
  const end = () => { if (drag) { drag = false; setPlaying(was); } };
  TL.c.addEventListener('pointerup', end); TL.c.addEventListener('pointercancel', end);
})();

document.addEventListener('keydown', e => {
  if (!S.pack || e.target.closest('input,button')) return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!S.playing); }
  if (e.code === 'ArrowRight' || e.code === 'ArrowLeft') {
    e.preventDefault(); setPlaying(false);
    const n = Math.max(0, Math.min(S.pack.n - 1, S.n + (e.code === 'ArrowRight' ? 1 : -1)));
    seek(S.pack.t_us[n] / 1e6 + 0.001, performance.now());
  }
});

addEventListener('resize', () => { if (S.pack) render(performance.now(), false, true); });

function drawStill(c, img, people, kind, grid) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const W = Math.round(c.clientWidth * dpr), H = Math.round(W * 3 / 4);
  if (!W) return;
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.fillStyle = '#0A0B0C'; g.fillRect(0, 0, W, H);
  g.drawImage(img, 0, 0, W, H);
  if (kind === 'ev') {
    g.globalCompositeOperation = 'multiply'; g.fillStyle = EV_TINT; g.fillRect(0, 0, W, H);
    g.globalCompositeOperation = 'source-over';
  }
  const [gw, gh] = grid, m = new Float32Array(gw * gh);
  for (const p of people) decode(p.mask_rle, m);
  const mc = document.createElement('canvas'); mc.width = gw; mc.height = gh;
  const mg = mc.getContext('2d'), data = mg.createImageData(gw, gh), px = data.data, [r, gg, b] = COL[kind];
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const i = y * gw + x, v = m[i], o = i * 4;
    const lo = v ? Math.min(x > 0 ? m[i - 1] : 0, x < gw - 1 ? m[i + 1] : 0, y > 0 ? m[i - gw] : 0, y < gh - 1 ? m[i + gw] : 0) : 0;
    px[o] = r; px[o + 1] = gg; px[o + 2] = b; px[o + 3] = Math.min(0.92, 0.26 * v + 1.15 * (v - lo)) * 255;
  }
  mg.putImageData(data, 0, 0);
  g.imageSmoothingQuality = 'high';
  g.filter = `blur(${(W / gw * 0.45).toFixed(2)}px)`;
  g.drawImage(mc, 0, 0, W, H);
  g.filter = 'none';
}

const stills = [];
const EX_KEEP = ['ex02', 'ex01', 'ex04'];
const EX_NAME = { ex02: 'On the zebra', ex01: 'Beside the tram', ex04: 'Dark junction' };
const PAINT_NAME = ['Dark SUV', 'White car', 'Parked cars', 'Far end', 'Road curve', 'Tram stop'];
async function loadExamples() {
  let pack;
  try { const r = await fetch('v6/examples/pack.json'); if (!r.ok) return; pack = await r.json(); } catch (e) { return; }
  const grid = $('#exgrid');
  grid.innerHTML = '';
  for (const it of pack.items.filter(x => EX_KEEP.includes(x.id))) {
    const fig = document.createElement('figure');
    fig.className = 'ex';
    fig.innerHTML = '<div class="pair"><canvas></canvas><canvas></canvas></div><figcaption class="cap"></figcaption>';
    fig.querySelector('figcaption').textContent = it.caption;
    const [cc, ec] = fig.querySelectorAll('canvas');
    cc.setAttribute('aria-label', `Camera frame: ${it.caption}`);
    ec.setAttribute('aria-label', `Event camera frame: ${it.caption}`);
    grid.appendChild(fig);
    Promise.all([loadImage(`v6/examples/${it.id}/cam.webp`), loadImage(`v6/examples/${it.id}/ev.webp`)]).then(([ci, ei]) => {
      const draw = () => { drawStill(cc, ci, it.people.cam, 'cam', pack.mask_grid); drawStill(ec, ei, it.people.ev, 'ev', pack.mask_grid); };
      stills.push(draw); fig._draw = draw; draw();
    });
  }
  const tabs = document.createElement('div');
  tabs.className = 'tabs group';
  grid.before(tabs);
  const figs = [...grid.children];
  const show = n => {
    figs.forEach((f, m) => f.classList.toggle('on', m === n));
    [...tabs.children].forEach((b, m) => b.setAttribute('aria-pressed', String(m === n)));
    if (figs[n]._draw) figs[n]._draw();
  };
  pack.items.filter(x => EX_KEEP.includes(x.id)).forEach((it, n) => { const b = document.createElement('button'); b.className = 'opt'; b.textContent = EX_NAME[it.id]; b.addEventListener('click', () => show(n)); tabs.appendChild(b); });
  show(0);
}

async function loadPaintings() {
  let pack;
  try { const r = await fetch('v6/paint/pack.json'); if (!r.ok) return; pack = await r.json(); } catch (e) { return; }
  const grid = $('#paintgrid');
  grid.innerHTML = '';
  for (const it of pack.items) {
    const fig = document.createElement('figure');
    fig.className = 'ptrip';
    const base = `v6/paint/${it.id}`;
    fig.innerHTML = `<div class="three"><div class="cell evc"><span>Event frame</span><img class="ev" alt="Event frame given to Muse Image" src="${base}/events.webp"></div><div class="cell"><span>Muse Image</span><img alt="Picture painted by Muse Image from the event frame" src="${base}/painting.webp"></div><div class="cell"><span>Real camera</span><img alt="Real camera frame at the same instant" src="${base}/camera.webp"></div></div><figcaption class="cap"></figcaption>`;
    fig.querySelector('figcaption').textContent = it.caption;
    grid.appendChild(fig);
  }
  const tabs = document.createElement('div');
  tabs.className = 'tabs group';
  grid.before(tabs);
  const figs = [...grid.children];
  const show = n => {
    figs.forEach((f, m) => f.classList.toggle('on', m === n));
    [...tabs.children].forEach((b, m) => b.setAttribute('aria-pressed', String(m === n)));
    if (figs[n]._draw) figs[n]._draw();
  };
  pack.items.forEach((it, n) => { const b = document.createElement('button'); b.className = 'opt'; b.textContent = PAINT_NAME[n] || it.id; b.addEventListener('click', () => show(n)); tabs.appendChild(b); });
  show(0);
}

let stillTimer = 0;
addEventListener('resize', () => { clearTimeout(stillTimer); stillTimer = setTimeout(() => stills.forEach(f => f()), 200); });

loadClip('live');
loadExamples();
loadPaintings();
setPlaying(S.playing);
requestAnimationFrame(tick);
