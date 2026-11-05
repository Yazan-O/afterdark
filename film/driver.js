window.__film = (() => {
  let S = null, peo = null, empty = new Map(), ov = null, keyEl = null, lastKey = '', lastOv = '';
  const FPS = 30;
  const rateAt = ft => { let a = 0; for (const [d, r] of S.path) { if (ft < a + d - 1e-9) return r; a += d; } return S.path[S.path.length - 1][1]; };
  const lerp = (kv, x) => {
    if (x <= kv[0][0]) return kv[0][1];
    for (let i = 1; i < kv.length; i++) if (x < kv[i][0]) { const [a, u] = kv[i - 1], [b, v] = kv[i]; return u + (v - u) * (x - a) / (b - a); }
    return kv[kv.length - 1][1];
  };
  const within = (list, ft) => (list || []).filter(c => ft >= c.at && ft < c.to);
  function css(extra) {
    const s = document.createElement('style');
    s.textContent = `
#caption{font-size:44px;line-height:1.2;left:40px;right:40px;bottom:72px;max-width:1560px;min-height:0;text-shadow:2px 2px 0 #000,-1px -1px 0 #000}
#stamp{font-size:26px;opacity:1;text-shadow:2px 2px 0 #000}
#credit{font-size:20px;opacity:.85}
#low{left:40px;right:40px;bottom:22px}
#reason{font-size:30px;line-height:1.28;max-width:800px;text-shadow:2px 2px 0 #000,-1px -1px 0 #000}
#reason b{font-size:24px}
#ft{position:fixed;left:40px;top:36px;right:40px;white-space:pre-wrap;font:500 34px/1.34 "IBM Plex Mono","Plex Local",monospace;color:#EDE6D6;text-shadow:2px 2px 0 #000,-1px -1px 0 #000;pointer-events:none}
#ft.card{top:40%;font-size:56px;text-align:left;left:160px}
#ft.big{font-size:44px;line-height:1.36}
#layer.big{font-size:36px}
#caption,#low,#strip,#keycap,#ft{z-index:3}
body.scrim::after{content:"";position:fixed;left:0;right:0;bottom:0;height:230px;background:linear-gradient(rgba(0,0,0,0),rgba(0,0,0,.66));pointer-events:none;z-index:2}
.fblur{position:fixed;backdrop-filter:blur(11px);z-index:1;display:none}
#ft.endurl{left:40px;top:430px;font-size:38px;line-height:1.4}
#ft.models{font-size:42px;line-height:1.42;top:150px;left:90px;right:60px}
#ft.wait{top:38%;left:120px;right:120px;font-size:60px;line-height:1.3}
${extra || ''}`;
    document.head.append(s);
    ov = document.createElement('p'); ov.id = 'ft'; document.body.append(ov);
    keyEl = $('keycap');
  }
  async function prep(shot, extraCss) {
    S = shot;
    await window.__ready;
    css(extraCss);
    await document.fonts.load('500 44px "IBM Plex Mono"').catch(() => {});
    await document.fonts.load('500 44px "Plex Local"').catch(() => {});
    await document.fonts.ready;
    for (let i = 0; i < 40 && !(M.people.size && M.drift); i++) await new Promise(r => setTimeout(r, 100));
    M.kl = null;
    peo = M.people; Agent.auto = false; Beat.done = true; Beat.on = false;
    Object.assign(T, { session: 0, sheet: S.sheet ? S.sheet[0][1] : 9 });
    if (S.kind === 'card') { canvas.style.visibility = 'hidden'; document.body.style.background = '#000'; }
    if (S.kind === 'card' || S.kind === 'still') { $('strip').style.visibility = 'hidden'; $('low').style.visibility = 'hidden'; }
    if (S.kind !== 'still' && S.kind !== 'card') document.body.classList.add('scrim');
    for (const b of S.blur || []) { const d = b.el = document.createElement('div'); d.className = 'fblur'; d.style.cssText = `left:${b.box[0]}px;top:${b.box[1]}px;width:${b.box[2] - b.box[0]}px;height:${b.box[3] - b.box[1]}px`; document.body.append(d); }
    if (S.dim != null) canvas.style.filter = `brightness(${S.dim})`;
    await Fx.prep(S.fx || []);
    return { seconds: M.seconds, people: peo.size };
  }
  async function step(i) {
    const ft = i / FPS;
    let dirty = i === 0;
    if (S.kind !== 'card' && (S.kind !== 'still' || i === 0)) {
      M.people = ft >= (S.mask_from ?? 0) ? peo : empty;
      const r = rateAt(ft);
      await window.__fx(ft, S.fx || []);
      if (r === 0 && !T.hold) enterHold(S.hold_at ?? T.t);
      if (r > 0 && T.hold) leaveHold();
      sim(i === 0 ? 0 : (r > 0 ? r : (S.hold_clock ?? 1)) / FPS);
      T.sheet = S.sheet ? lerp(S.sheet, ft) : 9;
      const k = clamp(Math.floor(T.t * 10), 0, M.n_paint - 1), k1 = Math.min(k + 1, M.n_paint - 1), kc = clamp(Math.round(T.t * 10), 0, M.n_paint - 1), pp = paintPair();
      await Promise.all([loadImg('paint', 'paint', pp.a, loadToken), loadImg('paint', 'paint', pp.b, loadToken), loadImg('back', 'backing', k, loadToken), loadImg('back', 'backing', k1, loadToken), loadImg('cam', 'cam', kc, loadToken)]);
      flushUploads(1e9);
      let view = 0;
      for (const c of within(S.keys, ft)) view = c.key === 'space' ? 1 : 2;
      K.space = view === 1;
      const dev = devAmount();
      draw(view, dev);
      gl.finish();
      drawStrip();
      const cap = within(S.cues, ft).map(c => c.text).join('');
      hud(view, dev, cap);
      const suf = S.hide_rate ? '' : r === 0 ? '  held' : r < .995 ? `  ${(+r.toFixed(3))}x` : '';
      if (suf && !Dom.sb.textContent.endsWith(suf)) Dom.sb.textContent += suf;
      if (S.strip) { const h = lerp(S.strip, ft); strip.c.style.height = h + 'px'; $('low').style.bottom = (h + 8) + 'px'; $('caption').style.bottom = (h + 58) + 'px'; }
    }
    for (const b of S.blur || []) b.el.style.display = ft >= b.at && ft < b.to ? 'block' : 'none';
    const kp = within(S.keypic, ft)[0];
    const down = kp ? (kp.down || []).some(d => ft >= d[0] && ft < d[1]) : false;
    const kk = kp ? kp.key + down : '';
    if (kk !== lastKey) {
      lastKey = kk;
      keyEl.hidden = !kp;
      if (kp) { keyEl.textContent = kp.key; keyEl.classList.toggle('down', down); keyEl.classList.toggle('wide', kp.key === 'Space'); }
    }
    const tx = within(S.text, ft)[0];
    const ok = tx ? (tx.cls || '') + '|' + tx.lines.join('\n') + '|' + Math.floor((ft - tx.at) / (tx.reveal || 1e9)) : '';
    if (ok !== lastOv) {
      lastOv = ok;
      dirty = true;
      ov.className = tx ? tx.cls || '' : '';
      ov.textContent = tx ? (tx.reveal ? tx.lines.slice(0, 1 + Math.floor((ft - tx.at) / tx.reveal)) : tx.lines).join('\n') : '';
    }
    return { t: M.t0_us + Math.round(T.t * 1e6), T: T.t, dirty };
  }
  return { prep, step };
})();
