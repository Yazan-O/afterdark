'use strict';
(() => {
  const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const load = src => new Promise(res => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = src; });

  document.documentElement.classList.add('js');
  const rv = [...document.querySelectorAll('section h2, section .sub, section figure, .facts li, .route, .uses p, .cmp, .lens-cap')];
  rv.forEach(el => el.classList.add('rv'));
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const sib = [...e.target.parentElement.children].filter(x => x.classList.contains('rv'));
    e.target.style.transitionDelay = `${Math.min(4, sib.indexOf(e.target)) * 80}ms`;
    e.target.classList.add('in');
    io.unobserve(e.target);
  }), { rootMargin: '0px 0px -8% 0px' });
  rv.forEach(el => io.observe(el));

  async function lens() {
    const box = document.getElementById('lens');
    if (!box) return;
    const [raw, ev, mask] = await Promise.all(['raw', 'ev', 'mask'].map(n => load(`v6/wow/hero_${n}.${n === 'mask' ? 'png' : 'webp'}`)));
    const meta = await (await fetch('v6/wow/wow.json')).json();
    if (!raw || !ev || !mask) return;
    const c = box.querySelector('canvas'), g = c.getContext('2d');
    const rev = document.createElement('canvas'), rg = rev.getContext('2d');
    const tmp = document.createElement('canvas'), tg = tmp.getContext('2d');
    const tag = box.querySelector('.tag');
    let W = 0, H = 0, R = 0;
    const st = { x: 0.3, y: 0.55, tx: 0.3, ty: 0.55, user: false, last: 0, idleAt: 0 };
    const pb = meta.box, px = (pb[0] + pb[2]) / 2, py = (pb[1] + pb[3]) / 2;

    function build() {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      W = Math.round(c.clientWidth * dpr); H = Math.round(W * 3 / 4);
      c.width = rev.width = tmp.width = W; c.height = rev.height = tmp.height = H;
      R = Math.max(70 * dpr, W * 0.17);
      rg.fillStyle = '#000'; rg.fillRect(0, 0, W, H);
      rg.filter = `blur(${(W / 900).toFixed(2)}px)`;
      rg.drawImage(ev, 0, 0, W, H);
      rg.filter = 'none';
      rg.globalCompositeOperation = 'multiply'; rg.fillStyle = '#9FB8C2'; rg.fillRect(0, 0, W, H);
      rg.globalCompositeOperation = 'source-over';
      const mc = document.createElement('canvas'); mc.width = W; mc.height = H;
      const mg = mc.getContext('2d');
      mg.drawImage(mask, 0, 0, W, H);
      mg.globalCompositeOperation = 'source-in'; mg.fillStyle = '#86BFB2'; mg.fillRect(0, 0, W, H);
      rg.globalAlpha = 0.22; rg.drawImage(mc, 0, 0); rg.globalAlpha = 1;
      rg.filter = `drop-shadow(0 0 ${(W / 160).toFixed(1)}px rgba(134,191,178,.9))`;
      rg.globalAlpha = 0.35; rg.drawImage(mc, 0, 0); rg.globalAlpha = 1; rg.filter = 'none';
    }

    function frame(now) {
      const dt = Math.min(64, now - (st.last || now)); st.last = now;
      if (!st.user || now - st.idleAt > 2500) {
        st.user = false;
        const s = now / 1000;
        st.tx = 0.5 + 0.36 * Math.sin(s * 0.31) ; st.ty = 0.55 + 0.16 * Math.sin(s * 0.53 + 1);
        const d = Math.hypot(st.tx - px, st.ty - py);
        if (d < 0.25) { st.tx += (px - st.tx) * 0.5; st.ty += (py - st.ty) * 0.5; }
      }
      const a = 1 - Math.exp(-dt / (st.user ? 90 : 400));
      st.x += (st.tx - st.x) * a; st.y += (st.ty - st.y) * a;
      const cx = st.x * W, cy = st.y * H;
      g.drawImage(raw, 0, 0, W, H);
      tg.globalCompositeOperation = 'source-over';
      tg.clearRect(0, 0, W, H); tg.drawImage(rev, 0, 0);
      tg.globalCompositeOperation = 'destination-in';
      const gr = tg.createRadialGradient(cx, cy, R * 0.55, cx, cy, R);
      gr.addColorStop(0, 'rgba(0,0,0,1)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
      tg.fillStyle = gr; tg.fillRect(0, 0, W, H);
      g.drawImage(tmp, 0, 0);
      g.strokeStyle = 'rgba(236,234,228,.18)'; g.lineWidth = Math.max(1, W / 900);
      g.beginPath(); g.arc(cx, cy, R * 0.97, 0, Math.PI * 2); g.stroke();
      const near = Math.max(0, 1 - Math.hypot((st.x - px) * W, (st.y - py) * H) / (R * 0.9));
      tag.style.opacity = near.toFixed(2);
      tag.style.transform = `translate(${(pb[0] * c.clientWidth - tag.offsetWidth - 10).toFixed(0)}px, ${(pb[1] * c.clientHeight - 6).toFixed(0)}px)`;
      if (!RM) requestAnimationFrame(frame);
    }

    const at = e => { const r = c.getBoundingClientRect(); st.tx = (e.clientX - r.left) / r.width; st.ty = (e.clientY - r.top) / r.height; st.user = true; st.idleAt = performance.now(); };
    c.addEventListener('pointermove', at);
    c.addEventListener('pointerdown', at);
    build();
    addEventListener('resize', () => { build(); if (RM) frame(performance.now()); });
    if (RM) { st.x = px; st.y = py; frame(performance.now()); } else requestAnimationFrame(frame);
  }

  async function cloud() {
    const box = document.getElementById('cloud');
    if (!box || !window.THREE) return;
    const buf = await (await fetch('v6/wow/cloud.bin')).arrayBuffer();
    const a = new Uint16Array(buf), n = a.length / 4;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const on = [0.53, 0.75, 0.70], off = [0.85, 0.70, 0.42];
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (a[i * 4] - 320) / 240;
      pos[i * 3 + 1] = -(a[i * 4 + 1] - 240) / 240;
      pos[i * 3 + 2] = (a[i * 4 + 2] / 65535 - 0.5) * 5;
      const k = a[i * 4 + 3] ? on : off;
      col[i * 3] = k[0]; col[i * 3 + 1] = k[1]; col[i * 3 + 2] = k[2];
    }
    const T = window.THREE;
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.BufferAttribute(pos, 3));
    geo.setAttribute('color', new T.BufferAttribute(col, 3));
    const mat = new T.PointsMaterial({ size: 0.011, vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false });
    const pts = new T.Points(geo, mat);
    const scene = new T.Scene();
    scene.add(pts);
    const cam = new T.PerspectiveCamera(38, 16 / 9, 0.1, 100);
    const ren = new T.WebGLRenderer({ canvas: box.querySelector('canvas'), antialias: true, alpha: true });
    ren.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    const size = () => { const w = box.clientWidth, h = Math.round(w * 9 / 16); ren.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); };
    size(); addEventListener('resize', size);
    let yaw = -0.9, pitch = 0.18, vy = RM ? 0 : 0.08, drag = null, seen = false, last = 0;
    new IntersectionObserver(es => { seen = es[0].isIntersecting; }).observe(box);
    const c = box.querySelector('canvas');
    c.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, yaw, pitch }; c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointermove', e => { if (!drag) return; yaw = drag.yaw + (e.clientX - drag.x) / 220; pitch = Math.max(-0.7, Math.min(0.9, drag.pitch + (e.clientY - drag.y) / 300)); });
    const end = () => { drag = null; };
    c.addEventListener('pointerup', end); c.addEventListener('pointercancel', end);
    function draw(now) {
      const dt = Math.min(64, now - (last || now)) / 1000; last = now;
      if (seen) {
        if (!drag) yaw += vy * dt;
        const r = 6.2;
        cam.position.set(r * Math.sin(yaw) * Math.cos(pitch), r * Math.sin(pitch), r * Math.cos(yaw) * Math.cos(pitch));
        cam.lookAt(0, 0, 0);
        ren.render(scene, cam);
      }
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
  }

  function compare() {
    const box = document.getElementById('cmp');
    if (!box) return;
    const top = box.querySelector('.wipe-top'), handle = box.querySelector('.wipe-handle'), wipe = box.querySelector('.wipe');
    let x = 0.5;
    const set = v => { x = Math.max(0, Math.min(1, v)); wipe.setAttribute('aria-valuenow', String(Math.round(x * 100))); top.style.clipPath = `inset(0 0 0 ${(x * 100).toFixed(2)}%)`; handle.style.left = `${(x * 100).toFixed(2)}%`; };
    set(0.5);
    let on = false;
    const at = e => { const r = wipe.getBoundingClientRect(); set((e.clientX - r.left) / r.width); };
    wipe.addEventListener('pointerdown', e => { on = true; wipe.setPointerCapture(e.pointerId); at(e); });
    wipe.addEventListener('pointermove', e => { if (on) at(e); });
    wipe.addEventListener('pointerup', () => { on = false; });
    wipe.addEventListener('keydown', e => { if (e.key === 'ArrowLeft') set(x - 0.05); if (e.key === 'ArrowRight') set(x + 0.05); });
    box.querySelectorAll('[data-k]').forEach(b => b.addEventListener('click', () => {
      const k = b.dataset.k;
      box.querySelectorAll('[data-k]').forEach(o => o.setAttribute('aria-pressed', String(o === b)));
      const ims = box.querySelectorAll('img[data-src]');
      ims.forEach(im => im.classList.add('out'));
      setTimeout(() => ims.forEach(im => { im.onload = () => im.classList.remove('out'); im.src = im.dataset.src.replace('{k}', k); }), RM ? 0 : 380);
    }));
  }

  lens(); cloud(); compare();
})();
