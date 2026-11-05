'use strict';
const Fx = {
  own: [], imgs: {}, sn: -1, sk: '', mk: '', held: false, labels: [],
  s: { dim: 0, flare: 0, dev: 0, mark: 0, sb: [0, 0, 0, 0], db: [0, 0, 0, 0], view: 0, line: null, strip: [], stripred: [], img: null },
  init(hf) {
    this.tex = hf ? tex2d(1280, 960, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR) : tex2d(1280, 960, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.generateMipmap(gl.TEXTURE_2D);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    this.vbo = gl.createBuffer(); this.mvbo = gl.createBuffer(); this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(0);
  },
  reset() { this.sn = -1; this.mk = ''; this.held = false; this.clear(1, 1); },
  clear(r, g) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.colorMask(!!r, !!g, false, false); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.colorMask(true, true, true, true);
  },
  img(name) {
    if (this.imgs[name]) return this.imgs[name];
    const cache = this.imgs;
    const pending = fetch(`moments/${M.id}/${name}`).then(r => {
      if (!r.ok) throw new Error('texture unavailable');
      return r.blob();
    }).then(b => createImageBitmap(b)).then(bm => {
      const t = tex2d(1, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bm);
      bm.close();
      return cache[name] = t;
    }).catch(e => {
      if (cache[name] === pending) delete cache[name];
      throw e;
    });
    return cache[name] = pending;
  },
  prep(list) { return Promise.all(list.filter(c => c.develop).map(c => this.img(c.develop.image))); },
  hold(vt, list) {
    let k = null;
    for (const c of list) if (vt >= c.at && vt < c.to) { if (c.hold != null) k = c.hold; else if (c.spot && c.spot.k != null) k = c.spot.k; }
    if (k != null && !T.hold) { enterHold(k / 10); this.held = true; }
    if (k == null && this.held && T.hold) { this.held = false; leaveHold(); }
  },
  collect(tc, half, box, m) {
    const out = [], b0 = Math.max(0, Math.floor((tc - half) * 30)), b1 = Math.min(M.n_bins - 1, Math.floor((tc + half) * 30));
    const x0 = box[0] - m, x1 = box[2] + m, y0 = box[1] - m, y1 = box[3] + m;
    for (let b = b0; b <= b1; b++) for (let i = M.idx[b], e = Math.min(M.idx[b + 1], M.have); i < e; i++) {
      const v = M.ev[i], x = v & 1023, y = (v >>> 10) & 511;
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      const dt = Math.abs((b + (v >>> 20) / 4096) / 30 - tc);
      if (dt <= half) out.push(v, dt * 1000 | 0);
    }
    return out;
  },
  subset(tc, box) {
    const key = tc + '|' + box;
    if (this.sk === key) return;
    this.sk = key; this.sn = -1;
    const raw = this.collect(tc, .5, box, 26), off = new Int32Array(502), n = raw.length / 2;
    for (let i = 1; i < raw.length; i += 2) off[raw[i] + 1]++;
    for (let i = 1; i < 502; i++) off[i] += off[i - 1];
    const put = off.slice(), arr = new Uint32Array(n);
    for (let i = 0; i < raw.length; i += 2) arr[put[raw[i + 1]]++] = raw[i];
    this.off = off; this.box = box;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
  },
  begin(buf, box, ch, fe) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, 1280, 960);
    gl.useProgram(this.prog.p);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.vertexAttribIPointer(0, 1, gl.UNSIGNED_INT, 0, 0);
    gl.uniform4fv(this.prog.u.box, box);
    gl.uniform2f(this.prog.u.ch, ch[0], ch[1]);
    gl.uniform2f(this.prog.u.fe, fe[0], fe[1]);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  },
  fill(ms) {
    const n = this.off[clamp(Math.round(ms / 2), 0, 500) + 1], all = this.off[401];
    if (n === this.sn) return;
    if (n < this.sn || this.sn < 0) { this.clear(1, 0); this.sn = 0; }
    const b = this.box, wt = Math.min(1.6, 4 * (b[2] - b[0]) * (b[3] - b[1]) / Math.max(all, 1));
    this.begin(this.vbo, b, [wt, 0], [20, 26]);
    gl.colorMask(true, false, false, false);
    gl.drawArrays(gl.POINTS, this.sn, n - this.sn);
    gl.colorMask(true, true, true, true);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.main);
    gl.bindTexture(gl.TEXTURE_2D, this.tex); gl.generateMipmap(gl.TEXTURE_2D);
    this.sn = n;
  },
  marks(c) {
    const tc = c.k != null ? c.k / 10 : T.frozen ? T.frozenAt : T.t, ms = c.win ?? 300, key = tc.toFixed(3) + ms + JSON.stringify(c.marks);
    if (key === this.mk) return;
    this.mk = key;
    this.clear(0, 1);
    const pm = M.masks.get(Math.round(tc * 10));
    const on = (x, y) => {
      if (!pm) return 1;
      const u = clamp(x / 4 - .5, 0, 158), v = clamp(y / 4 - .5, 0, 118), i = u | 0, j = v | 0, f = u - i, g = v - j, q = j * 160 + i;
      return ((pm[q] * (1 - f) + pm[q + 1] * f) * (1 - g) + (pm[q + 160] * (1 - f) + pm[q + 161] * f) * g) / 255;
    };
    for (const m of c.marks) {
      const raw = this.collect(tc, ms / 2000, m.box, 4), keep = [];
      for (let i = 0; i < raw.length; i += 2) if (on(raw[i] & 1023, (raw[i] >>> 10) & 511) > .1 + .8 * (i * .30901699 % 1)) keep.push(raw[i]);
      const arr = new Uint32Array(keep);
      if (!arr.length) continue;
      const b = m.box, wt = Math.min(2.4, .55 * 4 * (b[2] - b[0]) * (b[3] - b[1]) / arr.length);
      this.begin(this.mvbo, b, [0, wt], [-2.5, 3.5]);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
      gl.colorMask(false, true, false, false);
      gl.drawArrays(gl.POINTS, 0, arr.length);
      gl.colorMask(true, true, true, true);
    }
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.main);
  },
  tag(c, a) {
    const ms = c ? c.marks.filter(m => m.label) : [], el = $('tags');
    while (this.labels.length < ms.length) { const s = document.createElement('span'); el.append(s); this.labels.push(s); }
    this.labels.forEach((s, i) => {
      const m = ms[i];
      s.hidden = !m;
      if (!m) return;
      if (s.textContent !== m.label) s.textContent = m.label;
      s.style.opacity = a;
      s.style.transform = `translate(${(fit.ox + m.box[2] * fit.sc) * fit.s + 10}px,${(fit.oy + m.box[1] * fit.sc) * fit.s}px)`;
    });
  },
  at(vt, list) {
    const s = { dim: 0, flare: 0, dev: 0, mark: 0, sb: this.s.sb, db: this.s.db, view: 0, line: null, strip: [], stripred: [], img: null };
    const env = (c, a, b) => sstep(0, a, vt - c.at) * (1 - sstep(c.to - b, c.to, vt));
    let tc = null, fc = null, dc = null, mc = null, past = null;
    for (const c of list) {
      if (c.flare && vt >= c.at) past = c;
      if (vt < c.at || vt >= c.to) continue;
      if (c.spot) { s.sb = c.spot.box; if (c.spot.k != null) tc = c.spot.k / 10; s.dim = Math.max(s.dim, env(c, 1.4, .8) * (c.spot.dim ?? 1)); }
      if (c.flare) fc = c;
      if (c.develop) dc = c;
      if (c.marks) mc = c;
      if (c.stripmarks) s.strip.push(...c.stripmarks);
      if (c.stripred) s.stripred.push(...c.stripred);
      if (c.line) s.line = typeof c.line === 'string' ? { text: c.line } : c.line;
      if (c.view) s.view = c.view;
    }
    if (tc == null) tc = T.frozen ? T.frozenAt : T.t;
    const src = fc || (dc && past) || null, box = src ? src.flare : dc && dc.develop.box;
    if (box) {
      this.subset(src && src.k != null ? src.k / 10 : dc && dc.k != null ? dc.k / 10 : tc, box);
      let ms = (src && src.win) || 800;
      if (fc) { const g = fc.grow ?? Math.max(.6, (fc.to - fc.at) * .7); ms = Math.max(16, ms * Math.pow(clamp((vt - fc.at) / g), 1.35)); }
      this.fill(ms);
    }
    if (dc) {
      const im = this.imgs[dc.develop.image];
      if (im instanceof WebGLTexture) { s.img = im; s.db = dc.develop.box; s.dev = Math.max(1e-3, sstep(0, dc.develop.over ?? 4, vt - dc.at) * (1 - sstep(dc.to - (dc.develop.out || 1e-3), dc.to, vt))); }
      else Promise.resolve(this.img(dc.develop.image)).catch(() => {});
    }
    if (fc) s.flare = env(fc, .3, .9) * (1 - sstep(.42, .96, s.dev));
    if (mc) { this.marks(mc); s.mark = env(mc, .9, .7) * .92; }
    this.tag(mc, s.mark);
    this.s = s;
  }
};

