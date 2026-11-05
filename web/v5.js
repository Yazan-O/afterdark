'use strict';
const Q = new URLSearchParams(location.search);
const RENDER = Q.get('render') === '1', FILM = Q.get('cap') === 'film', COARSE = matchMedia('(pointer:coarse)').matches;
const $ = id => document.getElementById(id);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const W0 = 640, H0 = 480, TAU_R = 0.9, TAU_G = 0.07, TAU_A = 0.06;
const canvas = $('gl');
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: RENDER, powerPreference: 'high-performance' });

const QUAD_VS = `#version 300 es
out vec2 v;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);v=p;gl_Position=vec4(p*2.-1.,0,1);}`;

const DECAY_FS = `#version 300 es
precision highp float;in vec2 v;out vec4 o;uniform sampler2D p;uniform vec4 k;uniform vec2 px;uniform float diff;
void main(){vec4 c=texture(p,v);
float n=texture(p,v+vec2(px.x,0)).r+texture(p,v-vec2(px.x,0)).r+texture(p,v+vec2(0,px.y)).r+texture(p,v-vec2(0,px.y)).r;
c.r=mix(c.r,n*.25,diff);o=c*k;}`;

const EV_VS = `#version 300 es
precision highp float;precision highp int;
layout(location=0) in uint e;
uniform highp sampler2D mask;uniform vec2 win;uniform vec4 view;uniform float size,budget,age,fade,dim,paper;uniform vec4 chan;
flat out vec4 c;
void main(){
uint x=e&1023u,y=(e>>10)&511u,p=(e>>19)&1u,d=e>>20;
float u=float(d)/4096.,h=fract(float(gl_VertexID)*.61803398875);
gl_Position=vec4(2,2,2,1);gl_PointSize=1.;c=vec4(0);
if(u<win.x||u>=win.y||h>budget)return;
float m=step(.5,texture(mask,(vec2(x,y)+.5)/vec2(640.,480.)).r);
vec2 j=paper>.5?vec2(.5):fract(float(gl_VertexID)*vec2(.7548776662,.5698402910));
gl_Position=vec4((vec2(x,y)+j)*view.xy+view.zw,0,1);
if(paper>.5){c=chan*vec4(1,1,1,m);return;}
float a=clamp(1.-(age+1.-u)/fade,0.,1.);
gl_PointSize=size+(m>.5?.9:0.);
c=m>.5?vec4(1.,.231,.114,clamp(a*2.,0.,1.)):vec4(.929,.902,.839,a*dim*(p==1u?1.:.58));
}`;

const FX_VS = `#version 300 es
precision highp float;precision highp int;
layout(location=0) in uint e;
uniform vec4 box;uniform vec2 ch,fe;
flat out vec4 c;
void main(){
vec2 p=vec2(float(e&1023u),float((e>>10)&511u))+fract(float(gl_VertexID)*vec2(.7548776662,.5698402910));
vec2 q=abs(p-(box.xy+box.zw)*.5)-(box.zw-box.xy)*.5;
gl_Position=vec4(p/vec2(320.,240.)-1.,0,1);gl_PointSize=1.;
c=vec4(ch*(1.-smoothstep(fe.x,fe.y,length(max(q,0.))+min(max(q.x,q.y),0.))),0,1);
}`;

const EV_FS = `#version 300 es
precision highp float;flat in vec4 c;out vec4 o;void main(){o=c;}`;

const VIEW_FS = `#version 300 es
precision highp float;out vec4 o;
uniform sampler2D P,PA,PB,BA,BB,C,RP,MA,MB,FE;
uniform vec2 res,org;uniform float sc,f,m,dev,mode,events;uniform vec3 da,db;uniform vec2 swap;uniform vec4 fx,sbox,dbox,fbox;
const vec3 BONE=vec3(.929,.902,.839),VERM=vec3(1.,.231,.114);
vec3 cam(vec2 s){return pow(max(texture(C,s).rgb-.06,0.)/.94,vec3(.7));}
float hash(vec2 q){vec3 a=fract(q.xyx*.1031);a+=dot(a,a.yzx+33.33);return fract((a.x+a.y)*a.z);}
float sd(vec2 p,vec4 b){vec2 q=abs(p-(b.xy+b.zw)*.5)-(b.zw-b.xy)*.5;return length(max(q,0.))+min(max(q.x,q.y),0.);}
float ink(sampler2D t,vec2 s){float a=0.;for(int i=-2;i<=2;i++)for(int j=-2;j<=2;j++)a+=texture(t,s+vec2(i,j)*vec2(.0075,.01)).r*exp(-float(i*i+j*j)/3.);return a/8.76;}
void main(){
vec2 s=(vec2(gl_FragCoord.x,res.y-gl_FragCoord.y)-org)/(vec2(640.,480.)*sc);
if(s.x<0.||s.y<0.||s.x>1.||s.y>1.){o=vec4(0,0,0,1);return;}
if(mode>.5&&mode<1.5){o=vec4(cam(s),1);return;}
ivec2 cell=ivec2(s*vec2(640.,480.));
vec4 e=texture(P,s),n=texelFetch(P,cell,0);
float g=hash(vec2(cell)),fresh=clamp(e.g*1.4,0.,1.);
vec2 sa=(s-.5-da.yz)/da.x+.5,sb=(s-.5-db.yz)/db.x+.5;
vec2 ea=step(vec2(0),sa)*step(sa,vec2(1)),eb=step(vec2(0),sb)*step(sb,vec2(1));
float ia=ea.x*ea.y,ib=eb.x*eb.y,mc=mix(m,clamp(m,.003,.997),swap.x+swap.y>0.?1.:0.),wa=(1.-mc)*(1.-swap.x*smoothstep(.02,.3,texture(MA,sa).r))+1e-4*(1.-m),wb=mc*(1.-swap.y*smoothstep(.02,.3,texture(MB,sb).r))+1e-4*m;
vec3 pa=mix(texture(PA,sa).rgb,texture(PB,sb).rgb,ib>ia?1.:ia>ib?0.:wb/(wa+wb));
vec2 ep=s*vec2(640.,480.);
if(fx.z>0.){float en=1.-exp(-textureLod(FE,s,3.).r*1.3),q=fx.z,l=(1.-q)*(.72+.56*g),dd=sd(ep,dbox),rv=smoothstep(l,l+.28,en+q*q)*(1.-smoothstep(-20.,30.,dd));vec3 rc=texture(RP,s).rgb,dc=abs(rc-pa);rv*=mix(1.,smoothstep(.012,.075,max(dc.r,max(dc.g,dc.b))),smoothstep(-30.,-14.,dd));pa=mix(pa,pow(rc,vec3(1.+1.5*(1.-rv))),rv);}
float th=.1+.5*g,x=e.r*dev*1.5,lim=(1.-.72*dev)*th;
float rev=smoothstep(lim,lim+.55,x);
float fl=smoothstep(.5,1.,dev);
float r=max(fl,rev);
vec3 col=pow(pa,vec3(1.+1.6*(1.-r)))*r*(1.+.08*fresh*dev);
if(mode>1.5){float bk=mix(ink(BA,s),ink(BB,s),f);col=pa*smoothstep(.1,.62,bk);}
col=pow(col,vec3(.84));
float hist=min(n.g,1.);
col=max(col,BONE*hist*mix(.5,.12,dev)*events);
if(fx.x>0.){float d=max(sd(ep,sbox),0.);col*=1.-fx.x*mix(.6,.3,exp(-d*d/26000.));}
vec2 fe=texture(FE,s).rg;
col=mix(col,VERM,(1.-exp(-fe.g))*fx.w);
col+=BONE*min(log(1.+fe.r*14.)/5.7,1.)*(1.-smoothstep(-24.,22.,sd(ep,fbox)+(hash(gl_FragCoord.xy)-.5)*26.))*fx.y;
col=mix(col,VERM,clamp(n.a*events,0.,1.));
o=vec4(col,1);
}`;

function program(vs, fs) {
  const p = gl.createProgram();
  for (const [t, s] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const h = gl.createShader(t); gl.shaderSource(h, s); gl.compileShader(h);
    if (!gl.getShaderParameter(h, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(h));
    gl.attachShader(p, h);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i).name; u[a] = gl.getUniformLocation(p, a); }
  return { p, u };
}

const T = { t: 0, prev: 0, sheet: 0, session: 0, frozen: false, hold: false, holdFrom: -9, leftAt: -9, leftK: 0, live: null, last: -1, bin: -1 };
const K = { space: false, d: false };
let M = null, P, V, E, paperFbo = [], paperTex = [], flip = 0, vbo, slots = [], capN = 16000, maskTex, maskA, maskB, blackTex, curMask = -2;
let budget = 1, fit = { sc: 1, ox: 0, oy: 0, W: 1, H: 1, s: 1 }, loadToken = 0, moments = [], momentIx = 0;

function tex2d(w, h, inner, fmt, type, filter, data = null) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, inner, w, h, 0, fmt, type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function initGL() {
  P = program(QUAD_VS, DECAY_FS); E = program(EV_VS, EV_FS); V = program(QUAD_VS, VIEW_FS); Fx.prog = program(FX_VS, EV_FS);
  const hf = gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float');
  for (let i = 0; i < 2; i++) {
    const t = hf ? tex2d(W0, H0, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR) : tex2d(W0, H0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    paperTex.push(t); paperFbo.push(f);
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  maskTex = tex2d(160, 120, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR, new Uint8Array(19200));
  maskA = tex2d(160, 120, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR, new Uint8Array(19200));
  maskB = tex2d(160, 120, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR, new Uint8Array(19200));
  blackTex = tex2d(1, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST, new Uint8Array([0, 0, 0, 255]));
  vbo = gl.createBuffer();
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.disable(gl.DEPTH_TEST);
  Fx.init(hf);
  gl.bindVertexArray(Fx.main = vao);
}

function sizeSlots(n) {
  capN = n; slots = new Array(8).fill(-1);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, n * 4 * 8, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribIPointer(0, 1, gl.UNSIGNED_INT, 0, 0);
}

function resize() {
  const dpr = RENDER ? 1 : Math.min(devicePixelRatio || 1, 2);
  const cw = RENDER ? 1920 : innerWidth, ch = RENDER ? 1080 : innerHeight;
  let w = Math.max(1, Math.round(cw * dpr)), h = Math.max(1, Math.round(ch * dpr));
  const over = Math.max(1, w / 2560, h / 1600);
  w = Math.round(w / over); h = Math.round(h / over);
  canvas.width = w; canvas.height = h;
  if (RENDER) { canvas.style.width = cw + 'px'; canvas.style.height = ch + 'px'; }
  let sc, oy;
  if (w / h >= 1) {
    sc = Math.max(w / W0, h / H0);
    const over2 = H0 * sc - h;
    oy = over2 > 0 ? -over2 / 3 : 0;
  } else {
    sc = (w / 0.72) / W0;
    oy = (h - H0 * sc) * 0.42;
  }
  fit = { sc, ox: (w - W0 * sc) / 2, oy, W: w, H: h, s: cw / w };
  fit.crop = { sc, ox: fit.ox, oy };
  fit.whole = w < h ? { sc: w / W0, ox: 0, oy: (h - H0 * w / W0) * 0.42 } : null;
  drawStrip(true);
}

function clearPaper() {
  for (const f of paperFbo) { gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
}

function bindBin(b) {
  const s = b % 8;
  const a = M.idx[b], n = Math.min(M.idx[b + 1], M.have) - a;
  if (n <= 0) return null;
  if (slots[s] !== b) {
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, s * capN * 4, M.ev, a, n);
    slots[s] = b;
  }
  return { first: s * capN, n };
}

function evUniforms(paper, view) {
  gl.useProgram(E.p);
  gl.uniform1f(E.u.paper, paper);
  gl.uniform4fv(E.u.view, view);
  gl.uniform1f(E.u.budget, budget);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, maskTex);
  gl.uniform1i(E.u.mask, 0);
}

function splatBins(t0, t1, chanFn) {
  const b0 = Math.floor(t0 * 30), b1 = Math.min(Math.floor((t1 - 1e-9) * 30), M.n_bins - 1);
  for (let b = b0; b <= b1; b++) {
    if (b < 0) continue;
    const r = bindBin(b); if (!r) continue;
    gl.uniform2f(E.u.win, clamp(t0 * 30 - b), clamp(t1 * 30 - b, 0, 1.001));
    gl.uniform4fv(E.u.chan, chanFn(b, r.n));
    gl.drawArrays(gl.POINTS, r.first, r.n);
  }
}

const paperView = [1 / 320, 1 / 240, -1, -1];

function binWeight(b, n, t0, tau) {
  return M.K * (M.counts[b] / n) / budget * (tau ? Math.exp(-(t0 - b / 30) / tau) : 1);
}

function preroll(t) {
  clearPaper();
  if (!M.have) return;
  gl.bindFramebuffer(gl.FRAMEBUFFER, paperFbo[flip]);
  gl.viewport(0, 0, W0, H0);
  evUniforms(1, paperView);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  const b1 = Math.floor(t * 30), b0 = Math.max(0, b1 - 45);
  gl.uniform1f(E.u.size, 1);
  for (let b = b0; b <= b1; b++) splatBins(b / 30, Math.min((b + 1) / 30, t + 1e-6), (bb, n) => { const a = Math.exp(-(t - bb / 30) / TAU_G) / budget * 1.3; return [binWeight(bb, n, t, TAU_R), a, 0, 0]; });
  gl.disable(gl.BLEND);
}

const Paper = {
  step(dt, ranges) {
    const src = flip, dst = 1 - flip;
    gl.bindFramebuffer(gl.FRAMEBUFFER, paperFbo[dst]);
    gl.viewport(0, 0, W0, H0);
    gl.useProgram(P.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, paperTex[src]);
    gl.uniform1i(P.u.p, 0);
    gl.uniform2f(P.u.px, 1 / W0, 1 / H0);
    gl.uniform1f(P.u.diff, clamp(dt * 7));
    gl.uniform4f(P.u.k, Math.exp(-dt / TAU_R), Math.exp(-dt / TAU_G), 0, Math.exp(-dt / TAU_A));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 3);
    flip = dst;
    evUniforms(1, paperView);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.uniform1f(E.u.size, 1);
    for (const [a, b] of ranges) splatBins(a, b, (bb, n) => [binWeight(bb, n), 1 / budget * 1.3, 0, 1.1 * 16000 / capN * (M.counts[bb] / n) / 20.4 / budget]);
    gl.disable(gl.BLEND);
  }
};

const BEATS = { crosswalk: 68 };
const FILLED = { crosswalk: [38, 81, 84] };
function enterHold(t0) { Object.assign(T, { frozen: true, hold: true, holdFrom: T.session, frozenAt: t0, frozenClock: 0, t: t0, prev: t0 }); }
function leaveFreeze() { T.frozen = false; T.leftAt = T.session; T.leftK = real(clamp(Math.round(T.t * 10), 0, M.n_paint - 1)); }
function leaveHold() { T.hold = false; leaveFreeze(); }
const holdAmt = () => T.hold ? sstep(0, .22, T.session - T.holdFrom) : T.session >= T.leftAt ? 1 - sstep(0, .22, T.session - T.leftAt) : 0;
const Beat = { k: -1, on: false, done: false, until: 0, dur: +(Q.get('beats') ?? 5),
  arm(id, meta) { this.k = meta.beat_k ?? BEATS[id] ?? -1; this.on = false; this.done = RENDER && Q.get('beat') !== '1'; },
  step(dt) {
    if (this.on) { if (T.session >= this.until && !K.space) this.end(); return; }
    if (this.k < 0 || this.done || T.frozen || T.sheet < 4.5) return;
    const at = this.k / 10;
    if (T.t <= at && T.t + dt >= at) {
      enterHold(at);
      Agent.stop();
      Object.assign(this, { on: true, until: T.session + this.dur });
    }
  },
  end() { if (this.on) { this.on = false; this.done = true; leaveHold(); } },
  pick() { enterHold(clamp(Math.round(T.t * 10), 0, M.n_paint - 1) / 10); Object.assign(this, { on: true, done: true, until: T.session + 6 }); }
};

function sim(dt) {
  Beat.step(dt);
  T.prev = T.t; T.session += dt; T.sheet += dt;
  const ranges = [];
  if (!T.frozen) {
    T.t += dt;
    if (T.t >= M.seconds) {
      ranges.push([T.prev, M.seconds]);
      T.t -= M.seconds; T.sheet = 1.6; T.prev = 0; flip = 0; clearPaper(); Agent.wrap();
    }
    ranges.push([T.prev, T.t]);
  } else if (dt > 0) {
    T.frozenClock = (T.frozenClock + dt) % 0.1;
    const base = T.frozenAt - 0.05 + T.frozenClock, a = clamp(base, 0, M.seconds - dt);
    ranges.push([a, a + dt]);
  }
  Agent.tick();
  if (dt > 0 || ranges.length) Paper.step(Math.max(dt, 1e-4), ranges);
  syncMask();
}

function syncMask() {
  const k = Math.floor(T.t * 10);
  const m = M.people.get(k);
  const id = m ? k : -1;
  if (id === curMask) return;
  curMask = id;
  gl.bindTexture(gl.TEXTURE_2D, maskTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 160, 120, gl.RED, gl.UNSIGNED_BYTE, m || M.zero);
}

function nearest(kind, k) {
  const a = M.tx[kind], n = kind === 'paint' ? M.nPic : M.n_paint;
  if (a[k]) return a[k];
  if (kind === 'cam') return blackTex;
  for (let d = 1; d < n; d++) { if (a[k - d]) return a[k - d]; if (a[k + d]) return a[k + d]; }
  return blackTex;
}

const ID3 = [1, 0, 0];
function real(k) { const f = M.filled; while (k > 0 && f.has(k)) k--; return k; }
const warp = (q, w) => q && DRIFT > 0 ? [Math.pow(q[0], w * DRIFT), q[1] * w * DRIFT, q[2] * w * DRIFT] : ID3;
function keyList() {
  if (M.chosen) return M.chosen;
  const d = M.drift, st = d ? d.stride : ST, g = Math.max(1, Math.round(ST / st)), keys = [], pairs = [];
  for (let k = 0; k < M.n_paint; k += st * g) keys.push(k);
  if (d) for (let i = 0; i + 1 < keys.length; i++) {
    let q = [1, 0, 0];
    for (let j = 0; j < g; j++) { const c = d.pairs[i * g + j]; if (c) q = [q[0] * c[0], q[1] + c[1], q[2] + c[2]]; }
    pairs.push(q);
  }
  return { keys, pairs };
}

function paintPair() {
  if (M.pf > 10) { const a = clamp(Math.floor(T.t * M.pf), 0, M.nPic - 1); return { a, b: a, m: 0, da: ID3, db: ID3, sa: 0, sb: 0 }; }
  const { keys, pairs } = M.kl || (M.kl = keyList()), tk = T.t * 10;
  let i = 0;
  while (i < keys.length - 2 && keys[i + 1] <= tk) i++;
  const ka = keys[i], kb = keys[i + 1] ?? ka, a = real(ka), b = real(kb), q = pairs[i], len = Math.max(kb - ka, 1), u = clamp((tk - ka) / len), h = XF * 5 / len;
  const base = { a, b, m: sstep(.5 - h, .5 + h, u), da: warp(q, u), db: warp(q, u - 1), sa: sstep(0, .4, u), sb: sstep(0, .4, 1 - u) };
  const exact = real(clamp(Math.round(tk), 0, M.n_paint - 1)), front = base.m < .5 ? [a, base.da] : [b, base.db];
  if (T.frozen) { const m = sstep(0, .3, T.session - T.holdFrom); return { a: front[0], b: exact, m, da: front[1], db: ID3, sa: 0, sb: 0 }; }
  const back = T.session - T.leftAt;
  if (back >= 0 && back < .3) return { a: T.leftK, b: front[0], m: sstep(0, .3, back), da: ID3, db: front[1], sa: 0, sb: 0 };
  return base;
}

function setMask(tex, cur, k) {
  if (M[cur] === k) return;
  M[cur] = k;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 160, 120, gl.RED, gl.UNSIGNED_BYTE, M.people.get(k) || M.zero);
}

function draw(view, dev) {
  const k = clamp(Math.floor(T.t * 10), 0, M.n_paint - 1), k1 = Math.min(k + 1, M.n_paint - 1);
  const f = T.frozen ? 0 : clamp(T.t * 10 - k);
  const pp = paintPair(), live = T.live && T.live.tex;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, fit.W, fit.H);
  gl.useProgram(V.p);
  const bind = (unit, tex, name) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(V.u[name], unit); };
  bind(0, paperTex[flip], 'P');
  bind(1, live || nearest('paint', pp.a), 'PA');
  bind(2, live || nearest('paint', pp.b), 'PB');
  gl.activeTexture(gl.TEXTURE8);
  setMask(maskA, 'curA', pp.a); setMask(maskB, 'curB', pp.b);
  bind(8, maskA, 'MA'); bind(9, maskB, 'MB');
  gl.uniform2f(V.u.swap, pp.sa, pp.sb);
  gl.uniform1f(V.u.m, pp.m);
  gl.uniform3fv(V.u.da, pp.da);
  gl.uniform3fv(V.u.db, pp.db);
  bind(4, nearest('back', k), 'BA');
  bind(5, T.frozen ? nearest('back', k) : nearest('back', k1), 'BB');
  bind(6, nearest('cam', Math.round(T.t * 10) >= M.n_paint ? k : Math.round(T.t * 10)), 'C');
  gl.uniform2f(V.u.res, fit.W, fit.H);
  gl.uniform2f(V.u.org, fit.ox, fit.oy);
  gl.uniform1f(V.u.sc, fit.sc);
  gl.uniform1f(V.u.f, f);
  gl.uniform1f(V.u.dev, dev);
  gl.uniform1f(V.u.mode, view);
  bind(7, Fx.s.img || blackTex, 'RP');
  bind(10, Fx.tex, 'FE');
  const q = Fx.s;
  gl.uniform4f(V.u.fx, q.dim, q.flare, q.dev, q.mark);
  gl.uniform4fv(V.u.sbox, q.sb);
  gl.uniform4fv(V.u.dbox, q.db);
  gl.uniform4fv(V.u.fbox, Fx.box || q.sb);
  gl.uniform1f(V.u.events, (view === 2 ? 0.5 : 1) * (1 - holdAmt()));
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 3);
  if (view !== 1) drawPoints(dev, view);
}

function drawPoints(dev, view) {
  const { sc, ox, oy, W, H } = fit;
  evUniforms(0, [2 * sc / W, -2 * sc / H, 2 * ox / W - 1, 1 - 2 * oy / H]);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.uniform1f(E.u.size, Math.max(1, sc * 0.62));
  gl.uniform1f(E.u.fade, 1.6 * (1 - holdAmt()) + 1e-3);
  gl.uniform1f(E.u.dim, (view === 2 ? 0.28 : 1) * (1 - 0.76 * dev));
  gl.uniform2f(E.u.win, 0, 1.001);
  const now = (T.frozen ? T.frozenAt : T.t) * 30, cur = Math.min(Math.floor(now), M.n_bins - 1);
  for (let b = cur - 1; b <= cur; b++) {
    if (b < 0) continue;
    const r = bindBin(b); if (!r) continue;
    const age = now - b - 1;
    gl.uniform1f(E.u.age, T.frozen ? 0.2 : age);
    gl.uniform2f(E.u.win, 0, T.frozen ? 1.001 : clamp(now - b, 0, 1.001));
    gl.drawArrays(gl.POINTS, r.first, r.n);
  }
  gl.disable(gl.BLEND);
}

const Agent = {
  steps: [], seen: new Set(), until: 0,
  load(tr) {
    this.steps = []; this.seen.clear(); this.auto = true; this.until = 0; this.holding = false; this.dimCue = null; Fx.own = [];
    if (!tr) { Fx.imgs = {}; Fx.sk = ''; Fx.reset(); }
    if (!tr || !tr.steps) return;
    let k = 0;
    for (const s of tr.steps) { const a = s.args || {}; if (a.k != null) k = a.k; this.steps.push({ ...s, a, t: k / 10 }); if (s.result && s.result.image) Promise.resolve(Fx.img(s.result.image)).catch(() => {}); }
  },
  ticks() { return [...new Set(this.steps.filter(s => s.a.k != null).map(s => s.t))]; },
  wrap() { if (this.steps.every(s => this.seen.has(s.i))) this.seen.clear(); this.stop(); },
  aim(s) {
    const q = s.result || {};
    if (s.tool === 'compare') { const b = (q.camera_no_painted || []).filter(e => e.support && e.support.backed); return b.length ? b[0].box : null; }
    if (s.tool === 'repaint') return q.image && q.aimed_at ? q.aimed_at.box : null;
    if (s.tool === 'look' && s.a.x != null) {
      if (q.left_out_here && q.left_out_here[0]) return q.left_out_here[0].box;
      const r = this.rep(s.t), b = r && r.result.aimed_at.box;
      if (b && s.a.x >= b[0] && s.a.x <= b[2] && s.a.y >= b[1] && s.a.y <= b[3]) return b;
    }
    return null;
  },
  rep(t) { return this.steps.find(o => o.t === t && o.tool === 'repaint' && o.result.image); },
  ok(s) {
    const b = this.aim(s), r = this.rep(s.t), a = r && r.result.aimed_at.box;
    return !!b && (!r || (b[0] < a[2] && a[0] < b[2] && b[1] < a[3] && a[1] < b[3]));
  },
  fire(s) {
    this.seen.add(s.i);
    const q = s.result || {}, b = this.aim(s), t0 = T.session, k = Math.round(s.t * 10);
    const add = (d, o) => { Fx.own.push({ at: t0, to: t0 + d, ...o }); return d; };
    const line = { text: s.reason, box: b };
    if (q.image) this.loadRepaint(q.image);
    if (!b || !this.ok(s)) { if (!this.rep(s.t)) add(3.6, { line: { text: s.reason } }); return false; }
    if (!T.hold) { enterHold(s.t); this.holding = true; }
    let d;
    const win = s.a.window_ms || 800;
    if (!this.dimCue && !(s.tool === 'compare' && s.a.concept === 'person')) Fx.own.push(this.dimCue = { at: t0, to: 1e9, spot: { k, box: b } });
    if (s.tool === 'compare' && s.a.concept === 'person') d = add(3.4, { line, k, win: 300, stripmarks: [s.t], stripred: q.camera_no_painted.some(e => e.support.backed) ? [s.t] : [], marks: q.camera_no_painted.filter(e => e.support.backed).map(e => ({ box: e.box })) });
    else if (s.tool === 'compare') d = add(2.8, { line, stripmarks: [s.t] });
    else if (s.tool === 'look' && s.a.view === 'camera') d = add(2, { line, view: 1 });
    else if (s.tool === 'look') d = add(3.6, { line, flare: b, win, grow: 2.7 });
    else d = add(6.4, { line, flare: b, win: 800, grow: .1, develop: { image: q.image, box: [b[0], b[1] - 78, b[2] + 15, b[3] + 21], over: 3.8, out: .9 } });
    this.until = t0 + d;
    return true;
  },
  loadRepaint(img) { Promise.resolve(Fx.img(img)).then(t => { this.rpTex = t; }).catch(() => {}); },
  stop() { this.until = 0; this.dimCue = null; Fx.own = []; if (this.holding) { this.holding = false; leaveHold(); } },
  next() {
    if (this.auto) for (const s of this.steps) if (Math.abs(s.t - T.frozenAt) < 1e-6 && !this.seen.has(s.i) && this.ok(s) && this.fire(s)) return;
    this.holding = false; leaveHold();
    if (this.dimCue) { this.dimCue.to = T.session + .8; this.dimCue = null; }
  },
  tick() {
    if (!M) return;
    const a0 = T.prev, a1 = T.t;
    if (!T.frozen && a1 > a0 && this.auto && T.sheet >= 3.4) for (const s of this.steps) {
      if (s.t < a0 || s.t >= a1 || this.seen.has(s.i)) continue;
      if (this.fire(s)) break;
    }
    if (this.until && T.session >= this.until) { this.until = 0; if (this.holding) this.next(); Fx.own = Fx.own.filter(c => c.to > T.session); }
  }
};

function loadImg(kind, dir, k, tok) {
  const a = M.tx[kind];
  if (a[k] || M.req[kind][k]) return M.req[kind][k] || Promise.resolve();
  const url = `moments/${M.id}/${dir}/${String(k).padStart(4, '0')}.webp`;
  const me = M;
  return M.req[kind][k] = fetch(url).then(r => r.ok ? r.blob() : null).then(b => b && createImageBitmap(b)).then(bm => {
    if (!bm) return;
    if (tok !== loadToken) { bm.close(); return; }
    me.pending.push({ kind, k, bm });
    if (RENDER) flushUploads(1e9);
  }).catch(() => {});
}

function flushUploads(n) {
  while (n-- > 0 && M && M.pending.length) {
    const { kind, k, bm } = M.pending.shift();
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bm);
    for (const [p, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, p, v);
    bm.close();
    M.tx[kind][k] = t;
  }
}

async function preload(tok) {
  const order = [], n = M.n_paint, seen = new Set();
  for (const st of [8, 4, 2, 1]) for (let k = 0; k < n; k += st) if (!seen.has(k)) { seen.add(k); order.push(k); }
  const jobs = [];
  const po = [], ps = new Set();
  for (const st of [24, 12, 6, 3, 1]) for (let k = 0; k < M.nPic; k += st) if (!ps.has(k)) { ps.add(k); po.push(k); }
  for (const k of po) jobs.push(['paint', 'paint', k]);
  for (const k of order) { jobs.push(['back', 'backing', k]); jobs.push(['cam', 'cam', k]); }
  let i = 0;
  const worker = async () => {
    while (i < jobs.length && tok === loadToken) {
      const [a, b, c] = jobs[i++];
      await loadImg(a, b, c, tok);
    }
  };
  await Promise.all([1, 2, 3, 4, 5].map(worker));
}

function decodeMask(frame) {
  const m = new Uint8Array(19200);
  for (const p of frame.people || []) {
    let i = 0, v = 0;
    for (const run of p.mask_rle || []) { if (v) m.fill(255, i, Math.min(i + run, 19200)); i += run; v = 1 - v; }
  }
  return m;
}

async function streamEvents(me, url, tok) {
  const total = me.idx[me.n_bins];
  me.ev = new Uint32Array(total);
  const bytes = new Uint8Array(me.ev.buffer);
  const r = await fetch(url);
  if (!r.ok) throw new Error('events ' + r.status);
  const rd = r.body.getReader();
  let at = 0;
  for (;;) {
    const { done, value } = await rd.read();
    if (done || tok !== loadToken) break;
    bytes.set(value, at); at += value.length;
    me.have = Math.floor(at / 4);
  }
  if (tok === loadToken && at !== bytes.length) throw new Error('incomplete events');
}

async function loadMoment(id, opts = {}) {
  const tok = ++loadToken;
  Dom.sa.textContent = Dom.sb.textContent = Dom.layer.textContent = ''; Dom.lastStamp = Dom.layerName = '';
  const response = await fetch(`moments/${id}/meta.json`);
  if (!response.ok) throw new Error('metadata unavailable');
  const meta = await response.json();
  if (tok !== loadToken) return;
  if (M) for (const k of ['paint', 'back', 'cam']) M.tx[k].forEach(t => t && gl.deleteTexture(t));
  M = { ...meta, id, have: 0, tx: { paint: [], back: [], cam: [] }, req: { paint: [], back: [], cam: [] }, pending: [], people: new Map(), masks: new Map(), zero: new Uint8Array(19200), ev: new Uint32Array(0) };
  M.idx = FILM ? meta.film_index : meta.index;
  M.filled = new Set(meta.filled_paint ?? FILLED[id] ?? []);
  M.pf = meta.paint_fps || 10; M.nPic = Math.round(meta.seconds * M.pf);
  const c = M.counts, mean = c.reduce((a, b) => a + b, 0) / c.length;
  M.maxC = Math.max(...c);
  M.K = 0.55 / (mean * 30 / (W0 * H0) * TAU_R);
  let cap = 0; for (let b = 0; b < M.n_bins; b++) cap = Math.max(cap, M.idx[b + 1] - M.idx[b]);
  sizeSlots(cap);
  const t0 = opts.t ?? Math.min(OPEN_T, meta.seconds / 2);
  Object.assign(T, { t: t0, prev: t0, sheet: opts.sheet || 0, frozen: false, hold: false, live: null, bin: -1, primed: false });
  curMask = -2; flip = 0; clearPaper(); Beat.arm(id, meta);
  const ev = streamEvents(M, `moments/${id}/${FILM ? 'events_film.bin' : 'events.bin'}`, tok).catch(e => {
    if (tok === loadToken) assetFailure();
    if (RENDER) throw e;
  });
  fetch(`moments/${id}/people.json`).then(r => r.ok ? r.json() : null).then(j => {
    if (j && tok === loadToken) for (const fr of j.frames || []) if ((fr.people || []).length) { const m = decodeMask(fr); M.people.set(fr.k, m); M.masks.set(fr.k, m); }
  }).catch(() => {});
  const motion = Promise.all(['drift', 'keys'].map(n => fetch(`moments/${id}/${n}.json`).then(r => r.ok ? r.json() : null).then(j => { if (j && tok === loadToken) { M[n === 'keys' ? 'chosen' : 'drift'] = j; M.kl = null; } }).catch(() => {})));
  Agent.load(null);
  fetch(`moments/${id}/agent.json`).then(r => r.ok ? r.json() : null).then(j => { if (j && tok === loadToken) { Agent.load(j); drawStrip(true); } }).catch(() => {});
  Audit.load(null);
  fetch(`moments/${id}/audit.json`).then(r => r.ok ? r.json() : null).then(j => { if (j && tok === loadToken) { Audit.load(j); drawStrip(true); } }).catch(() => {});
  drawStrip(true);
  if (!RENDER) preload(tok);
  if (RENDER) { await ev; await motion; }
}

const strip = { c: $('strip'), lit: null, dim: null, bin: -1 };
function drawStrip(full) {
  if (!M) return;
  const c = strip.c, dpr = RENDER ? 1 : Math.min(devicePixelRatio || 1, 2), w = (RENDER ? 1920 : innerWidth) * dpr | 0, h = 22 * dpr | 0;
  if (!w) return;
  if (full || !strip.lit || c.width !== w) {
    c.width = w; c.height = h;
    const mk = (al) => {
      const o = document.createElement('canvas'); o.width = w; o.height = h;
      const x = o.getContext('2d');
      for (let b = 0; b < M.n_bins; b++) {
        const v = M.counts[b] / M.maxC, x0 = b / M.n_bins * w, x1 = (b + 1) / M.n_bins * w;
        x.fillStyle = `rgba(237,230,214,${(v * v * al).toFixed(3)})`;
        x.fillRect(x0, h * .45, Math.max(1, x1 - x0), h * .55);
      }
      return o;
    };
    strip.dim = mk(.34); strip.lit = mk(1);
    strip.ticks = Agent.ticks();
  }
  const x = c.getContext('2d'), cut = T.t / M.seconds * w;
  x.clearRect(0, 0, w, h);
  x.drawImage(strip.dim, 0, 0);
  x.drawImage(strip.lit, 0, 0, cut, h, 0, 0, cut, h);
  Audit.ticks(x, w, h, cut, dpr);
  x.fillStyle = '#EDE6D6';
  x.fillRect(cut, 0, dpr, h);
  for (const t of strip.ticks) x.fillRect(t / M.seconds * w, 0, dpr, h * .4);
  for (const t of Fx.s.strip) x.fillRect(t / M.seconds * w - dpr, 0, 3 * dpr, h);
  x.fillStyle = '#FF3B1D';
  for (const t of Fx.s.stripred) x.fillRect(t / M.seconds * w - dpr, 0, 3 * dpr, h);
  strip.nm = Fx.s.strip.length;
}

const CAP = {
  person1: 'Someone is there.',
  person2: 'The camera finds them. The event sensor draws them.',
  camera: 'The real camera. Same street.',
  dream: 'The sensor was silent here. This part is invented.',
  colour: 'The sensor records no colour. Every colour here is a guess.',
  saw: 'This is everything it saw.',
  hint1: COARSE ? 'Press and hold: the real camera.' : 'Hold Space: the real camera.',
  hint2: COARSE ? 'Two fingers: what it never saw.' : 'Hold D: what it never saw.'
};
const Story = { personAt: -1, usedCam: false, usedDream: false, liveLine: '', liveUntil: 0, autoT: 0 };
const who = (w, n) => `${n} ${n === 1 ? w : w === 'person' ? 'people' : w + 's'}`;
const Audit = {
  data: null, told: new Set(), seen: false, until: 0, zoom: 0, key: '', cx: $('marks').getContext('2d'),
  load(j) { this.data = j; this.seen = false; this.until = 0; },
  on() { return !!this.data && !RENDER && !Inspect.run && !T.live && !Agent.holding && !Fx.s.line && (T.hold || !!strip.drag); },
  frame() { return this.data.frames[clamp(Math.round((T.hold ? T.frozenAt : T.t) * 10), 0, this.data.frames.length - 1)]; },
  text() {
    const f = this.frame(), sum = o => Object.values(o).reduce((a, b) => a + b, 0), kinds = [...new Set([...Object.keys(f.camera), ...Object.keys(f.painted)])], n = f.left_out.length, left = {};
    for (const o of f.left_out) left[o.who] = (left[o.who] || 0) + 1;
    const one = kinds.length === 1;
    let s = `Camera ${one ? who(kinds[0], sum(f.camera)) : sum(f.camera)}, painting ${sum(f.painted)}. `;
    s += !n ? 'Nothing left out.' : one ? `Left out ${n}.` : `Left out ${Object.entries(left).map(([w, c]) => who(w, c)).join(', ')}.`;
    if (f.invented) s += ` ${f.invented} not in camera.`;
    if (f.left_out.some(o => o.activity)) s += ' Activity inside this box.';
    return s;
  },
  total() {
    if (!this.data || this.on() || Agent.holding || Fx.s.line) return '';
    if (!this.until) {
      if (!this.seen || this.told.has(M.id)) return '';
      this.told.add(M.id); this.until = T.session + 5;
    }
    if (T.session >= this.until) return '';
    const t = Object.values(this.data.totals), all = t.reduce((a, r) => a + r.left_out, 0);
    return `These ${M.seconds} seconds: ${all} sightings left out, ${this.data.totals.person.left_out} of them people.`;
  },
  ticks(x, w, h, cut, dpr) {
    if (!this.data) return;
    const fr = this.data.frames, top = h * .4, tw = Math.max(dpr, w / fr.length - dpr), most = Math.max(1, ...fr.map(f => f.left_out.length));
    fr.forEach((f, k) => {
      const c = f.left_out.length, people = f.left_out.filter(o => o.who === 'person').length, at = k / fr.length * w, th = c ? Math.max(dpr, c / most * top) : dpr, red = c ? th * people / c : 0;
      x.globalAlpha = at <= cut ? 1 : .55;
      x.fillStyle = '#EDE6D6';
      x.fillRect(at, top - th, tw, th - red);
      x.fillStyle = '#FF3B1D';
      x.fillRect(at, top - red, tw, red);
    });
    x.globalAlpha = 1;
  },
  step(dt) {
    const a = fit.crop, b = fit.whole;
    if (!a || !b) return;
    const want = this.on() ? 1 : 0;
    this.zoom = Math.abs(want - this.zoom) < .004 ? want : this.zoom + (want - this.zoom) * clamp(dt * 7);
    const z = sstep(0, 1, this.zoom);
    fit.sc = a.sc + (b.sc - a.sc) * z; fit.ox = a.ox + (b.ox - a.ox) * z; fit.oy = a.oy + (b.oy - a.oy) * z;
  },
  draw() {
    const on = this.on(), cw = innerWidth, ch = innerHeight, f = on ? this.frame() : null;
    const key = on ? `${this.data.frames.indexOf(f)}|${fit.sc.toFixed(3)}|${cw}|${ch}` : '';
    if (key === this.key) return;
    this.key = key;
    const c = this.cx.canvas, dpr = Math.min(devicePixelRatio || 1, 2);
    const pw = cw * dpr | 0, ph = ch * dpr | 0;
    if (c.width !== pw || c.height !== ph) { c.width = pw; c.height = ph; }
    this.cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cx.clearRect(0, 0, cw, ch);
    if (!on) return;
    this.seen = true;
    const cap = Dom.cap.getBoundingClientRect();
    for (const o of f.left_out) {
      const px = v => (fit.ox + v * fit.sc) * fit.s, py = v => (fit.oy + v * fit.sc) * fit.s, b = o.box, t = o.activity ? 5 : 2;
      const x0 = clamp(px(b[0]), 6, cw - 16), x1 = clamp(Math.max(px(b[2]), x0 + 10), x0 + 10, cw - 6);
      const under = x0 < cap.right && x1 > cap.left, floor = under ? cap.top - 8 : ch - 60;
      let y = py(b[3]) + 3;
      if (y + t > floor) y = py(b[1]) - t - 3;
      y = clamp(y, 6, floor - t);
      this.cx.fillStyle = '#000'; this.cx.fillRect(x0 - 1, y - 1, x1 - x0 + 2, t + 2);
      this.cx.fillStyle = o.who === 'person' ? '#FF3B1D' : '#EDE6D6'; this.cx.fillRect(x0, y, x1 - x0, t);
    }
  }
};
function storyCaption() {
  if (T.frozen && Story.liveLine) return Story.liveLine;
  if (Story.liveLine && T.session < Story.liveUntil) return Story.liveLine;
  const tl = Audit.total();
  if (tl) return tl;
  const s = T.session;
  if (Beat.on) return Story.usedCam ? '' : CAP.hint1;
  if (s > .35 && s < 3.2) return CAP.saw;
  const k = Math.floor(T.t * 10);
  if (Story.personAt < 0 && M.people.has(k) && s > 3.2) Story.personAt = s;
  if (Story.personAt >= 0) {
    const d = s - Story.personAt;
    if (d < 2.6) return CAP.person1;
    if (d < 6.2) return CAP.person2;
  }
  if (s > 8 && !Story.usedCam && !autoView()) return CAP.hint1;
  if (Story.usedCam && !Story.usedDream && s > Story.camEnd + 1.2 && !autoView()) return CAP.hint2;
  if (Story.usedCam && Story.usedDream && !Story.colourDone) { Story.colourAt = Story.colourAt || s + 1.5; if (s > Story.colourAt && s < Story.colourAt + 5) return CAP.colour; if (s >= Story.colourAt + 5) Story.colourDone = true; }
  return '';
}

function autoView() {
  if (!COARSE || RENDER || Pointer.n > 0) return 0;
  const idle = T.session - Story.autoT;
  if (idle < 6) return 0;
  const a = (idle - 6) % 18;
  if (a < 3.5) return 1;
  if (a >= 9 && a < 12.5) return 2;
  return 0;
}

const cues = () => window.__cues || [];
function renderCue(vt) {
  let cap = '', view = 0, kc = null;
  for (const c of cues()) if (vt >= c.at && vt < c.to) {
    if (c.text) cap = c.text;
    if (c.key === 'space') view = 1;
    if (c.key === 'd') view = 2;
    if (c.keycap) kc = { wide: (c.key || c.keycap) === 'space', down: !!c.key };
  }
  return { cap, view, kc };
}

const Dom = { sa: $('sa'), sb: $('sb'), layer: $('layer'), cap: $('caption'), reason: $('reason'), lastStamp: '', layerName: '', layerUntil: 0, lastCap: null, reasonKey: null };
const quiet = Object.assign(document.createElement('canvas'), { width: 64, height: 36 }).getContext('2d', { willReadFrequently: true });
function quietSpot(pl) {
  const el = Dom.reason, w = el.offsetWidth, h = el.offsetHeight, cw = RENDER ? 1920 : innerWidth, ch = RENDER ? 1080 : innerHeight, floor = Dom.cap.getBoundingClientRect().top - 12;
  quiet.drawImage(canvas, 0, 0, 64, 36);
  const d = quiet.getImageData(0, 0, 64, 36).data, px = pl ? (fit.ox + pl.x * fit.sc) * fit.s : 0, py = pl ? (fit.oy + pl.y * fit.sc) * fit.s : 0, pr = pl ? pl.r * fit.sc * fit.s + 24 : 0;
  let best = [16, 16], low = 1e9;
  for (const fy of [.03, .14, .25, .36, .47, .58, .69]) for (const fx of [0, .25, .5, 1]) {
    const x = Math.round(16 + fx * (cw - w - 32)), y = Math.round(fy * ch);
    if (y + h > floor) continue;
    if (pl && px + pr > x && px - pr < x + w && py + pr > y && py - pr < y + h) continue;
    let sum = 0, top = 0, n = 0;
    for (let v = y * 36 / ch | 0; v <= Math.min(35, (y + h) * 36 / ch); v++) for (let u = x * 64 / cw | 0; u <= Math.min(63, (x + w) * 64 / cw); u++) {
      const i = (v * 64 + u) * 4, l = (d[i] + d[i + 1] + d[i + 2]) / 765;
      sum += l; top = Math.max(top, l); n++;
    }
    const cost = top + sum / Math.max(n, 1) + (pl ? Math.hypot(x + w / 2 - px, y + h / 2 - py) / cw * .05 : 0);
    if (cost < low) { low = cost; best = [x, y]; }
  }
  el.style.transform = `translate(${best[0]}px,${best[1]}px)`;
}
function hud(view, dev, capText) {
  const live = T.live || (T.frozen && !T.hold);
  let layer = view === 1 ? 'Recorded camera, brightened' : view === 2 ? 'Muse Image painting' : dev < .4 ? 'Measured events' : 'Muse Image painting';
  if (view === 1 && !M.tx.cam[clamp(Math.round(T.t * 10), 0, M.n_paint - 1)]) {
    layer = 'Recorded camera unavailable';
    capText = 'Camera frame unavailable at this time.';
  }
  if (T.frozen && !T.hold && !T.live?.tex && view === 0) layer = 'Measured events';
  if (Audit.on() && view !== 1) layer = 'Muse Image painting, smoothed';
  const us = M.t0_us + Math.round((T.frozen ? T.frozenAt : T.t) * 1e6);
  const s = `${M.place || ''}  ${M.seq}  ${us} µs  `;
  if (s !== Dom.lastStamp) { Dom.sa.textContent = s; Dom.lastStamp = s; }
  if (layer !== Dom.layerName) { Dom.layer.textContent = Dom.layerName = layer; Dom.layerUntil = T.session + 2.8; }
  Dom.layer.classList.toggle('big', T.session < Dom.layerUntil);
  const mode = Inspect.run ? Inspect.stamp() : `  ${live ? 'Running live' : 'Recorded'}`;
  if (Dom.sb.textContent !== mode) Dom.sb.textContent = mode;
  if (capText !== Dom.lastCap) { Dom.cap.textContent = capText; Dom.lastCap = capText; }
  const ln = Fx.s.line, key = ln ? ln.text : null, el = Dom.reason;
  if (key !== Dom.reasonKey) {
    Dom.reasonKey = key;
    el.textContent = key || '';
    el.style.visibility = key ? 'visible' : 'hidden';
    if (key) { const b = ln.box; quietSpot(b ? { x: (b[0] + b[2]) / 2, y: (b[1] + b[3]) / 2, r: Math.max(b[2] - b[0], b[3] - b[1]) * .72 + 26 } : null); }
  }
}

function devAmount() {
  if (T.live && T.live.tex) return sstep(0, 3.2, T.session - T.live.from);
  if (T.frozen && !T.hold) return 0;
  return sstep(.75, 4.1, T.sheet);
}

const Sound = (() => {
  let ctx = null, out = null, on = true, last = -1;
  const el = $('sound');
  function start() {
    if (ctx || RENDER) { if (ctx && ctx.state === 'suspended') ctx.resume(); return; }
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    out = ctx.createGain(); out.gain.value = on ? 1 : 0;
    const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 900;
    hp.connect(ctx.destination); out.connect(hp);
    ctx.click = ctx.createGain(); ctx.click.gain.value = on ? 1 : 0; ctx.click.connect(ctx.destination);
  }
  function grains(b, gain) {
    if (!ctx || !M.have || !on) return;
    const a = M.idx[b], n = Math.min(M.idx[b + 1], M.have) - a;
    if (n <= 0) return;
    const v = M.counts[b] / M.maxC, sr = ctx.sampleRate, len = Math.ceil(sr / 30) + 64;
    const count = v < .12 ? 0 : Math.round(v * v * v * 110);
    if (!count) return;
    let sx = 0;
    const samp = Math.max(1, n >> 7);
    let m = 0; for (let i = 0; i < n; i += samp) { sx += M.ev[a + i] & 1023; m++; }
    const pan = clamp(sx / m / W0 * 2.2 - .6), L = Math.cos(pan * 1.57), R = Math.sin(pan * 1.57);
    const buf = ctx.createBuffer(2, len, sr), l = buf.getChannelData(0), r = buf.getChannelData(1);
    for (let g = 0; g < count; g++) {
      const e = M.ev[a + (g * n / count | 0)], at = (e >>> 20) / 4096 * (len - 64) | 0, amp = (.18 + ((e & 255) / 255) * .5) * gain;
      for (let i = 0; i < 48; i++) { const s = amp * (Math.random() * 2 - 1) * Math.exp(-i / 9); l[at + i] += s * L; r[at + i] += s * R; }
    }
    const src = ctx.createBufferSource(); src.buffer = buf; src.connect(out); src.start(ctx.currentTime + .03);
  }
  function click(down) {
    if (!ctx || !on) return;
    const sr = ctx.sampleRate, n = sr * .05 | 0, buf = ctx.createBuffer(1, n, sr), d = buf.getChannelData(0);
    const f = down ? 1900 : 2600;
    for (let i = 0; i < n; i++) {
      const t = i / sr;
      d[i] = (.5 * (Math.random() * 2 - 1) * Math.exp(-t / .0016) + .5 * Math.sin(6.283 * f * t) * Math.exp(-t / .004) + (i > sr * .009 ? .3 * Math.sin(6.283 * 320 * t) * Math.exp(-(t - .009) / .006) : 0));
    }
    const s = ctx.createBufferSource(); s.buffer = buf; s.connect(ctx.click); s.start();
  }
  function frame(view) {
    if (!ctx || !M || T.frozen) return;
    const b = Math.floor(T.t * 30);
    if (b !== last) { last = b; grains(b, 1); }
    out.gain.setTargetAtTime(!on || view === 1 ? 0 : view === 2 ? .12 : 1, ctx.currentTime, .004);
  }
  el.onclick = () => { start(); on = !on; el.classList.toggle('off', !on); if (ctx) { ctx.click.gain.value = on ? 1 : 0; if (on) ctx.resume(); } };
  return { start, click, frame };
})();

const Live = {
  async press() {
    const b = $('paint');
    if (T.live) { this.done(); return; }
    Beat.end(); Beat.done = true; Agent.stop();
    Sound.click(true);
    T.frozen = true; T.holdFrom = T.session; T.frozenAt = Math.floor(T.t * 10) / 10; T.frozenClock = 0; T.t = T.frozenAt;
    T.live = { tex: null, from: 0, started: performance.now(), controller: new AbortController() };
    const run = T.live;
    b.textContent = 'waiting';
    b.disabled = true;
    const tick = () => { if (T.live === run && !run.tex) { Story.liveLine = `Muse Image  ${((performance.now() - run.started) / 1000).toFixed(1)} s`; setTimeout(tick, 100); } };
    tick();
    try {
      const deadline = new Promise((_, reject) => {
        run.timer = setTimeout(() => { run.controller.abort(); reject(new Error('live deadline')); }, 60000);
      });
      const j = await Promise.race([(async () => {
        const r = await fetch('/api/paint', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ moment: M.id, k: Math.round(T.frozenAt * 10) }), signal: run.controller.signal });
        if (!r.ok) throw new Error('paint unavailable');
        const j = await r.json();
        if (j.error || !j.image) throw new Error(j.error || 'no image');
        if (T.live !== run) return;
        const image = await fetch(j.image, { signal: run.controller.signal });
        if (!image.ok) throw new Error('image unavailable');
        const bm = await createImageBitmap(await image.blob());
        if (T.live !== run || run.controller.signal.aborted) { bm.close(); return; }
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bm);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        bm.close();
        run.tex = t; run.from = T.session;
        return j;
      })(), deadline]);
      if (T.live !== run) return;
      Story.liveLine = `Muse Image  ${(j.seconds != null ? +j.seconds : (performance.now() - run.started) / 1000).toFixed(1)} s`;
      b.textContent = 'continue'; b.disabled = false;
    } catch (e) {
      if (T.live !== run) return;
      Story.liveLine = 'Muse Image did not answer. Back to the recorded painting.';
      Story.liveUntil = T.session + 3.5;
      this.done(true);
    } finally {
      clearTimeout(run.timer);
    }
  },
  done(keepLine) {
    const b = $('paint');
    if (T.live) { clearTimeout(T.live.timer); T.live.controller.abort(); }
    if (T.live && T.live.tex) gl.deleteTexture(T.live.tex);
    T.live = null; leaveFreeze();
    if (!keepLine) Story.liveLine = '';
    T.sheet = Math.max(T.sheet, 4.5);
    b.textContent = 'paint this moment'; b.disabled = false;
    Sound.click(false);
  },
  async probe() {
    try {
      const r = await fetch('/api/status');
      if (r.ok && (await r.json()).live && M) {
        const b = $('paint'), i = $('inspect');
        b.hidden = i.hidden = false;
        b.onclick = e => { e.stopPropagation(); Sound.start(); this.press(); };
        i.onclick = e => { e.stopPropagation(); Sound.start(); Inspect.press(); };
      }
    } catch (e) {}
  }
};

const Inspect = {
  run: null, line: '', log: [],
  press() {
    if (this.run) { this.leave(); return; }
    if (T.live) Live.done();
    const k = clamp(Math.round(T.t * 10), 0, M.n_paint - 1);
    Beat.end(); Beat.done = true; Agent.stop(); Agent.auto = false; Fx.reset();
    T.sheet = Math.max(T.sheet, 4.5);
    Sound.click(true);
    enterHold(k / 10);
    const now = performance.now();
    const run = this.run = { k, id: M.id, started: now, heard: now, steps: 0, queue: [], busy: 0, dim: null, flare: null, end: null, finished: 0, model: 'muse-spark-1.3', control: new AbortController(), textures: [] };
    this.log = [{ ms: 0, kind: 'press', k, moment: M.id }];
    $('paint').hidden = true;
    $('inspect').textContent = 'stop';
    this.line = `Muse Spark is choosing what to check at ${(k / 10).toFixed(1)} s.`;
    this.listen(run);
  },
  async listen(run) {
    try {
      const r = await fetch(`/api/agent?moment=${encodeURIComponent(run.id)}&k=${run.k}`, { signal: run.control.signal });
      if (!r.ok) { const j = await r.json().catch(() => ({})); this.fail(run, j.error); return; }
      const rd = r.body.getReader(), dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        for (let i; (i = buf.indexOf('\n\n')) >= 0; buf = buf.slice(i + 2)) if (buf.startsWith('data: ')) this.hear(run, JSON.parse(buf.slice(6, i)));
      }
      if (this.run === run && !run.queue.some(m => m.end) && !run.end) this.fail(run);
    } catch (e) { if (this.run === run) this.fail(run); }
  },
  hear(run, m) {
    if (this.run !== run) return;
    run.heard = performance.now();
    if ('beat' in m) return;
    const { image, result, ...rest } = m;
    this.log.push({ ms: Math.round(run.heard - run.started), kind: m.tool || (m.start ? 'start' : m.end ? 'end' : m.error ? 'error' : 'done'), ...rest, result, image: image ? image.length : undefined });
    if (m.error) this.fail(run);
    else if (!m.done) run.queue.push(m);
  },
  fail(run, line) {
    if (this.run !== run) return;
    this.log.push({ ms: Math.round(performance.now() - run.started), kind: 'fallback', line: line || '' });
    Story.liveLine = line || (Agent.steps.length ? 'The live check did not finish. Back to the recorded run.' : 'The live check did not finish. The drive plays on.');
    Story.liveUntil = T.session + 4;
    this.leave();
  },
  leave() {
    const run = this.run;
    if (!run) return;
    this.run = null; this.line = '';
    run.control.abort();
    Fx.own = []; Fx.reset();
    for (const key of run.textures) { if (Fx.imgs[key] instanceof WebGLTexture) gl.deleteTexture(Fx.imgs[key]); delete Fx.imgs[key]; }
    Agent.auto = true;
    if (T.hold) leaveHold();
    $('inspect').textContent = 'inspect this moment';
    $('paint').hidden = false;
    Sound.click(false);
  },
  stamp() {
    const r = this.run, ms = (r.finished || performance.now()) - r.started;
    return `  Running live  ${r.model}  step ${r.steps} of 8  ${(ms / 1000).toFixed(0)} s`;
  },
  say(run, line, dwell) {
    this.line = line;
    run.busy = T.session + dwell;
    this.log.push({ ms: Math.round(performance.now() - run.started), kind: 'line', line });
  },
  aim(run, box) {
    if (run.dim) run.dim.to = T.session + .6;
    run.dim = { at: T.session, to: 1e9, stripmarks: [run.k / 10], spot: { k: run.k, box } };
    Fx.own.push(run.dim);
  },
  flare(run, box, win, grow) {
    if (run.flare) run.flare.to = T.session;
    run.flare = { at: T.session, to: 1e9, k: run.k, flare: box, win, grow };
    Fx.own.push(run.flare);
  },
  quiet(run) {
    if (run.dim) { run.dim.to = T.session + .8; run.dim = null; }
    if (run.flare) { run.flare.to = T.session + .8; run.flare = null; }
  },
  compare(run, s) {
    const rows = Object.entries(s.result.by_concept || {}), left = [];
    let cam = 0, painted = 0;
    for (const [who, v] of rows) { cam += v.counts.camera; painted += v.counts.painted; for (const o of v.camera_no_painted) left.push({ who, ...o }); }
    left.sort((a, b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]));
    this.say(run, `It checks ${rows.map(r => r[0]).join(', ')}. The camera finds ${cam}, the painting ${painted}.`, 2.8);
    run.queue.unshift({ left });
  },
  left(run, left) {
    if (!left.length) { this.say(run, 'Nothing the camera found is left out here.', 1.6); return; }
    const n = {};
    for (const o of left) n[o.who] = (n[o.who] || 0) + 1;
    const said = Object.entries(n).map(([w, c]) => `${c} ${c > 1 ? (w === 'person' ? 'people' : w + 's') : w}`).join(', ');
    this.aim(run, left[0].box);
    const people = left.filter(o => o.who === 'person');
    if (people.length && M.masks.has(run.k)) Fx.own.push({ at: T.session, to: 1e9, k: run.k, win: 300, marks: people.map(o => ({ box: o.box })) });
    this.say(run, `Left out of the painting: ${said}.`, 2.6);
  },
  look(run, s) {
    const q = s.result;
    if (s.a.view === 'camera') {
      Fx.own.push({ at: T.session, to: T.session + 2.4, view: 1 });
      this.say(run, 'The camera at this spot.', 2.4);
      return;
    }
    if (s.a.view !== 'events') { run.busy = T.session + .3; return; }
    const off = o => Math.hypot((o.box[0] + o.box[2]) / 2 - q.x, (o.box[1] + o.box[3]) / 2 - q.y);
    const h = (q.left_out_here || []).slice().sort((a, b) => off(a) - off(b))[0], win = q.window_ms;
    const box = h ? h.box : q.x != null ? [q.x - q.r, q.y - q.r, q.x + q.r, q.y + q.r] : null;
    if (box) { this.aim(run, box); this.flare(run, box, win, 2.7); }
    const events = (h ? h.events : q.events).toLocaleString('en-US'), times = h ? h.vs_median : q.vs_median;
    const word = h && !h.held ? 'Too little activity inside this box' : 'Activity inside this box';
    this.say(run, `${word}: ${events} events over ${win} ms, ${times} times the frame's middle.`, 3);
  },
  async repaint(run, s) {
    const q = s.result;
    if (q.error) { this.say(run, 'No repaint came back from that order.', 2); return; }
    if (q.withheld) { this.say(run, q.withheld.includes('facing') ? `This repaint of the ${q.aimed_at.who} faces another way than the camera shows, so it is not shown.` : `This repaint broke a rule for published paintings (${q.withheld.join(', ')}), so it is not shown.`, 3); return; }
    if (q.unshown) { this.say(run, `This patch of the ${q.aimed_at.who} failed the look check (${q.unshown}), so it is not shown.`, 3); return; }
    run.busy = Infinity;
    const key = `live/${run.k}.${q.version}`;
    try {
      const bm = await createImageBitmap(await (await fetch(s.image)).blob());
      if (this.run !== run) { bm.close(); return; }
      const t = tex2d(1, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bm);
      bm.close();
      Fx.imgs[key] = t; run.textures.push(key);
    } catch (e) { if (this.run === run) this.say(run, 'The repaint could not be shown.', 2); return; }
    const box = q.patch.box;
    const at = T.session, hold = q.helped ? 1e9 : at + 5.2;
    Fx.own.push({ at, to: hold, k: run.k, develop: { image: key, box, over: 3.8, out: q.helped ? 0 : .9 } });
    this.say(run, q.helped ? `The repaint develops: ${q.aimed_at.who}, placed by the camera's box.` : `This repaint of the ${q.aimed_at.who} did not pass the check.`, q.helped ? 4.2 : 5.2);
  },
  finish(run, m) {
    const e = m.end, gone = e.left_out;
    const text = e.state === 'verified' ? `Camera-guided repair verified: the camera gave the box, the ${e.who} now matches the camera, and nothing else was lost.`
      : e.state === 'evidence' ? `Not enough evidence to repaint: ${gone} left out, ${e.held} with activity inside the box.`
      : e.refused ? 'Unchanged: the painter refused this frame, so the painting stands.'
      : e.withheld && e.withheld.length ? 'Unchanged: the repaint was not shown, so the painting stands.'
      : e.repaints ? 'Unchanged: the repaint did not pass, so the painting stands.'
      : 'Unchanged: camera and painting agree on this frame.';
    if (e.state !== 'verified') this.quiet(run);
    run.end = e; run.finished = performance.now();
    this.line = text;
    this.log.push({ ms: Math.round(run.finished - run.started), kind: 'state', state: e.state, line: text });
    $('inspect').textContent = 'continue';
    Sound.click(false);
  },
  tick() {
    if (Story.liveLine && !T.live && !this.run && T.session >= Story.liveUntil) Story.liveLine = '';
    const run = this.run;
    if (!run || run.end) return;
    const now = performance.now();
    if (now - (run.ticked || now) > 5000) run.heard = now;
    run.ticked = now;
    if (!run.queue.length && now - run.heard > 30000) { this.fail(run); return; }
    if (T.session < run.busy || !run.queue.length) return;
    const m = run.queue.shift();
    if (m.left) { this.left(run, m.left); return; }
    if (m.end) { this.finish(run, m); return; }
    if (m.start) {
      if (m.start === 'repaint') this.say(run, `Repaint inside the camera's box: ${String(m.args.note || '').slice(0, 90)}`, .4);
      return;
    }
    run.steps++;
    Sound.click(true);
    const s = { ...m, a: m.args || {} };
    if (m.tool === 'compare' && !m.result.error) this.compare(run, s);
    else if (m.tool === 'look' && !m.result.error) this.look(run, s);
    else if (m.tool === 'repaint') this.repaint(run, s);
  }
};

function seek(t) {
  if (Inspect.run) Inspect.leave();
  Beat.end(); Agent.stop();
  T.t = T.prev = clamp(t, 0, M.seconds - 1e-3);
  T.sheet = Math.max(T.sheet, 4.5);
  flip = 0;
  preroll(T.t);
  drawStrip();
}

function setKey(name, val) {
  if (K[name] === val) return;
  K[name] = val;
  Sound.start();
  Sound.click(val);
  if (name === 'space' && val) Story.usedCam = true;
  if (name === 'space' && !val) { Story.camEnd = T.session; if (Beat.on) Beat.until = T.session + .7; }
  if (name === 'd' && val) Story.usedDream = true;
}

const Pointer = { n: 0, ids: new Set() };
function initInput() {
  addEventListener('keydown', e => {
    if (e.repeat && e.code !== 'Space') return;
    if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) setKey('space', true); }
    else if (e.code === 'KeyD') setKey('d', true);
    else if (e.code === 'ArrowRight' || e.code === 'ArrowLeft') { e.preventDefault(); shift(e.code === 'ArrowRight' ? 1 : -1); }
  });
  addEventListener('keyup', e => {
    if (e.code === 'Space') { e.preventDefault(); setKey('space', false); }
    else if (e.code === 'KeyD') setKey('d', false);
  });
  addEventListener('blur', () => { setKey('space', false); setKey('d', false); });
  const update = () => { Pointer.n = Pointer.ids.size; setKey('d', Pointer.n >= 2); setKey('space', Pointer.n === 1); Story.autoT = T.session; };
  canvas.addEventListener('pointerdown', e => { Pointer.ids.add(e.pointerId); canvas.setPointerCapture?.(e.pointerId); update(); });
  for (const n of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(n, e => { if (Pointer.ids.delete(e.pointerId)) update(); });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  const scrub = e => { const r = strip.c.getBoundingClientRect(); if (M) seek((e.clientX - r.left) / r.width * M.seconds); };
  strip.c.addEventListener('pointerdown', e => { strip.c.setPointerCapture?.(e.pointerId); strip.drag = true; Sound.start(); scrub(e); });
  strip.c.addEventListener('pointermove', e => { if (strip.drag) scrub(e); });
  strip.c.addEventListener('pointerup', () => { if (strip.drag && M) Beat.pick(); strip.drag = false; });
  let idleT;
  addEventListener('pointermove', () => { document.body.classList.remove('idle'); clearTimeout(idleT); idleT = setTimeout(() => document.body.classList.add('idle'), 2500); });
  addEventListener('resize', resize);
}

function shift(d) {
  if (Inspect.run) Inspect.leave();
  if (moments.length < 2 || T.live) { if (M && !T.live) seek((T.t + d * 1) % M.seconds); return; }
  momentIx = (momentIx + d + moments.length) % moments.length;
  loadMoment(moments[momentIx]).catch(assetFailure);
}

function resolveView() {
  if (K.space) return 1;
  if (K.d) return 2;
  return Fx.s.view || autoView();
}

const Perf = { last: 0, ema: 16, hold: 0 };
function loop(now) {
  requestAnimationFrame(loop);
  if (!M) return;
  const raw = Perf.last ? Math.max(0, now - Perf.last) : 16.67;
  Perf.last = now;
  Perf.ema += (Math.min(raw, 40) - Perf.ema) * .06;
  if (++Perf.hold > 40 && !document.hidden) {
    if (Perf.ema > 19.5) { budget = Math.max(.15, budget * .82); Perf.hold = 0; }
    else if (Perf.ema < 17.5 && budget < 1) { budget = Math.min(1, budget * 1.06); Perf.hold = 20; }
  }
  flushUploads(1);
  const stalled = M.have < M.idx[Math.min(Math.floor(T.t * 30) + 2, M.n_bins)];
  if (!stalled && !T.primed) { T.primed = true; preroll(T.t); }
  else if (!stalled) sim(Math.min(raw / 1000, .1));
  Inspect.tick();
  Fx.at(T.session, Fx.own);
  Audit.step(raw / 1000);
  const view = resolveView(), dev = devAmount();
  draw(view, dev);
  Sound.frame(view);
  const b = Math.floor(T.t * 30);
  if (b !== T.bin || Fx.s.strip.length !== strip.nm) { T.bin = b; drawStrip(); }
  const cap = Audit.on() ? Audit.text() : !K.space && !K.d && Inspect.line ? Inspect.line : view === 1 ? CAP.camera : view === 2 ? CAP.dream : storyCaption();
  hud(view, dev, cap);
  Audit.draw();
}

const OPEN_T = 1.2;
function initRender() {
  document.body.classList.add('render');
  const startT = +(Q.get('t') ?? OPEN_T), startSheet = Q.get('exposed') === '1' ? 9 : 0;
  let last = -1;
  const ready = (async () => {
    const id = Q.get('moment') || (await index())[0];
    await loadMoment(id, { t: startT, sheet: startSheet });
    preroll(startT);
    return true;
  })().catch(e => { assetFailure(); throw e; });
  ready.catch(() => {});
  window.__ready = ready;
  window.__step = async (frame, keys) => {
    await ready;
    if (frame <= last) {
      flip = 0; preroll(startT); curMask = -2; Agent.stop(); Agent.seen.clear(); Fx.reset();
      Object.assign(T, { t: startT, prev: startT, sheet: startSheet, session: 0, frozen: false, hold: false }); Beat.arm(M.id, M);
      last = -1;
    }
    const on = cues().some(c => c.spot || c.flare || c.develop || c.marks || c.stripmarks || c.line || c.hold != null);
    await Fx.prep(cues());
    Agent.auto = !on && !cues().some(c => c.agent != null);
    while (last < frame) {
      const vt = (last + 1) / 60;
      if (on) Fx.hold(vt, cues());
      for (const c of cues()) if (c.agent != null && vt >= c.at && Agent.steps[c.agent] && !Agent.seen.has(Agent.steps[c.agent].i)) Agent.fire(Agent.steps[c.agent]);
      sim(last < 0 ? 0 : 1 / 60); last++;
    }
    const k = clamp(Math.floor(T.t * 10), 0, M.n_paint - 1), k1 = Math.min(k + 1, M.n_paint - 1), kc = clamp(Math.round(T.t * 10), 0, M.n_paint - 1), pp = paintPair();
    await Promise.all([loadImg('paint', 'paint', pp.a, loadToken), loadImg('paint', 'paint', pp.b, loadToken), loadImg('back', 'backing', k, loadToken), loadImg('back', 'backing', k1, loadToken), loadImg('cam', 'cam', kc, loadToken)]);
    flushUploads(1e9);
    const vt = frame / 60, cue = renderCue(vt);
    if (on) Fx.at(vt, cues()); else Fx.at(T.session, Fx.own);
    let view = cue.view || Fx.s.view;
    if (keys) view = keys.space ? 1 : keys.d ? 2 : (keys.space === false && keys.d === false ? 0 : view);
    K.space = view === 1;
    const dev = devAmount();
    draw(view, dev);
    gl.finish();
    drawStrip();
    hud(view, dev, cue.cap);
    const kp = $('keycap'), kq = cue.kc;
    kp.hidden = !kq;
    if (kq) { kp.textContent = kq.wide ? 'Space' : 'D'; kp.classList.toggle('down', kq.down); kp.classList.toggle('wide', kq.wide); }
    return { t: T.t, view, dev, us: M.t0_us + Math.round(T.t * 1e6) };
  };
  window.__info = () => ({ seconds: M.seconds, frames: Math.round(M.seconds * 60), moment: M.id, cap: capN });
}

function assetFailure() {
  if (T.live) Live.done(true);
  loadToken++;
  M = null;
  $('paint').hidden = $('inspect').hidden = true;
  $('caption').textContent = 'Required recorded data is unavailable. Reload to try again.';
  $('sa').textContent = $('sb').textContent = $('layer').textContent = '';
  $('reason').style.visibility = 'hidden';
}

async function index() {
  const r = await fetch('moments/index.json');
  if (!r.ok) throw new Error('index unavailable');
  const list = (await r.json()).moments;
  if (!Array.isArray(list) || !list.length) throw new Error('empty index');
  return list;
}

function noGL() {
  const v = document.createElement('video');
  v.className = 'film'; v.src = 'film.mp4'; v.autoplay = v.muted = v.loop = v.playsInline = true; v.controls = false;
  v.onerror = () => { v.remove(); $('caption').textContent = 'WebGL2 is unavailable, and the fallback film is unavailable. Try a browser with WebGL2 enabled.'; };
  document.body.prepend(v);
  $('strip').remove(); canvas.remove();
}

const ST = +(Q.get('st') ?? 4), XF = +(Q.get('xf') ?? .3), DRIFT = +(Q.get('drift') ?? 1);
window.__fx = async (vt, list) => { await Fx.prep(list); Fx.hold(vt, list); Fx.at(vt, list); };
window.__dbg = () => ({ inspect: Inspect.log, line: Inspect.line, frozen: T.frozen, t: T.t, dev: devAmount(), budget, have: M && M.have, ema: Perf.ema, paint: M && M.tx.paint.filter(Boolean).length, K, view: resolveView() });

(async () => {
  if (!gl || Q.get('nogl') === '1') return noGL();
  initGL();
  resize();
  if (RENDER) { initRender(); return; }
  initInput();
  moments = await index();
  const want = Q.get('moment'); if (want && moments.includes(want)) momentIx = moments.indexOf(want);
  await loadMoment(moments[momentIx]);
  Live.probe();
  addEventListener('pointerdown', () => Sound.start(), { once: true });
  addEventListener('keydown', () => Sound.start(), { once: true });
  requestAnimationFrame(loop);
})().catch(assetFailure);
