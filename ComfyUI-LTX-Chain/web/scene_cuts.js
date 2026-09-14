// LTX Chain: Scene Cuts - waveform widget. Draws the LoadAudio file connected to `audio`,
// lets you place / drag / delete cut lines and writes them into the `cuts` text widget
// (which is what the backend node outputs as `scene_cuts`).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FPS = 24, OVERLAP = 8, RULER = 20;
const C = {
  bg: "#101215", ruler: "#181b20", grid: "#2a2e36", wave: "#5f9cae", cut: "#f0a63a",
  cutText: "#1b1408", play: "#ef6b5a", muted: "#8f949e", text: "#e8e5de",
};

function fmt(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return m + ":" + (s < 10 ? "0" : "") + s.toFixed(2);
}

function findLoadAudioFile(node) {
  const inp = node.inputs?.find((i) => i.name === "audio");
  if (!inp || inp.link == null) return null;
  const seen = new Set();
  let link = app.graph.links[inp.link];
  for (let hops = 0; link && hops < 8; hops++) {
    const origin = app.graph.getNodeById(link.origin_id);
    if (!origin || seen.has(origin.id)) break;
    seen.add(origin.id);
    const w = origin.widgets?.find((w) => w.name === "audio" && typeof w.value === "string");
    if (w) return w.value;
    // pass-through nodes (GetNode / Reroute / bypass): follow their first AUDIO input
    const up = origin.inputs?.find((i) => i.link != null && (i.type === "AUDIO" || i.type === "*"));
    if (!up) break;
    link = app.graph.links[up.link];
  }
  return null;
}

function stateSettings(node) {
  // read audio_start_sec / chunk_seconds from the State node this node's output feeds
  const out = node.outputs?.[0];
  for (const l of out?.links || []) {
    const link = app.graph.links[l];
    const target = link && app.graph.getNodeById(link.target_id);
    if (target?.type === "LTXChainState") {
      const get = (n, d) => { const w = target.widgets?.find((w) => w.name === n); return w ? +w.value : d; };
      return { astart: get("audio_start_sec", 0), chunk: get("chunk_seconds", 10), linked: true };
    }
  }
  return { astart: 0, chunk: 10, linked: false };
}

// ---------- automatic cut placement (beats / bars / section changes) ----------
// Everything runs on a mono 11025 Hz copy of the buffer: spectral-flux onset envelope ->
// tempo by autocorrelation -> beat grid -> downbeats (bar starts) -> section novelty
// (checkerboard kernel on smoothed band energies) -> cuts on downbeats near novelty peaks,
// spaced around the requested scene length.
function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

function analyzeAudio(buffer) {
  const SR = 11025, N = 1024, HOP = 256;
  // mono + resample (linear)
  const chs = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) chs.push(buffer.getChannelData(c));
  const ratio = buffer.sampleRate / SR, len = Math.floor(chs[0].length / ratio);
  const x = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const p = i * ratio, k = Math.floor(p), f = p - k;
    let v = 0;
    for (const ch of chs) v += ch[k] * (1 - f) + (ch[Math.min(k + 1, ch.length - 1)] || 0) * f;
    x[i] = v / chs.length;
  }
  const frames = Math.max(1, Math.floor((len - N) / HOP));
  const fps = SR / HOP;
  const win = new Float32Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  const NB = 12, edges = [];
  for (let b = 0; b <= NB; b++) edges.push(Math.round(2 * Math.pow(N / 4, b / NB)));  // log bands 2..256 bins
  const bands = new Float32Array(frames * NB), onset = new Float32Array(frames), low = new Float32Array(frames);
  let prev = new Float32Array(N / 2);
  const re = new Float32Array(N), im = new Float32Array(N);
  for (let t = 0; t < frames; t++) {
    const off = t * HOP;
    for (let i = 0; i < N; i++) { re[i] = x[off + i] * win[i]; im[i] = 0; }
    fftRadix2(re, im);
    let flux = 0;
    const mag = new Float32Array(N / 2);
    for (let k = 0; k < N / 2; k++) {
      mag[k] = Math.log1p(Math.sqrt(re[k] * re[k] + im[k] * im[k]) * 10);
      const d = mag[k] - prev[k];
      if (d > 0) flux += d;
    }
    prev = mag;
    onset[t] = flux;
    for (let b = 0; b < NB; b++) {
      let s = 0;
      for (let k = edges[b]; k < edges[b + 1]; k++) s += mag[k];
      bands[t * NB + b] = s / Math.max(1, edges[b + 1] - edges[b]);
    }
    let lo = 0;
    for (let k = 2; k < 12; k++) lo += mag[k];
    low[t] = lo;
  }
  // onset envelope: remove local mean
  const env = new Float32Array(frames), W = Math.round(fps * 0.5);
  for (let t = 0; t < frames; t++) {
    let s = 0, n = 0;
    for (let u = Math.max(0, t - W); u < Math.min(frames, t + W); u++) { s += onset[u]; n++; }
    env[t] = Math.max(0, onset[t] - s / n);
  }
  // tempo: autocorrelation of env for 60..180 BPM, prefer ~120
  let bestLag = Math.round(fps * 0.5), bestScore = -1;
  const minLag = Math.round((60 / 180) * fps), maxLag = Math.round((60 / 60) * fps);
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0;
    for (let t = lag; t < frames; t++) s += env[t] * env[t - lag];
    const bpm = (60 * fps) / lag;
    const w = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120) / 0.9, 2));
    if (s * w > bestScore) { bestScore = s * w; bestLag = lag; }
  }
  // refine the lag with sub-frame accuracy by parabolic interpolation on the autocorrelation
  const ac = (lag) => { let s = 0; for (let t = lag; t < frames; t++) s += env[t] * env[t - lag]; return s; };
  const a0 = ac(bestLag - 1), a1 = ac(bestLag), a2 = ac(bestLag + 1);
  const denom = a0 - 2 * a1 + a2;
  const period = bestLag + (denom !== 0 ? (0.5 * (a0 - a2)) / denom : 0);
  const bpm = (60 * fps) / period;
  // beat phase: the offset that best lines up with the envelope
  let bestPhase = 0, bestP = -1;
  for (let ph = 0; ph < period; ph += 0.5) {
    let s = 0;
    for (let t = ph; t < frames; t += period) s += env[Math.round(t)] || 0;
    if (s > bestP) { bestP = s; bestPhase = ph; }
  }
  const beats = [];
  for (let t = bestPhase; t < frames; t += period) beats.push(t);
  // downbeats: which of the 4 beat offsets carries the most low-frequency energy + onset
  let bestOff = 0, bestD = -1;
  for (let o = 0; o < 4; o++) {
    let s = 0;
    for (let i = o; i < beats.length; i += 4) { const t = Math.round(beats[i]); s += (env[t] || 0) + 0.5 * (low[t] || 0); }
    if (s > bestD) { bestD = s; bestOff = o; }
  }
  const downbeats = [];
  for (let i = bestOff; i < beats.length; i += 4) downbeats.push(beats[i] / fps);
  // section novelty: checkerboard kernel over smoothed, normalised band energies
  const K = Math.round(fps * 3), feat = bands;
  const novelty = new Float32Array(frames);
  const mean = new Float32Array(NB);
  for (let t = 0; t < frames; t++) for (let b = 0; b < NB; b++) mean[b] += feat[t * NB + b] / frames;
  for (let t = K; t < frames - K; t += 4) {
    let d = 0;
    for (let b = 0; b < NB; b++) {
      let s1 = 0, s2 = 0;
      for (let u = t - K; u < t; u++) s1 += feat[u * NB + b];
      for (let u = t; u < t + K; u++) s2 += feat[u * NB + b];
      const v = (s2 - s1) / K / (mean[b] + 1e-3);
      d += v * v;
    }
    novelty[t] = novelty[t + 1] = novelty[t + 2] = novelty[t + 3] = Math.sqrt(d);
  }
  let nmax = 0;
  for (let t = 0; t < frames; t++) if (novelty[t] > nmax) nmax = novelty[t];
  for (let t = 0; t < frames; t++) novelty[t] /= nmax || 1;
  const noveltyAt = (sec) => { const t = Math.round(sec * fps); let m = 0; for (let u = Math.max(0, t - (K >> 1)); u < Math.min(frames, t + (K >> 1)); u++) if (novelty[u] > m) m = novelty[u]; return m; };
  return { bpm, downbeats, noveltyAt, duration: len / SR };
}

function autoCuts(analysis, { start, end, target, min, mode }) {
  const { downbeats, noveltyAt } = analysis;
  const cuts = [];
  let cur = start;
  const max = target * 1.6;
  while (end - cur > target + min) {
    const lo = cur + Math.max(min, target * 0.6), hi = Math.min(cur + max, end - min);
    let best = null, bestScore = -Infinity;
    for (const d of downbeats) {
      if (d < lo || d > hi) continue;
      const closeness = 1 - Math.abs(d - (cur + target)) / target;          // prefer the requested length
      const score = mode === "sections" ? 2.0 * noveltyAt(d) + closeness : closeness;
      if (score > bestScore) { bestScore = score; best = d; }
    }
    if (best == null) {                                                    // no downbeat in range: fixed step
      best = cur + target;
      if (best > end - min) break;
    }
    cuts.push(+best.toFixed(2));
    cur = best;
  }
  return cuts;
}

function setupWaveform(node) {
  const cutsWidget = node.widgets.find((w) => w.name === "cuts");

  const el = document.createElement("div");
  el.style.cssText = "display:flex;flex-direction:column;gap:4px;height:100%;font:12px 'Segoe UI',sans-serif;color:" + C.text + ";";
  el.innerHTML = `
    <div class="ltxc-bar" style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
      <button data-a="play">再生</button>
      <button data-a="reload">再読込</button>
      <span data-r="clock" style="font-family:Consolas,monospace;min-width:12ch">0:00.00 / 0:00.00</span>
      <label>スナップ <select data-r="snap"><option value="0.1">0.1s</option><option value="0.5">0.5s</option><option value="1">1s</option><option value="0">なし</option></select></label>
      <label>ズーム <input data-r="zoom" type="range" min="1" max="16" step="1" value="1" style="width:90px;vertical-align:middle"></label>
      <button data-a="clear">全部消す</button>
      <span style="width:1px;height:18px;background:#2f343d"></span>
      <button data-a="auto">自動カット</button>
      <label>目標 <input data-r="target" type="number" value="20" min="4" step="1" style="width:56px">s</label>
      <select data-r="mode"><option value="sections">構成の変わり目</option><option value="beats">小節の頭で等間隔</option></select>
      <span data-r="file" style="color:${C.muted};margin-left:auto"></span>
    </div>
    <div data-r="scroller" style="flex:1;min-height:120px;overflow-x:auto;overflow-y:hidden;background:${C.bg};border-radius:4px"><canvas></canvas></div>
    <div data-r="info" style="color:${C.muted};font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"></div>`;
  for (const b of el.querySelectorAll("button, select, input"))
    b.style.cssText += "background:#23272e;color:" + C.text + ";border:1px solid #2f343d;border-radius:4px;padding:2px 8px;font-size:12px;";
  const R = (n) => el.querySelector(`[data-r="${n}"]`);
  const scroller = R("scroller"), cv = scroller.querySelector("canvas"), ctx = cv.getContext("2d");
  cv.style.cursor = "crosshair";

  const st = {
    ac: null, buffer: null, duration: 60, peaks: null, file: null,
    cuts: parseCuts(cutsWidget.value), selected: -1, dragging: -1,
    playing: false, src: null, playStart: 0, playOffset: 0, zoom: 1,
  };

  function parseCuts(s) {
    return [...new Set(String(s || "").split(/[\s,;]+/).filter(Boolean).map(Number).filter((v) => !isNaN(v)))].sort((a, b) => a - b);
  }
  function writeCuts() {
    const v = st.cuts.map((t) => +t.toFixed(2)).join(", ");
    if (cutsWidget.value !== v) { cutsWidget.value = v; }
    node.setDirtyCanvas(true, true);
  }
  cutsWidget.callback = (v) => { st.cuts = parseCuts(v); st.selected = -1; refresh(); };

  // ---------- geometry / drawing ----------
  const cssW = () => Math.max(200, Math.floor((scroller.clientWidth - 2) * st.zoom));
  const cssH = () => Math.max(100, scroller.clientHeight - 12);
  const xOf = (t) => (t / st.duration) * cssW();
  const tOf = (x) => (x / cssW()) * st.duration;

  function computePeaks(columns) {
    const buf = st.buffer;
    const out = new Float32Array(columns * 2);
    if (!buf) return out;
    const chs = [];
    for (let c = 0; c < buf.numberOfChannels; c++) chs.push(buf.getChannelData(c));
    const len = chs[0].length, per = len / columns;
    for (let x = 0; x < columns; x++) {
      let mn = 1, mx = -1;
      const a = Math.floor(x * per), b = Math.min(len, Math.floor((x + 1) * per));
      const stride = Math.max(1, Math.floor((b - a) / 300));
      for (let i = a; i < b; i += stride) {
        let v = 0;
        for (const ch of chs) v += ch[i];
        v /= chs.length;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      out[x * 2] = mn; out[x * 2 + 1] = mx;
    }
    return out;
  }
  function layout() {
    const w = cssW(), h = cssH(), dpr = window.devicePixelRatio || 1;
    cv.style.width = w + "px"; cv.style.height = h + "px";
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    st.peaks = computePeaks(w);
    draw();
  }
  function tickStep() {
    const pps = cssW() / st.duration;
    for (const s of [0.5, 1, 2, 5, 10, 15, 30, 60, 120]) if (s * pps >= 60) return s;
    return 300;
  }
  function bounds() {
    const { astart } = stateSettings(node);
    const a0 = Math.min(astart, st.duration), end = st.duration;
    const inner = st.cuts.filter((c) => c > a0 + 0.5 && c < end - 0.5);
    const clean = [a0];
    for (const b of inner) if (b - clean[clean.length - 1] >= 0.5) clean.push(b);
    clean.push(end);
    return clean;
  }
  function draw() {
    const w = cssW(), h = cssH(), mid = RULER + (h - RULER) / 2, amp = (h - RULER) / 2 - 4;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = C.ruler; ctx.fillRect(0, 0, w, RULER);
    const { astart } = stateSettings(node);
    if (astart > 0) { ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.fillRect(0, RULER, xOf(Math.min(astart, st.duration)), h - RULER); }
    const step = tickStep();
    ctx.font = "10px Consolas,monospace"; ctx.fillStyle = C.muted; ctx.textBaseline = "top";
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    for (let t = 0; t <= st.duration + 1e-6; t += step) {
      const x = Math.round(xOf(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, RULER - 5); ctx.lineTo(x, h); ctx.stroke();
      ctx.fillText(fmt(t).replace(/\.\d+$/, ""), x + 3, 4);
    }
    if (!st.buffer) {
      ctx.fillStyle = C.muted; ctx.font = "13px 'Segoe UI',sans-serif"; ctx.textBaseline = "middle";
      ctx.fillText(st.file ? "読み込み中… " + st.file : "audio 入力に LoadAudio をつなぐと波形が出ます", 12, mid);
    } else {
      ctx.fillStyle = C.wave;
      for (let x = 0; x < w; x++) {
        const mn = st.peaks[x * 2], mx = st.peaks[x * 2 + 1];
        const y1 = mid - mx * amp, y2 = mid - mn * amp;
        ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
      }
    }
    const b = bounds();
    ctx.font = "600 11px 'Segoe UI',sans-serif"; ctx.textBaseline = "top";
    for (let i = 0; i + 1 < b.length; i++) {
      if (i % 2 === 1) { ctx.fillStyle = "rgba(255,255,255,.035)"; ctx.fillRect(xOf(b[i]), RULER, xOf(b[i + 1]) - xOf(b[i]), h - RULER); }
      ctx.fillStyle = C.muted; ctx.fillText("シーン " + (i + 1), xOf(b[i]) + 5, RULER + 4);
    }
    st.cuts.forEach((t, i) => {
      const x = Math.round(xOf(t)) + 0.5;
      ctx.strokeStyle = C.cut; ctx.lineWidth = i === st.selected ? 3 : 1.5;
      ctx.beginPath(); ctx.moveTo(x, RULER); ctx.lineTo(x, h); ctx.stroke();
      const label = fmt(t); ctx.font = "500 10px Consolas,monospace";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = C.cut; ctx.fillRect(x - tw / 2, h - 16, tw, 14);
      ctx.fillStyle = C.cutText; ctx.textBaseline = "middle"; ctx.fillText(label, x - tw / 2 + 4, h - 9);
    });
    const pt = currentTime(), px = Math.round(xOf(pt)) + 0.5;
    ctx.strokeStyle = C.play; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    R("clock").textContent = fmt(pt) + " / " + fmt(st.duration);
  }
  function updateInfo() {
    const { chunk, linked } = stateSettings(node), b = bounds(), stepS = chunk - OVERLAP / FPS;
    let total = 0; const parts = [];
    for (let i = 0; i + 1 < b.length; i++) {
      const L = b[i + 1] - b[i];
      const m = L <= chunk ? 1 : Math.ceil((L - chunk) / stepS - 1e-9) + 1;
      const last = L <= chunk ? L : L - (m - 1) * stepS;
      total += m;
      parts.push(`S${i + 1} ${L.toFixed(1)}s/${m}clip${m > 1 && last < 1.5 ? "(末尾" + last.toFixed(1) + "s)" : ""}`);
    }
    R("info").textContent = (st.bpm ? `≈${st.bpm.toFixed(1)} BPM  ` : "") + `合計 ${total} クリップ（chunk ${chunk}s${linked ? "" : " ※State 未接続: 既定値"}）  ` + parts.join("  ");
    R("info").title = R("info").textContent;
  }
  function refresh() { updateInfo(); draw(); writeCuts(); }

  // ---------- mouse ----------
  const snapTo = (t) => { const s = +R("snap").value; t = Math.max(0, Math.min(st.duration, t)); return s > 0 ? Math.round(t / s) * s : t; };
  const localXY = (e) => { const r = cv.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * cssW(), ((e.clientY - r.top) / r.height) * cssH()]; };
  const hit = (x) => { let best = -1, bd = 7; st.cuts.forEach((t, i) => { const d = Math.abs(xOf(t) - x); if (d < bd) { bd = d; best = i; } }); return best; };
  cv.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    const [x, y] = localXY(e);
    if (y < RULER) { seek(tOf(x)); return; }
    let h = hit(x);
    if (h < 0) { const t = snapTo(tOf(x)); st.cuts.push(t); st.cuts.sort((a, b) => a - b); h = st.cuts.indexOf(t); }
    st.selected = h; st.dragging = h;
    cv.setPointerCapture(e.pointerId);
    refresh();
  });
  cv.addEventListener("pointermove", (e) => {
    const [x] = localXY(e);
    if (st.dragging >= 0) { st.cuts[st.dragging] = snapTo(tOf(x)); draw(); }
    else cv.style.cursor = hit(x) >= 0 ? "ew-resize" : "crosshair";
  });
  cv.addEventListener("pointerup", (e) => {
    if (st.dragging < 0) return;
    const t = st.cuts[st.dragging]; st.cuts.sort((a, b) => a - b); st.selected = st.cuts.indexOf(t); st.dragging = -1;
    refresh();
  });
  cv.addEventListener("dblclick", (e) => {
    e.stopPropagation();
    const [x] = localXY(e), h = hit(x);
    if (h >= 0) { st.cuts.splice(h, 1); st.selected = -1; refresh(); }
  });
  cv.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); const h = hit(localXY(e)[0]); if (h >= 0) { st.cuts.splice(h, 1); st.selected = -1; refresh(); } });
  cv.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
  el.addEventListener("keydown", (e) => {
    if ((e.key === "Delete" || e.key === "Backspace") && st.selected >= 0) { st.cuts.splice(st.selected, 1); st.selected = -1; refresh(); e.preventDefault(); }
    if (e.key === " ") { togglePlay(); e.preventDefault(); }
  });
  el.tabIndex = 0;
  el.addEventListener("pointerdown", (e) => e.stopPropagation());
  el.addEventListener("dblclick", (e) => e.stopPropagation());

  el.querySelector('[data-a="clear"]').onclick = () => { st.cuts = []; st.selected = -1; refresh(); };
  el.querySelector('[data-a="auto"]').onclick = () => {
    if (!st.buffer) return;
    const btn = el.querySelector('[data-a="auto"]');
    btn.textContent = "解析中…"; btn.disabled = true;
    setTimeout(() => {
      try {
        if (!st.analysis || st.analysis.file !== st.file) { st.analysis = analyzeAudio(st.buffer); st.analysis.file = st.file; }
        const { astart } = stateSettings(node), target = Math.max(4, +R("target").value || 20);
        st.cuts = autoCuts(st.analysis, { start: astart, end: st.duration, target, min: Math.max(4, target * 0.4), mode: R("mode").value });
        st.selected = -1; st.bpm = st.analysis.bpm; refresh();
      } finally { btn.textContent = "自動カット"; btn.disabled = false; }
    }, 20);
  };
  el.querySelector('[data-a="reload"]').onclick = () => load(true);
  el.querySelector('[data-a="play"]').onclick = () => togglePlay();
  R("snap").onchange = () => el.focus();
  R("zoom").oninput = () => {
    const centre = (scroller.scrollLeft + scroller.clientWidth / 2) / cssW();
    st.zoom = +R("zoom").value; layout();
    scroller.scrollLeft = centre * cssW() - scroller.clientWidth / 2;
  };

  // ---------- playback ----------
  function currentTime() { return st.playing && st.ac ? Math.min(st.duration, st.playOffset + (st.ac.currentTime - st.playStart)) : st.playOffset; }
  function seek(t) { const was = st.playing; if (was) stop(); st.playOffset = Math.max(0, Math.min(st.duration, t)); if (was) start(); else draw(); }
  function start() {
    if (!st.buffer) return;
    if (!st.ac) st.ac = new (window.AudioContext || window.webkitAudioContext)();
    if (st.ac.state === "suspended") st.ac.resume();
    st.src = st.ac.createBufferSource(); st.src.buffer = st.buffer; st.src.connect(st.ac.destination);
    st.src.onended = () => { if (st.playing && currentTime() >= st.duration - 0.05) { st.playing = false; st.playOffset = 0; el.querySelector('[data-a="play"]').textContent = "再生"; draw(); } };
    st.playStart = st.ac.currentTime; st.src.start(0, st.playOffset); st.playing = true;
    el.querySelector('[data-a="play"]').textContent = "停止";
    tick();
  }
  function stop() { st.playOffset = currentTime(); st.playing = false; try { st.src && st.src.stop(); } catch (e) {} el.querySelector('[data-a="play"]').textContent = "再生"; draw(); }
  function togglePlay() { st.playing ? stop() : start(); }
  function tick() {
    if (!st.playing) return;
    draw();
    const px = xOf(currentTime());
    if (px < scroller.scrollLeft || px > scroller.scrollLeft + scroller.clientWidth) scroller.scrollLeft = px - 40;
    requestAnimationFrame(tick);
  }

  // ---------- audio file ----------
  async function load(force) {
    const file = findLoadAudioFile(node);
    if (!file) { st.file = null; st.buffer = null; R("file").textContent = ""; layout(); updateInfo(); return; }
    if (file === st.file && st.buffer && !force) return;
    if (st.playing) stop();
    st.file = file; st.buffer = null; R("file").textContent = file; layout();
    const i = file.lastIndexOf("/");
    const params = new URLSearchParams({ filename: i >= 0 ? file.slice(i + 1) : file, subfolder: i >= 0 ? file.slice(0, i) : "", type: "input" });
    try {
      const res = await fetch(api.apiURL("/view?" + params.toString()));
      if (!res.ok) throw new Error(res.status);
      if (!st.ac) st.ac = new (window.AudioContext || window.webkitAudioContext)();
      const buf = await st.ac.decodeAudioData(await res.arrayBuffer());
      if (st.file !== file) return;
      st.buffer = buf; st.duration = buf.duration; st.playOffset = 0; st.analysis = null; st.bpm = 0;
      R("file").textContent = file + "（" + fmt(buf.duration) + "）";
    } catch (err) {
      R("file").textContent = "読み込めません: " + file;
    }
    layout(); updateInfo();
  }

  const widget = node.addDOMWidget("waveform", "LTXC_WAVE", el, {
    getValue() { return undefined; },
    setValue() {},
    getMinHeight() { return 230; },
    hideOnZoom: false,
  });
  widget.serialize = false;
  widget.serializeValue = () => undefined;
  // The DOM overlay is sized from `widget.width ?? node.width`; some frontend panels stamp a
  // narrow preview width onto widget.width, which would pin the overlay at ~230 px. Always
  // follow the node width.
  Object.defineProperty(widget, "width", { get: () => node.size[0], set() {}, configurable: true });

  new ResizeObserver(() => { layout(); }).observe(scroller);
  const onConn = node.onConnectionsChange;
  node.onConnectionsChange = function () { onConn?.apply(this, arguments); setTimeout(() => { load(false); updateInfo(); }, 50); };
  // the State node's chunk/start may change any time: refresh the summary when the node is redrawn
  const onDraw = node.onDrawForeground;
  let lastKey = "";
  node.onDrawForeground = function () {
    onDraw?.apply(this, arguments);
    const { astart, chunk } = stateSettings(node), key = astart + "|" + chunk;
    if (key !== lastKey) { lastKey = key; updateInfo(); draw(); }
  };
  const onConfigure = node.onConfigure;
  node.onConfigure = function () {
    onConfigure?.apply(this, arguments);
    st.cuts = parseCuts(cutsWidget.value); st.selected = -1;
    setTimeout(() => { load(false); refresh(); }, 100);
  };
  setTimeout(() => { load(false); updateInfo(); }, 100);
  if (node.size[0] < 620) node.setSize([720, Math.max(node.size[1], 330)]);
}

app.registerExtension({
  name: "ltxchain.scenecuts",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LTXChainSceneCuts") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setupWaveform(this);
      return r;
    };
  },
});
