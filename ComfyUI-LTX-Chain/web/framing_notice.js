// LTX Chain: Auto Scenes - live "face size" estimate. The face is what breaks first when a shot gets
// wide: stage 1 runs at half resolution and the VAE packs 32 px into one latent cell, so below
// ~120 px of face height in the OUTPUT frame the model can no longer hold the face. This widget
// reads the State node's resolution + Auto Scenes' framing_limit and shows the estimate and a
// warning; a dialog pops up once when the limit is moved into the risky range.
import { app } from "../../scripts/app.js";

// fraction of the output height a face takes in each framing (measured on this project's renders)
const FACE_FRACTION = { "close-up": 0.45, "chest-up": 0.30, "waist-up": 0.22, "mid-thigh": 0.15, "full body": 0.10, "wide": 0.06 };
const SAFE_PX = 120, BORDER_PX = 90;

function outputHeight() {
  // State node: resolution preset "WxH — ..." or auto -> generation_width/height + image aspect unknown
  const state = app.graph?._nodes?.find((n) => n.type === "LTXChainState");
  if (!state) return null;
  const res = state.widgets?.find((w) => w.name === "resolution")?.value;
  if (typeof res === "string" && !res.startsWith("auto")) {
    const m = res.match(/^(\d+)x(\d+)/);
    if (m) return { h: parseInt(m[2], 10), w: parseInt(m[1], 10), label: m[0] };
  }
  const gh = state.widgets?.find((w) => w.name === "generation_height")?.value;
  const gw = state.widgets?.find((w) => w.name === "generation_width")?.value;
  if (gh) return { h: gh, w: gw, label: `auto ≈ ${gw}x${gh}` };
  return null;
}

function estimate(limit) {
  const out = outputHeight();
  if (!out) return null;
  const frac = FACE_FRACTION[limit] ?? 0.06;
  const px = Math.round(out.h * frac);
  const level = px >= SAFE_PX ? "ok" : px >= BORDER_PX ? "border" : "risk";
  return { px, level, out, limit };
}

function label(e) {
  if (!e) return "顔サイズ推定: State ノードが見つかりません";
  const mark = e.level === "ok" ? "✓ 安定" : e.level === "border" ? "△ 境界" : "⚠ 崩れやすい";
  return `顔サイズ推定 (${e.out.label}, ${e.limit}): 約 ${e.px}px  ${mark}`;
}

function showDialog(e) {
  const body = `<div style="max-width:560px;text-align:left;line-height:1.55;font-size:13px">
    <div style="font-size:14px;margin-bottom:6px;color:#f0a63a">⚠ この構図では顔が崩れやすい範囲です</div>
    出力 <b>${e.out.label}</b> で一番引いた構図 <b>${e.limit}</b> のとき、顔の高さは <b>約 ${e.px}px</b> と推定されます。<br><br>
    LTX-2.5 は Stage 1 を出力の半分の解像度で動かし、VAE は 32px を 1 潜在セルにまとめるため、
    出力で顔が <b>${SAFE_PX}px 未満</b>（Stage 1 で潜在 2 セル未満）になると MSR 参照があっても顔の骨格を保てず、
    ぼけ・頭身崩れが出ます（目安: ≥${SAFE_PX}px 安定 / ${BORDER_PX}〜${SAFE_PX}px 境界 / &lt;${BORDER_PX}px 崩れやすい）。<br><br>
    <b>対策</b><br>
    ・<code>framing_limit</code> を 1 段寄せる（waist-up / chest-up）<br>
    ・State の <code>resolution</code> を上げる（576x1024 / 640x1152）<br>
    ・全身を出したいシーンは短くし、<code>msr_clips = all</code> ＋ Stage 1 に全身参照<br>
    <small>※ 公式仕様ではなく、このプロジェクトの実測に基づく目安です。extra_cameras の自作アングルは推定に含まれません。</small></div>`;
  const dlg = app.ui?.dialog;
  if (dlg && typeof dlg.show === "function") dlg.show(body); else alert(body.replace(/<[^>]+>/g, ""));
}

app.registerExtension({
  name: "LTXChain.FramingNotice",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "LTXChainAutoScenes") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      const limitW = node.widgets?.find((w) => w.name === "framing_limit");
      if (!limitW || node.__ltxFace) return r;
      node.__ltxFace = true;
      // read-only info line drawn under the widgets (NOT a widget, so it is never serialized into
      // widgets_values and can never shift the saved values)
      node.__ltxFaceText = "";
      const onDraw = node.onDrawForeground;
      node.onDrawForeground = function (ctx) {
        const r = onDraw?.apply(this, arguments);
        if (this.flags?.collapsed || !this.__ltxFaceText) return r;
        ctx.save();
        ctx.font = "12px sans-serif";
        ctx.fillStyle = this.__ltxFaceLevel === "risk" ? "#f0a63a" : this.__ltxFaceLevel === "border" ? "#e6d27a" : "#9ad0a0";
        ctx.textAlign = "left";
        ctx.fillText(this.__ltxFaceText, 12, this.size[1] - 8);
        ctx.restore();
        return r;
      };
      const baseSize = node.computeSize;
      node.computeSize = function () { const sz = baseSize.apply(this, arguments); sz[1] += 22; return sz; };
      let lastLevel = null;
      const refresh = (fromUser) => {
        const e = estimate(limitW.value);
        node.__ltxFaceText = label(e);
        node.__ltxFaceLevel = e?.level;
        if (fromUser && e && e.level === "risk" && lastLevel !== "risk") showDialog(e);
        if (e) lastLevel = e.level;
        node.setDirtyCanvas(true, true);
      };
      const orig = limitW.callback;
      limitW.callback = function (v, ...rest) { const o = orig?.call(this, v, ...rest); refresh(true); return o; };
      // State resolution changes are elsewhere: poll cheaply while the node is alive
      setTimeout(() => refresh(false), 500);
      node.__ltxTimer = setInterval(() => refresh(false), 2000);
      const onRemoved = node.onRemoved;
      node.onRemoved = function () { clearInterval(node.__ltxTimer); return onRemoved?.apply(this, arguments); };
      return r;
    };
  },
});
