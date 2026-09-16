// LTX Chain: State - when a "(test ...)" resolution preset is picked, show a one-time notice
// explaining what those small sizes are (and are not) good for.
import { app } from "../../scripts/app.js";

const TEXT_JA = `
<b>テスト用の小さいサイズを選びました。</b><br><br>
<b>向いている用途</b><br>
・動き（motion）・カメラワーク・シーン切替のタイミング・プロンプトの効き方を速く確認する<br>
・<code>length_mode = seconds</code> で曲の一部だけ回す<br><br>
<b>判断に使えないもの</b><br>
・<b>顔の一貫性・リップシンク・体型（頭身）</b> — 顔の画素が 40〜60px しかなく本番と別物になります<br>
・512 未満は手や背景の崩れも増えます<br><br>
<b>注意</b><br>
・本番セッションへの <code>redo_clips</code> には使えません（フレームサイズが混ざる）。テストは新しいセッションで<br>
・速度は画素数ほどは縮みません（480p 相当で約 0.8 倍、tiny で約 0.6 倍）<br><br>
<small>本番は 512x896 以上を選んでください。</small>`;

function showNotice(node, value) {
  const dlg = app.ui?.dialog;
  const body = `<div style="max-width:560px;text-align:left;line-height:1.55;font-size:13px">
    <div style="font-size:14px;margin-bottom:6px;color:#f0a63a">⚠ ${value}</div>${TEXT_JA}</div>`;
  if (dlg && typeof dlg.show === "function") {
    dlg.show(body);
  } else {
    alert(body.replace(/<[^>]+>/g, ""));
  }
}

app.registerExtension({
  name: "LTXChain.ResolutionNotice",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "LTXChainState") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const w = this.widgets?.find((x) => x.name === "resolution");
      if (!w || w.__ltxNotice) return r;
      w.__ltxNotice = true;
      const orig = w.callback;
      let last = w.value;
      w.callback = function (value, ...rest) {
        const out = orig?.call(this, value, ...rest);
        const isTest = typeof value === "string" && /\(test/i.test(value);
        const wasTest = typeof last === "string" && /\(test/i.test(last);
        if (isTest && !wasTest) showNotice(w, value); // once per switch into a test size, not on every re-pick
        last = value;
        return out;
      };
      return r;
    };
  },
});
