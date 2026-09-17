# LTX2.5 LIP SYNC with MSR — 作業メモ

作成日: 2026-09-13 〜 2026-09-14
環境: ComfyUI v0.34.3（Easy-Install, `E:\download\ComfyUI-Easy-Install-Windows\ComfyUI-Easy-Install\ComfyUI`）/ RTX 4070 Ti 12GB / RAM 32GB
モデル: LTX-2.5 22B distilled（int8 convrot）+ Gemma4 12B（int8 convrot）+ LTX-2.5 MSR IC-LoRA

---

## 1. 目的

元のワークフロー `LTX-2.5-lip sync.json`（参照画像 1 枚 + mp3 → 5 秒のリップシンク動画）を、
**Run を 1 回押すだけで曲の最後まで自動で生成し続け、1 本の動画にまとめる**チェーン生成に拡張する。

- 5 秒（任意秒数）ずつ生成 → 終わったフレームを次の参照にして自動で次を生成
- mp3 全部 / 指定秒数だけ を選べる
- 各クリップと連結済み動画を `output/LTX2.5Chains/<日付>_v<番号>/` に保存
- 顔ができるだけ変わらないように MSR（Multi-Subject Reference）で顔参照を固定

---

## 2. 作ったもの

### カスタムノード `custom_nodes/ComfyUI-LTX-Chain/nodes.py`

| ノード | 役割 |
|---|---|
| **LTX Chain: State** | 今回のクリップ番号を決め、参照フレーム・mp3 開始秒・生成秒数・出力サイズを出す。2 本目以降は前クリップの受け渡しフレームを読む |
| **LTX Chain: Step** | 生成結果を色正規化 → 受け渡しフレーム保存 → クリップ mp4 保存 → 同じワークフローを次のクリップ番号で `/prompt` に再投入。最後のクリップで全部を連結して `final.mp4` を作る |
| **LTX Chain: Prompt Switch** | 歌が始まる秒（`vocal_start_sec`）より前のクリップは `prompt_intro`、以降は `prompt_singing` を使う |
| **LTX Chain: Prompt Cache** | Enhancer を 1 セッションで 1 回だけ実行して使い回す（現在は未使用・バイパス） |

### ワークフロー（`user/default/workflows/`）

- `LTX-2.5-lip sync-CHAIN.json` … チェーン生成のみ
- `LTX-2.5-lip sync-CHAIN+MSR.json` … チェーン生成 + MSR 顔参照（REF 1 / REF 2）
- `LTX-2.5-lip sync-CHAIN+MSR-SCENES.json` … 上に加えて N クリップごとにシーン（アングル）を切り替える版（§10）
- 元の `LTX-2.5-lip sync.json` は無変更

### 出力フォルダ構成

```
output/LTX2.5Chains/20260914_v03/
  clip_001.mp4, clip_002.mp4, ...   各クリップ（そのクリップ分の音声付き）
  final.mp4                          全クリップ連結 + 元 mp3 を通しで付けた完成動画
  handoff_000.npy / last_frame_000.png   受け渡しフレーム
```
同じ日に何度か回すと v01, v02 … と増える。

---

## 3. 使い方

1. LoadImage に開始画像、LoadAudio に mp3。MSR 版は REF 1 に顔がはっきり写った画像
2. **LTX Chain: State** を設定
   - `audio_start_sec` … mp3 のどこから使うか
   - `chunk_seconds` … 1 回で作る秒数（5 または 10）
   - `length_mode` … `all`（mp3 の最後まで）/ `seconds`（`length_seconds` 秒だけ）
   - `generation_width / height` … 生成サイズの目安（宽/高 ノードから接続済み）
   - `msr_clips` … `stage2_all`（既定）/ `first_only` / `all`
   - `overlap_frames` … 8（既定）
   - `handoff_color_match` … 1.0（既定）
   - `chain_iter` / `chain_state` … 内部用。0 / 空のまま
3. **Positive Prompt（Prompt Switch）** に `vocal_start_sec`、`prompt_intro`、`prompt_singing`
   - `vocals`（MelBandRoFormer の歌声）が接続済み。歌声区間が `min_vocal_ratio` 未満のクリップは自動で `prompt_intro`
   - `prompt_intro` が空だと常に `prompt_singing`。判定は `<セッション>/prompts.txt` で確認できる
   - MSR 版は最初の画像が MSR ガイドの `background` にも接続済み（背景の崩れ防止。不要なら線を外す）
4. Run を 1 回。止めたいときは Cancel（そのクリップ終了後に次は投入されない）

ComfyUI 再起動後や、ノードの入力欄・配線を変えたあとは **タブを閉じる → F5 → ワークフローを開き直す**（ブラウザは古いノード定義をキャッシュし、しかも開いているワークフローをファイルに自動上書き保存する）。

**画面の見方**: 左端の緑のボックス「① 入力・設定（ここだけ操作すればOK）」に、人が触るノード（開始画像 / 音声 / REF 1〜4 / State / Scene Cuts / Scene Prompt）を集約してある。右側はステップ別のボックス（モデル読み込み → 音声 → Stage 1 → Stage 2 → 出力）で、通常は触らない。配線用の Set/Get・スイッチは折りたたみ済み。使わない Enhancer 系は「使わない（バイパス中）」ボックスに寄せてある。ノードは自由にドラッグして微調整できる。

---

## 4. 仕組み

- Step ノードが hidden 入力 `PROMPT` / `EXTRA_PNGINFO` で自分のワークフローを受け取り、State の `chain_iter` と `chain_state`（セッション名・受け渡しファイル）だけ書き換えて HTTP で `/prompt` に再投入する
- 受け渡しは前クリップの **最後の 9 フレーム**（1 + `overlap_frames`）。LTX の VAE は 8 フレーム = 1 latent フレームなので 9 フレーム = 2 latent フレームぶんを `LTXVImgToVideoInplace` で固定する
- 各クリップの新規部分は `chunk_seconds − 8/24 秒`（5 秒設定なら 4.67 秒）。mp3 もその分ずらして切り出す
- 連結時、各クリップ（最後以外）は末尾 9 フレームを持たず、次のクリップの先頭 9 フレームを前クリップの本物の末尾からディゾルブして接続する
- 最後のクリップで PyAV により全クリップを再エンコード連結し、元 mp3 を `audio_start_sec` から動画長ぶん付ける
- mp3 末尾の端数が 1 秒未満なら最後のクリップは作らない

---

## 5. 起きた問題と対処（時系列）

| # | 症状 | 原因 | 対処 |
|---|---|---|---|
| 1 | 「不足しているパック」ダイアログ | ブラウザがノード定義をキャッシュ | F5 → 開き直し |
| 2 | 5 秒ごとに少しズームイン | `ImageResizeKJv2` divisible_by=32 で latent 格子とずれ中央クロップ | 64 に変更（後に #10 で根本対処） |
| 3 | 色がだんだん濃くなる | 生成のたびに少し暗く・平坦になる癖の蓄積 | 保存フレームを毎フレーム参照画像の平均・分散に正規化（`handoff_color_match`、窓 3 フレーム） |
| 4 | Enhancer が失敗 | `ComfyUI-LTXVideo` の Enhancer が Gemma 3 用。12B int8 エンコーダは文章生成で文字化け | 本体の `TextGenerateLTX2Prompt` + `gemma4_e4b_it_fp8_scaled` に差し替え |
| 5 | Enhancer ON で ComfyUI ごと落ちる | RAM 32GB に LTX 22B + Gemma 12B + Gemma E4B が乗らない | Enhancer 系 3 ノードをバイパス（未使用） |
| 6 | 最後の 1 秒でメモリエラー | mp3 の端数 0.2 秒で 0 フレームのクリップをエンコード | 端数 1 秒未満は切り捨て、空クリップのガード |
| 7 | 継ぎ目で明るさが段差 | 受け渡しフレームだけ色補正していた | 保存フレーム側を正規化（#3 と統合） |
| 8 | 継ぎ目直後に暗くなって戻る脈動 | 正規化の平滑化窓 15 が固定フレームと混ざる | 窓 3、固定フレームを統計から除外 |
| 9 | 継ぎ目で一瞬二重＋ボケ | Stage 1 の Inplace 1.0 → 悪化 / Stage 2 バイパス・0.6 → ポーズが変わる。真の原因は #10 と #11 | — |
| 10 | 受け渡しフレームが毎回 2.7% ズーム | `WanVideoImageResizeToClosest` が面積合わせで拡大 → `ImageResizeKJv2` がクロップ | この 2 ノードを削除し、State ノードがサイズ決定（初回のみリサイズ、以降は無加工で受け渡し） |
| 11 | 継ぎ目で顔が正面へスナップ | MSR の参照が「直前フレーム」として入るため、2 本目以降で参照ポーズに引き戻される（strength 0.3 でも同じ） | `easy ifElse` ×6 で MSR ガイドをクリップごとに ON/OFF。動きを決める Stage 1 は 1 本目だけ、ディテールを描く Stage 2 は毎クリップ MSR（`msr_clips = stage2_all`） |
| 12 | 長尺（100 秒〜）で背景がサイケ模様に崩れる（v19） | ボケた背景には参照がなく、生成のたびに「ボケの中の模様」が濃くなる自己回帰ドリフト | MSR ガイドの `background` 入力に最初の画像（LoadImage）を接続（Stage 1 / 2 両方）。Stage 2 で毎クリップ元の背景を参照する |
| 13 | 間奏・無音でも口パクする | `vocal_start_sec` 以降は全クリップ singing プロンプトだった | Prompt Switch に MelBandRoFormer の `vocals` を接続。クリップ内の歌声区間（`vocal_threshold_db` -40 dB 超）が `min_vocal_ratio`（0.2）未満なら `prompt_intro` を使う。判定は `<セッション>/prompts.txt` に記録 |

最終確認: v16（first_only）= 継ぎ目の動きが連続、輝度 92.2〜92.5 で平ら、シャープネス最低 33。v18（stage2_all）= 同じく連続で、シャープネス最低 38 とさらに良く、毎クリップ REF を参照できる。v19 = 190 秒フル尺で継ぎ目・顔は良好、背景が終盤で崩れ（→ #12）。v20 = #12/#13 適用後の 2 クリップ動作確認（エラーなし、継ぎ目連続、vocal ratio 0.24 / 0.82 と妥当）。

---

## 5.5 書き出しサイズ（resolution）

State に `resolution` を追加。LTX-2.5 が安全に出せる 64 の倍数プリセット（`RES_PRESETS`）から選ぶ。
`auto` = 従来どおり画像の縦横比のまま `generation_width×height` の面積で `_target_size` により決定。
数値プリセット = `_parse_resolution` で WxH を取り、`_fit_image` でその比率に中央クロップして書き出し。
12GB は面積 ~600k px 目安（例 576×1024 / 768×768 / 1024×576）。v01（20260915）で 768×768 出力を確認。

## 6. 現在の設定値（両ワークフロー）

- Stage 1 `LTXVImgToVideoInplace` strength **1.0** / Stage 2 **1.0**（bypass off）
- State: `overlap_frames 8`, `handoff_color_match 1.0`, `msr_clips stage2_all`, `generation 609×1056`
- CreateVideo / SaveVideo（旧クリップ保存）はバイパス（Step が音声付きで保存する）
- Enhancer / Prompt Cache / Enhancer 用 CLIPLoader はバイパス

## 7. 既知の制約・今後の候補

- `stage2_all` では Stage 2 の精錬時に毎クリップ REF を参照する。それでも長尺で顔が変わる場合は `all`（継ぎ目でスナップが戻る）か REF 画像の見直し
- 継ぎ目直後の数フレームはわずかに柔らかい（VAE の時間方向補間）。`overlap_frames 16` で改善する可能性あり（新規部分は 1 本あたり 0.67 秒減る）
- Enhancer を使うには RAM 増設か、より小さい Gemma（E2B int8, 5.2GB）が必要
- ComfyUI-Manager の再起動後は `user/comfyui.log` が書かれなくなる。進捗は `http://127.0.0.1:8188/history` で確認できる
- 部分的に失敗したセッションは、`LTXChainStep._concat` を手動で呼べば残っているクリップだけで `final.mp4` を作れる（2026-09-14 v03 で実施）

## 8. 関連ファイル

- ノード: `ComfyUI/custom_nodes/ComfyUI-LTX-Chain/nodes.py`
- ワークフロー: `ComfyUI/user/default/workflows/LTX-2.5-lip sync-CHAIN.json`, `LTX-2.5-lip sync-CHAIN+MSR.json`
- MSR LoRA: `ComfyUI/models/loras/LTX-2.5-Licon-MSR-V1.safetensors`
- 出力: `ComfyUI/output/LTX2.5Chains/`

## 10. シーン切り替え版（CHAIN+MSR-SCENES）

`CharSheet_Krea_LTX25_MSR.json` の「LTX MSR dynamic N scenes」（開始画像なし・MSR 参照 + プロンプトだけでシーンを生成）を
チェーン生成に組み込んだもの。曲全体をリップシンクしたまま、N クリップごとにカメラアングル／場所を切り替える。

- State `clips_per_scene`（既定 2、0 = 従来どおり）: N クリップごとにシーン番号が進む
- シーン切り替え直後のクリップ: State の `bypass_image` → Stage 1 `LTXVImgToVideoInplace.bypass` が True になり、
  受け渡しフレームを固定せず REF 画像（MSR、Stage 1 でも ON）とプロンプトだけから生成（ハードカット）。
  同じシーンの中は従来どおり受け渡し + ディゾルブ
- 色の正規化はシーンごとに `scene_ref_NNN.png`（そのシーン最初のフレーム）を基準にする
- 新ノード **LTX Chain: Scene Prompt**: `scenes`（`---` 行区切りのシーン説明、順番に使い尽きたら先頭へ）+
  `action_singing` / `action_intro`（`vocals` で自動判定）→ prompt。判定は `prompts.txt` に記録
- REF 1〜4 の LoadImage（顔アップ / 上半身 / 横顔 / 全身）。キャラクターシートは 1 枚のまま入れず、
  動画と同じ縦長に切り出して別スロットへ（MSR は参照を動画解像度に合わせる。横長シートは白パディングで縮み顔が潰れる）
- `background` 参照はシーンごとに背景が違うので未接続
- **波形でシーンを切る**: State `scene_cuts`（秒のカンマ区切り。空でなければ `clips_per_scene` より優先）。
  各カットがシーン境界（ハードカット）、区間内は chunk ずつチェーン、区間の終わりで `keep_frames` によりぴったりトリム
  （`_schedule()` がクリップ計画を作る。カットなしのときは従来と同じ計画）
- 新ノード **LTX Chain: Scene Cuts (waveform)**（`web/scene_cuts.js` の DOM ウィジェット）: `audio` に LoadAudio を
  つなぐと `/view` でファイルを取って波形を描画。クリックで線、ドラッグ移動、ダブルクリック / 右クリックで削除、
  目盛りクリックで再生位置。線は `cuts` テキストに入り、出力 `scene_cuts` → State。接続先 State の
  audio_start_sec / chunk_seconds を読んで区間ごとのクリップ数を表示。`__init__.py` に `WEB_DIRECTORY = "./web"`
- Scene Cuts の **自動カット**: スペクトルフラックス → 自己相関でテンポ → 拍 → 4 拍で小節頭 → 帯域エネルギーの
  チェッカーボード新規性で構成の変わり目 → 「目標秒」付近の小節頭に線を置く（`analyzeAudio` / `autoCuts`、ブラウザ内 ~0.3 s）
- ブラウザ単体版 `wave-cutter.html`（同じ機能、mp3 をローカルで読む）も同梱
- **クリップの作り直し（redo）**: State `redo_session`（フォルダ名）+ `redo_clips`（1 始まりの番号、複数可）で Run。
  セッションの `plan.json`（clip 0 で保存する時間割）を読み、指定クリップだけ順に再生成 → `final_N.mp4` を作り直す。
  古いクリップは `<セッション>/redo/clip_NNN_KK.mp4` に退避。継ぎ目: 出だしは前クリップの `handoff`（上書きしない）、
  終わりは次クリップが使った `handoff[0]` を **End pin**（`LTXVAddGuide` frame_idx −9、`pin_end` で ifElse 切替、Stage 1）で固定。
  次のクリップも redo リストにあるときはピンせず新しい handoff を保存してつなぎ直す。シードは State `seed` から
  `seed + n*1000003 + 試行回数*7919` を配る（RandomNoise ×2 は State 接続）。v23 clip 3 で検証: 両端の継ぎ目連続、中身は別テイク。
  plan.json のない古いセッション（v24 以前）は元と同じ設定のままで redo すること
- **シーン切り替えのクロスフェード**: State `scene_crossfade`（既定 ON）。シーン切り替えクリップも、直前のシーンの
  終わりからディゾルブでつなぐ（`overlap_frames` ぶん ≒ 9 フレーム / 0.4 秒）。生成は従来どおり独立（REF から新規）だが、
  Step の継ぎ目処理で `xfade_prev`（前クリップの handoff）を渡してディゾルブを焼き込む。重なり領域を使うので長さ・音は不変。
  OFF でハードカット。長くしたいときは overlap_frames を上げる。v25（clips_per_scene 1）で確認: シーン境界がディゾルブ、
  233 フレーム = 音声と一致
- 動作確認 v21（5 秒 × 2、clips_per_scene 1）: clip 2 が T2V+MSR で別アングル（顔アップ斜め）として生成、同一人物・同じ衣装、
  ハードカット。エラーなし

## 11. 全自動版（CHAIN+MSR-SCENES-AUTO）

画像 1 枚 + mp3 を入れ替えて Run するだけ。プロンプトは画像から自動生成する。

- **画像解析**: `comfyui-florence2` の Florence2ModelLoader + Florence2Run（task = `more_detailed_caption`,
  keep_model_loaded = False）。モデルは `models/LLM/Florence-2-Flux-Large`（ダウンロード済み・軽量・解析後アンロード）
- 新ノード **LTX Chain: Auto Scenes（画像→プロンプト）** `LTXChainAutoScenes`:
  - Florence の caption を lazy 入力で受け、人物ブロック「Image 1 is the singer: …（外見・服・場所）+ 同一性固定文」に整形。
    構図語（close-up portrait / shot of など）は除去して全シーンが寄らないようにする
  - 固定カメラ雛形 `CAMERAS`（中景ロック / 顔アップ斜め / 見上げ膝上 / 横顔 / ローアングル）× `num_scenes` と組み合わせて
    `scenes`（`---` 区切り）を生成。`action_singing` / `action_intro` も出力 → Scene Prompt に直結
  - 画像ハッシュ + num_scenes + style でキャッシュ（`output/LTX2.5Chains/_autoprompt/scenes_*.json`）。`check_lazy_status` で
    キャッシュがあれば Florence を評価しない。初回のみ解析し、直後に `unload_all_models()`
- ワークフロー `LTX-2.5-lip sync-CHAIN+MSR-SCENES-AUTO.json`: 上記 3 ノードを配線。**MSR の pic1（Stage 1/2）も開始画像
  （node 23）を共用**、REF 2〜4 はバイパス。差し替えるのは画像 1 枚 + mp3 だけ
- QwenVL は HF ダウンロード方式（HDD 逼迫のため不採用）。Florence-2 を採用
- 動作確認 v26（Florence 解析 + LTX 生成が同居して RAM クラッシュなし、人物一致、クロスフェード、233 フレーム = 音声一致）
- Auto Scenes の追加パラメータ: `num_singers`（1=ソロ / 2=デュエット掛け合い＋ハモリ / 3+=グループ）、`ethnicity`（国籍・人種を
  人物説明に明記。例 Japanese / Japanese and French mixed。空なら画像の見た目のまま）、`performance`（歌い方の強さ:
  subtle=控えめ / restrained=抑えめ・既定 / natural / expressive）。LTX は口を開けすぎる癖があるため既定を restrained にし、
  全ワークフローのネガティブに `exaggerated facial expression, wide open mouth, over-acting, grimacing, screaming, ...` を追加。
  ただし大きなロングトーンでは正しいリップシンクとして口が開く（完全に潰すと口パクずれに見える）。さらに抑えるなら subtle +
  顔アップシーンを減らす。これらはキャッシュキーに含まれるので変更で自動再生成
- `microphone`（auto / yes / no、既定 auto）: 以前はカメラ雛形と action 文のすべてに microphone がハードコードされていて、
  写真に無いマイクが必ず足されていた。`CAMERAS_MIC` / `CAMERAS_NOMIC` の2セット + `ACTION_INTRO_NOMIC` を用意し、
  auto は Florence の caption に microphone/mic の語があるかで判定（`_wants_mic`）。no のときは「microphone」という語を
  一切プロンプトに出さない（語があるだけで LTX が足す）。キャッシュキーに含む。AUTO ワークフローの extra_cameras 例からも
  「the microphone between her and the camera」を外した
- `gender`（auto / female / male、既定 auto）: auto は caption の最初の性別語（woman/girl/she… / man/boy/he…）で判定（`_detect_gender`）。
  female/male を指定すると人物タグに woman/man を明記し、caption 内の性別語も `TO_FEMALE`/`TO_MALE` で書き換える
  （`_force_gender`。Florence が短髪女性を man と読んだ時にプロンプト内で矛盾しないように）
- `emotion`（none + 12 プリセット: happy/joyful/tender/sad/melancholic/nostalgic/passionate/serene/playful/confident/dreamy/angry）
  + `emotion_custom`（自由記述、非空なら優先）: `EMOTIONS` は (歌唱中の文, 非歌唱区間の文) のペア。歌唱中は performance 句の後に
  「The performance carries …」、intro は「The face keeps …」として挿入。これらもキャッシュキーに含む。
  新規入力は widgets_values の位置ずれを避けるため既存入力の後ろ（extra_cameras の下）に追加する方針
- nostalgic を選んでも笑顔が多い問題: (1) プリセット文自体に「a faint bittersweet smile」があった → 非笑顔系 7 種
  （`NO_SMILE`: sad/melancholic/nostalgic/passionate/serene/dreamy/angry/serious。serious=真面目 は後から追加）は smile 語を消し「unsmiling / not smiling」を明記。
  (2) Florence の caption の「She is smiling」が人物ブロックとして全シーンに繰り返されるのが主因 → 非笑顔系のときは
  `_strip_smile` で caption から smiling/laughing/with a big smile 等を正規表現で除去（custom は `NO_SMILE_WORDS` を含むとき）
- `motion`（none / standing still / gentle sway / hand gestures / light dance in place / dancing / walking toward camera /
  walking sideways / strolling / sitting）+ `motion_custom`: `MOTIONS` は (歌唱中, 非歌唱, カメラ注記) の3組。カメラ注記がある
  動き（歩く・踊る）では雛形の「Camera locked.」を追従カメラ文に置換（固定カメラと矛盾させない）。sitting は「standing upright」
  →「seated」。歩く/踊るときは microphone=no 推奨（マイクスタンドと矛盾）
- `height`（none / petite / short / average height / tall / very tall）と `build`（none / very slim / slim / athletic / average /
  curvy / chubby / plump / heavy）: 人物タグの形容詞として「(a tall, slim 25-year-old Japanese woman)」の形で入れ、
  「The singer's height and body shape stay exactly the same in every shot.」を追加。指定時は caption 内の矛盾する語
  （`HEIGHT_WORDS` / `BUILD_WORDS`: tall/short/slim/plump…）を `_strip_words` で除去。キャッシュキーに含む
- 歩く motion がシーン切替ごとに静止→歩き出しになる件: シーン先頭クリップは前フレームを引き継がず MSR 参照（静止写真）
  だけから生成される構造のため（`use_msr_stage1 = ... or scene_start`、start frame 無し）。対策は文言のみ: 動き系プリセットに
  「already mid-stride at the very first frame, no pause and no standing start」を追加
- `camera`（auto / locked / slow push-in / slow pull-out / dolly left / dolly right / orbit / crane up / crane down / handheld /
  follow / mix (varies per scene)）+ `camera_custom`: `CAMERA_MOVES` の文で雛形の「Camera locked.」系を `_apply_camera` で
  置換（extra_cameras の行に固定文が無ければ末尾に追記）。優先順位 custom > preset > motion の追従カメラ > 固定。
  mix は `CAMERA_MIX` をアングル番号順に割り当て。push-in/pull-out/crane は同一シーン内でクリップをまたいで蓄積する
  （次クリップは前フレームの続きから）ので「very slowly, subtle」表現＋ tooltip で注意。
  ReelBids camera LoRA（dolly-in 専用・654MB）は不採用: ドリーインのみ、チェーンで寄りが蓄積、LoRA 3本目で VRAM/画質懸念
- orbit が効かなかった件（v10）: 置換が「Camera locked.」1文だけで、雛形冒頭の「Locked medium shot, camera level and still」が
  残り矛盾していた＋150語の caption の後ろに埋もれていた。`_apply_camera` を作り直し: locked/still 系の語を全部除去し、
  カメラ文をショット行の**先頭**に置く。さらに action_singing / action_intro の**末尾**に「Camera: …」として再掲
  （プロンプトの先頭付近と末尾が最も効く）。動く motion のときは performance の「barely any head or body movement」→
  「barely any head movement」に緩める。orbit + walking toward camera は意味的に矛盾するので auto(follow) か standing still 推奨
- very tall なのに全身ショットで子供体型（頭大・脚短）になる件（v12 clip 15〜）: MSR は顔だけ固定し体のプロポーションは
  自由。「very tall」の形容詞1語は比較対象のない単独ショットでは無意味。→ `HEIGHTS` / `BUILDS` を (形容詞, 体の説明) にし、
  「long-legged adult frame, the head small in proportion to the body, about eight and a half heads tall, never child-like」
  のようにプロポーションで記述＋「in every framing, including full-body shots」で固定。全4ワークフローのネガティブに
  「wrong body proportions, oversized head, tiny body, short stubby legs, chibi, dwarf, shrunken figure」を追加。
  根本対策は MSR REF 2 に全身写真を入れること（AUTO では REF2〜4 バイパス中）
- 顔ドリフト対策として `LTX-2.5-lip sync-CHAIN+MSR-SCENES-AUTO+ReActor.json` を追加: VAE Decode(74) → ReActorFaceSwap(206,
  source = REF 1 LoadImage 182) → Chain Step(179)。link 206 の始点を差し替え、304/305 を追加。クリップ単位（240 フレーム）で
  回すので 4 分動画の一括ロードを避けられ、hand-off も補正後 → 次クリップが正しい顔から生成される。既定 face_restore=none
  （速い）、顔アップが柔らかければ GFPGANv1.4 + visibility 0.5〜0.7。comfy validate は古い object_info を見るので
  LTXChain 系の unknown_class_type は無視（curl の object_info では存在する）
- 後がけ ReActor の結果: 顔は固定されるが (1) inswapper は口を閉じがち（ソース写真の表情を持ち込む）、(2) 口が大きく開くと
  出力が急変して顔が飛ぶ。hyperswap_256 でも同傾向。→ `LTXChainFaceKeepMouth`（buffalo_l の 5 点ランドマークから口の楕円
  マスクを作り元フレームを戻す）を追加したが、質感の差で貼り付け感は残る。
- 決定版 = **顔アンカー方式**: Step に `handoff_images`（任意）を追加。Decode → `LTXChainLastFrames`（hand-off 枚数だけ）→
  ReActor → KeepMouth → Step.handoff_images。保存クリップは images（純 LTX）のまま、hand-off だけ補正後になるので
  次クリップが正しい顔から始まる。補正は継ぎ目の 9 フレーム・ディゾルブで吸収。ReActor は 9 フレーム分で数秒。
  AUTO+ReActor ワークフローはこの配線に変更済み（ComfyUI 側 user/default/workflows にも同じものを配置）。
- onnxruntime: onnxruntime / onnxruntime-gpu / onnxruntime-openvino が同居して OpenVINO 版が GPU 版を隠していた
  （CPU 実行で ReActor が ~1 fps）。3 つ削除 → onnxruntime-gpu 1.26.0 のみ再インストールで CUDA EP 有効。それでも
  ReActor はフレーム単位の Python ループなので 512x896 で ~2 fps（4 分動画 ≈ 52 分）。
- v03 clip 14-15 の体型崩れ: mix の「slow pull-out」＋ strolling で人物が小さくなり再描画で子供体型に。mix から
  push-in/pull-out を除外、歩く系 motion では `CAMERA_MIX_MOVING`（follow/handheld/横ドリー）のみ。plan.json に
  msr_clips と生成解像度を記録するようにした
- v04（704x704・msr_clips=all）: 各クリップが引き→顔アップに収束、グレーのピラーボックス。原因 (1) 正方形サイズに縦長参照 →
  MSR の余白をモデルが描く、(2) Stage 1 に顔アップ参照（REF1〜3）が毎クリップ入り構図が顔アップへ引き戻される。
  → AUTO+ReActor の Stage 1 MSR Guide は pic1（全身）だけに配線変更、Stage 2 は全参照のまま
- State: `resolution` にテスト用小サイズ（448x832 / 384x704 / 448x576 / 512x512 / 832x448 / 704x384、いずれも 64 倍数）。
  `web/resolution_notice.js` が (test) 選択時に app.ui.dialog で注意を表示（通常→test に切り替えた時だけ）
- State: `scene_switching`（BOOLEAN、末尾に追加＝widgets_values の位置ずれ回避）。OFF で cuts/clips_per_scene を無視し 1 シーン
- 前作の背景が出た件: 原因は Auto Scenes の `style` 欄に残っていた「post-apocalyptic … ruined-city」。作品を変えるときは
  開始画像・mp3・REF・style の 4 点を見直す
- Auto Scenes `outfit`: 参照が太ももで切れると膝下を勝手に補完（短パン化）→ 服装の補足文を人物ブロックに追加し
  「The full outfit, including the legs and shoes, stays exactly the same…」で固定。jeans/pants を書くと caption の shorts を除去。
  Florence の text_input は more_detailed_caption では無視されるので使わない
- Auto Scenes `look` / `lighting` / `setting` プリセット（`LOOKS` / `LIGHTINGS` / `SETTINGS`）: 末尾 = look, lighting, style の順で
  結合。setting を写真以外にすると人物ブロックの「and the setting stay the same」を「Setting: …, replacing the background of
  the reference photo」に置換。既定は custom / auto なので既存ワークフローの挙動は不変。`art("")` が "an" を返す
  バグ（an duet）も修正
- Auto Scenes `weather` / `weather_when`: `WEATHERS` の文を該当ブロックのショット行末尾に「Weather: … through the whole shot」
  として追加（クリップ途中で消えにくい書き方）。`WEATHER_WHEN` はブロック番号 i と総数 n の述語（全部 / i%2 / i%3 / 最初 /
  最後 / 最初以外）。シーンはブロックを巡回するので「時々」はシーン単位で効く。同一シーン内は hand-off で粒子が続く
- Auto Scenes `vfx` / `vfx_when` / `vfx_custom`: `VFX` 20 種をショット行末尾に「VFX: …」。`VFX_WHEN` は (n, rng) → ブロック
  index の集合を返す（random 系は必ず 1 つ以上、`_key(image, num_scenes, vfx文)` を種にした固定 RNG → redo でも同じシーン）。
  雷・ストロボ・カメラフラッシュは Step の `_normalize_color`（3 フレーム平滑の毎フレーム正規化）に打ち消されるので
  handoff_color_match ≤ 0.3 を案内
- ズームアウトと顔崩れ: 公式仕様は無いが、決めるのは「出力フレーム内の顔の高さ px」。Stage 1 = 出力の半分、VAE = 32px/セル →
  出力で顔 120px ≒ Stage 1 で潜在 2 セルが下限（実測: ≥120 安定 / 90-120 境界 / <90 崩れ）。構図別の顔比率
  `FRAMING_FACE`（close-up .45 / chest-up .30 / waist-up .22 / mid-thigh .15 / full body .10 / wide .06）。
  Auto Scenes `framing_limit` で内蔵アングル（`ANGLE_FRAMING`）を上限より広いものを除外。`web/framing_notice.js` が
  State の resolution × framing_limit から推定 px をノード下部に **onDrawForeground で描画**（最初は text ウィジェットにしたら
  widgets_values の 28 番目に保存されてしまい、将来の入力追加でずれる原因になるので描画方式に変更・保存済み JSON からも除去）
- motion に `running toward camera` / `jogging toward camera`（カメラ注記: 同速で後退＋手持ち感）。走りの写真を pic1（Stage 1）に
  すると毎クリップ「走る姿勢」に引き戻されるので走りが止まりにくい（参照のポーズ引力を逆利用）
- ReActor `enabled=OFF` で IndexError: ReActor は無効時に 2 出力しか返さない（RETURN_TYPES は 3）バグ。KeepMouth.original を
  ReActor.ORIGINAL_IMAGE ではなく上流（Last Frames / Load Video）から直接取る配線に変更（全 ReActor 系ワークフロー、
  ComfyUI 側の作品別コピーも含む）。ReActor の 3 番目の出力には何も繋がないこと
- State `face_anchor`（BOOLEAN 入力＋出力 index 16）→ ReActor.enabled（入力化）に配線。State だけで ON/OFF。
  全 AUTO+ReActor ワークフロー（作品別コピー含む）に配線済み
- Step で「weight is on cpu, other tensors on cuda:0」: ReActor の出力が CUDA テンソルで KeepMouth 経由で handoff_images に
  入り、`_normalize_color` の CPU カーネルと衝突。KeepMouth 出力を .cpu()、Step で handoff_images.cpu()、
  `_normalize_color` はカーネル・参照を frames.device に置くよう修正
- v14 clip 19: 間奏なのに vocal ratio 0.32（歌声分離への楽器漏れ）で singing 判定。Scene Prompt に `force_intro_clips` /
  `force_singing_clips`（クリップ番号、範囲可、`_clip_list`）を追加して自動判定を上書き。ログに (forced intro) と出る

## 9. このフォルダの中身（Video-Sticher プロジェクト内のバックアップ）

```
LTX2.5-LIPSYNC-MSR/
  memo.md                              このメモ
  LTX-2.5-lip sync-CHAIN.json          チェーン生成ワークフロー
  LTX-2.5-lip sync-CHAIN+MSR.json      チェーン生成 + MSR ワークフロー
  LTX-2.5-lip sync-CHAIN+MSR-SCENES.json  シーン切り替え版（§10）
  LTX-2.5-lip sync-CHAIN+MSR-SCENES-AUTO.json  全自動版（§11: 画像+mp3 だけ）
  wave-cutter.html                     波形カッター（ブラウザ単体版）
  ComfyUI-LTX-Chain/                   カスタムノードのソース（nodes.py, __init__.py）
```

別の ComfyUI に入れるときは `ComfyUI-LTX-Chain/` を `ComfyUI/custom_nodes/` にコピーして再起動、
JSON はブラウザにドラッグ＆ドロップで開ける。前提パック: comfyui-kjnodes（Set/Get）, comfyui-easy-use（easy int / ifElse）,
ComfyUI-LTX2.5-MSR（MSR 版のみ）, comfyui-impact-pack, ComfyUI-MelBandRoFormer, comfy-mtb, derfuu_comfyui_moddednodes。
