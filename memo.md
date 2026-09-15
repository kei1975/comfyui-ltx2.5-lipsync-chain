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
