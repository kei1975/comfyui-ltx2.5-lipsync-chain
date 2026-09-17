"""
LTX Chain: generate a long video as a chain of short clips.

Each run produces one clip. The output node saves the clip, keeps the
last frame, and re-queues the same workflow with the next clip index so the
last frame becomes the reference image of the next clip. When the last
clip finishes, all clips are concatenated (dropping the duplicated
first frame of every clip after the first) and muxed with the original
audio into one final mp4.

Everything for one run lands in  output/LTX2.5Chains/<YYYYMMDD>_v<NN>/
    clip_001.mp4, clip_002.mp4, ...   (each clip, with its slice of audio)
    handoff_000.npy / last_frame_000.png   (hand-off frames)
    final.mp4                         (all clips joined + the original audio)
"""
import copy
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import urllib.request
from fractions import Fraction

import av
import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import InputImpl, Types
from server import PromptServer

CHAIN_TYPE = "LTX_CHAIN"
STATE_NODE_CLASS = "LTXChainState"
CHAINS_SUBDIR = "LTX2.5Chains"
CHUNK_CRF = 12
MIN_TAIL_SECONDS = 1.0


def _audio_seconds(audio):
    return audio["waveform"].shape[-1] / float(audio["sample_rate"])


def _chains_root():
    return os.path.join(folder_paths.get_output_directory(), CHAINS_SUBDIR)


def _session_dir(session):
    d = os.path.join(_chains_root(), session)
    os.makedirs(d, exist_ok=True)
    return d


def _new_session_name():
    """<YYYYMMDD>_v<NN>: the next unused version folder for today."""
    date = time.strftime("%Y%m%d")
    root = _chains_root()
    used = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            m = re.fullmatch(rf"{date}_v(\d+)", name)
            if m:
                used.append(int(m.group(1)))
    return f"{date}_v{(max(used) + 1) if used else 1:02d}"


# LTX-2.5-safe output sizes (all multiples of 64 so the half-res stage-1 latent grid divides them).
# Areas kept near ~500-750k px, which is safe on 12 GB VRAM at ~10 s clips.
RES_PRESETS = [
    "auto (match image / 画像の比率)",
    "576x1024 — portrait 9:16",
    "512x896 — portrait 9:16 (light)",
    "640x1152 — portrait 9:16 (large)",
    "576x768 — portrait 3:4",
    "640x832 — portrait 3:4",
    "704x704 — square 1:1",
    "768x768 — square 1:1 (large)",
    "768x576 — landscape 4:3",
    "832x640 — landscape 4:3",
    "1024x576 — landscape 16:9",
    "896x512 — landscape 16:9 (light)",
    # test sizes (~480p and below): fast previews of motion / camera / scene cuts. Faces get very
    # few pixels here, so identity, lip sync and body proportions are NOT representative.
    "448x832 — portrait 9:16 (test ~480p)",
    "384x704 — portrait 9:16 (test tiny)",
    "448x576 — portrait 3:4 (test)",
    "512x512 — square 1:1 (test)",
    "832x448 — landscape 16:9 (test ~480p)",
    "704x384 — landscape 16:9 (test tiny)",
]


def _parse_resolution(res):
    """'576x1024 — ...' -> (576, 1024) snapped to 64; returns None for the 'auto' preset."""
    if not res or res.startswith("auto"):
        return None
    tok = res.split()[0]
    try:
        w, h = (int(v) for v in tok.lower().split("x"))
    except ValueError:
        return None
    return max(64, w // 64 * 64), max(64, h // 64 * 64)


def _target_size(width, height, gen_w, gen_h, multiple=64):
    """Keep the input aspect ratio, use about gen_w*gen_h pixels, snap to `multiple`."""
    aspect = height / width
    area = gen_w * gen_h
    h = int(math.sqrt(area * aspect) // multiple * multiple)
    w = int(math.sqrt(area / aspect) // multiple * multiple)
    return max(w, multiple), max(h, multiple)


def _fit_image(img, w, h):
    """Resize (lanczos) keeping aspect, then centre-crop to exactly w x h. img: [B, H, W, 3]."""
    import comfy.utils
    B, H, W, _ = img.shape
    if (W, H) == (w, h):
        return img
    scale = max(w / W, h / H)
    rw, rh = max(w, int(round(W * scale))), max(h, int(round(H * scale)))
    x = comfy.utils.common_upscale(img.movedim(-1, 1), rw, rh, "lanczos", "disabled")
    top, left = (rh - h) // 2, (rw - w) // 2
    return x[:, :, top:top + h, left:left + w].movedim(1, -1).contiguous()


def _match_color(img, target, strength):
    """Per-channel mean/std transfer (Reinhard-style, in RGB) of `img` toward `target`.
    Keeps the hand-off frame from drifting darker / more saturated clip after clip."""
    x = img.float()
    t = target.float()
    mean_x, std_x = x.mean(dim=(0, 1, 2)), x.std(dim=(0, 1, 2)) + 1e-6
    mean_t, std_t = t.mean(dim=(0, 1, 2)), t.std(dim=(0, 1, 2)) + 1e-6
    matched = (x - mean_x) / std_x * std_t + mean_t
    out = x + (matched - x) * float(strength)
    return out.clamp(0.0, 1.0)


def _normalize_color(frames, target, strength, window=3):
    """Per-frame, per-channel mean/std normalisation of `frames` toward `target`.

    The generated frames of every clip come out a little darker / flatter than the frames the
    model was conditioned on, so both slow drift and a step at every seam appear.  Mapping each
    frame's global statistics to the reference (statistics smoothed over `window` frames so
    content motion does not pump the exposure) removes both without touching the generation."""
    x = frames.float()
    t = target.float().to(x.device)
    mean_t = t.mean(dim=(0, 1, 2))
    std_t = t.std(dim=(0, 1, 2)) + 1e-6
    mean_i = x.mean(dim=(1, 2))                      # [T, 3]
    std_i = x.std(dim=(1, 2)) + 1e-6                 # [T, 3]
    if window > 1 and x.shape[0] > 1:
        w = min(window, x.shape[0])
        w += (w + 1) % 2                             # odd
        pad = w // 2
        kernel = torch.ones(3, 1, w, dtype=x.dtype, device=x.device) / w
        def smooth(v):                               # [T, 3] -> [T, 3], edge padded
            v = v.t().unsqueeze(0)                   # [1, 3, T]
            v = torch.nn.functional.pad(v, (pad, pad), mode="replicate")
            return torch.nn.functional.conv1d(v, kernel, groups=3)[0].t()
        mean_i, std_i = smooth(mean_i), smooth(std_i)
    gain = (std_t / std_i).view(-1, 1, 1, 3)
    bias = (mean_t - mean_i * (std_t / std_i)).view(-1, 1, 1, 3)
    corrected = x * gain + bias
    return (x + (corrected - x) * float(strength)).clamp(0.0, 1.0)


def _parse_cuts(text):
    """'12.5, 24 40.2' -> sorted unique floats."""
    vals = []
    for tok in re.split(r"[\s,;]+", (text or "").strip()):
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            raise ValueError(f"scene_cuts: '{tok}' is not a number")
    return sorted(set(vals))


def _schedule(audio_start, target, chunk, step, fps, cuts):
    """Clip plan for the whole run: list of dicts {start, num_seconds, scene, scene_start, seg_end, keep_frames}.

    Without cuts: one segment, a final remainder shorter than MIN_TAIL_SECONDS is dropped (as before).
    With cuts: every cut is a scene boundary (hard cut); each segment is chained in chunk-sized clips
    that overlap by `chunk - step`, and the last clip of a segment is trimmed to the boundary."""
    end = audio_start + target
    bounds = [audio_start] + [c for c in cuts if audio_start + 0.5 < c < end - 0.5] + [end]
    # drop boundaries that would leave a segment shorter than 0.5 s
    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] >= 0.5:
            clean.append(b)
    if clean[-1] != end:
        clean[-1] = end
    plan = []
    for k in range(len(clean) - 1):
        seg_start, seg_end = clean[k], clean[k + 1]
        L = seg_end - seg_start
        if L <= chunk:
            m = 1
        else:
            extra = (L - chunk) / step
            m = int(math.ceil(extra - 1e-9)) + 1
            if not cuts:  # legacy rule: a remainder < MIN_TAIL_SECONDS is not worth a clip
                n_full = int(extra)
                tail = (extra - n_full) * step
                m = 1 + n_full + (1 if tail >= MIN_TAIL_SECONDS else 0)
        for j in range(m):
            start = seg_start + j * step
            remaining = seg_end - start
            num_seconds = int(max(1, min(chunk, math.ceil(remaining - 1e-6))))
            last_of_seg = j == m - 1
            plan.append({
                "start": start,
                "num_seconds": num_seconds,
                "scene": k,
                "scene_start": k > 0 and j == 0,
                "seg_end": seg_end,
                # last clip of a segment: cut exactly at the boundary; otherwise Step drops the hand-off tail
                "keep_frames": int(round(remaining * fps)) if last_of_seg else None,
            })
    return plan


class LTXChainState:
    """Decides which clip this run generates and which image to start from."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "audio_start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                                              "tooltip": "mp3 のどこから使い始めるか（秒）"}),
                "chunk_seconds": ("INT", {"default": 5, "min": 1, "max": 60,
                                          "tooltip": "1回の生成で作る秒数"}),
                "length_mode": (["all", "seconds"], {"default": "all",
                                                     "tooltip": "all = mp3 の最後まで / seconds = 下の秒数だけ"}),
                "length_seconds": ("INT", {"default": 60, "min": 1, "max": 100000,
                                           "tooltip": "length_mode が seconds のとき、合計で作る秒数"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "resolution": (RES_PRESETS, {"default": RES_PRESETS[0],
                                             "tooltip": "書き出しサイズ。LTX-2.5 が安全に出せる 64 の倍数のサイズから選ぶ。auto = 画像の縦横比のまま generation_width×height の面積で決める。数値を選ぶとその WxH で書き出し（画像はその比率に中央クロップ）。大きいほど VRAM を使うので 12GB では ~600k px 目安。(test) 付きは動き・カメラ・シーン切替を速く確認するための小サイズ（480p 相当以下）。顔の画素が少ないので顔の一貫性・リップシンク・体型の判断には使わないこと。本番と同じセッションで redo には使えない（サイズが混ざる）"}),
                "generation_width": ("INT", {"default": 609, "min": 64, "max": 4096,
                                             "tooltip": "生成サイズの目安（幅）。縦横比は入力画像のまま、面積がこの幅×高さになるよう 64 の倍数に丸めます"}),
                "generation_height": ("INT", {"default": 1056, "min": 64, "max": 4096,
                                              "tooltip": "生成サイズの目安（高さ）"}),
                "msr_clips": (["stage2_all", "first_only", "all"], {"default": "stage2_all",
                                                                     "tooltip": "MSR（顔参照）の使い方。stage2_all = Stage 1 は1本目だけ・Stage 2 は毎クリップ（推奨：継ぎ目を崩さず毎回顔を参照）/ first_only = 1本目だけ / all = 両ステージ毎クリップ（継ぎ目でスナップする）"}),
                "clips_per_scene": ("INT", {"default": 0, "min": 0, "max": 1000,
                                            "tooltip": "0 = 1 シーン（従来どおり）。N = N クリップごとにシーン（アングル）を切り替える。切り替え直後のクリップは受け渡しフレームを使わず、MSR の参照だけから新しく生成します（Scene Prompt ノードと組み合わせる）"}),
                "scene_cuts": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "シーンの切り替え位置（mp3 上の秒、カンマ区切り。例 12.5, 24, 40.2）。空でなければ clips_per_scene より優先。各区間は chunk_seconds ずつつないで生成し、区間の終わりでぴったり切る（波形カッターの出力を貼る）"}),
                "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 32, "step": 8,
                                           "tooltip": "前のクリップの最後の「1 + この枚数」のフレームを次のクリップの出だしに固定して動きを引き継ぐ（8 の倍数。8 推奨。0 = 最後の1枚だけ）"}),
                "handoff_color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                  "tooltip": "受け渡しフレームの色味を最初の参照画像に合わせる強さ（0 = なし）。クリップを重ねると色が濃くなっていくのを抑えます"}),
                "scene_crossfade": ("BOOLEAN", {"default": True,
                                                "tooltip": "シーンが切り替わるクリップも、直前のシーンの終わりからクロスフェード（ディゾルブ）でつなぐ。オフ = ハードカット。継ぎ目の長さは overlap_frames ぶん（既定 9 フレーム ≒ 0.4 秒）"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "基準シード。クリップごとに番号を足した値を seed 出力に出します（RandomNoise につなぐ）。作り直しのときは自動で変えます"}),
                "redo_session": ("STRING", {"default": "", "multiline": False,
                                            "tooltip": "作り直すセッションのフォルダ名（例 20260914_v22）。redo_clips と一緒に使う"}),
                "redo_clips": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "作り直すクリップ番号（clip_003 なら 3。カンマ区切りで複数可）。空なら通常の生成。終わったら空に戻す"}),

                "chain_iter": ("INT", {"default": 0, "min": 0, "max": 100000,
                                       "tooltip": "内部用。手動実行のときは 0 のままにしてください（自動で進みます）"}),
                "chain_state": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "内部用。手動実行のときは空のままにしてください"}),
                "scene_switching": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF",
                                                "tooltip": "シーン切替のワンボタン。OFF にすると scene_cuts と clips_per_scene を無視して 1 シーン（最初のカメラアングル）で最後まで生成。設定は消さずに残るので、ON に戻せば元どおり"}),
                "face_anchor": ("BOOLEAN", {"default": True, "label_on": "ReActor ON", "label_off": "OFF",
                                            "tooltip": "顔アンカー（ReActor）のワンボタン。face_anchor 出力を ReActor ノードの enabled につないでおくと、ここで ON/OFF できます。OFF = hand-off に顔補正をかけない（従来どおり）。ReActor の無いワークフローでは無視"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "INT", CHAIN_TYPE, "STRING", "INT", "INT", "BOOLEAN", "BOOLEAN", "INT", "BOOLEAN",
                    "INT", "IMAGE", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("image", "start_sec", "num_seconds", "chunk_index", "total_chunks", "chain", "info", "width", "height",
                    "use_msr_stage1", "use_msr_stage2", "scene_index", "bypass_image", "seed", "end_image", "pin_end", "face_anchor")
    OUTPUT_TOOLTIPS = (None, None, None, None, None, None, None, None, None, None, None,
                       "このクリップのシーン番号（0 始まり）。Scene Prompt ノードが使う",
                       "True = シーン切り替え直後のクリップ。Stage 1 の LTXVImgToVideoInplace の bypass につなぐ（受け渡しフレームを使わず参照だけから生成）",
                       "このクリップ用のシード（RandomNoise の noise_seed につなぐ）",
                       "作り直しのとき、次のクリップとの継ぎ目に合わせるための終端フレーム（LTXVAddGuide の image につなぐ）",
                       "True = 終端フレームを固定する（作り直しで次のクリップがあるとき）。AddGuide の切り替えスイッチにつなぐ",
                       "face_anchor の値をそのまま出力。ReActor ノードの enabled（入力に変換）につなぐ")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    def run(self, image, audio, audio_start_sec, chunk_seconds, length_mode, length_seconds,
            fps, chain_iter, chain_state, resolution=RES_PRESETS[0], generation_width=609, generation_height=1056,
            msr_clips="stage2_all", clips_per_scene=0, scene_cuts="", overlap_frames=8, handoff_color_match=1.0,
            scene_crossfade=True, seed=42, redo_session="", redo_clips="", scene_switching=True, face_anchor=True):
        available = max(0.0, _audio_seconds(audio) - audio_start_sec)
        target = available if length_mode == "all" else min(float(length_seconds), available)
        if target <= 0:
            raise ValueError(f"audio_start_sec ({audio_start_sec}s) is beyond the end of the audio "
                             f"({_audio_seconds(audio):.1f}s)")
        overlap = int(overlap_frames) - int(overlap_frames) % 8  # 1 + overlap frames = whole latent frames
        handoff = overlap + 1
        # every clip after the first re-generates its first `handoff` frames from the previous
        # clip, so each additional clip only adds chunk_seconds - overlap/fps of new video
        step = chunk_seconds - overlap / fps
        cuts = _parse_cuts(scene_cuts)
        if not scene_switching:
            cuts, clips_per_scene = [], 0
            logging.info("[LTX Chain] scene switching OFF: one scene for the whole run")
        if cuts:
            plan = _schedule(audio_start_sec, target, chunk_seconds, step, fps, cuts)
        else:
            plan = _schedule(audio_start_sec, target, chunk_seconds, step, fps, [])
            cps = int(clips_per_scene)
            for i, c in enumerate(plan):
                c["scene"] = i // cps if cps > 0 else 0
                c["scene_start"] = cps > 0 and i % cps == 0 and i > 0
                # cut mode trims at segment ends; in clips_per_scene mode the grid continues across
                # the cut, so only the final clip is trimmed (to the end of the audio)
                c["keep_frames"] = c["keep_frames"] if i == len(plan) - 1 else None
        # Redo: regenerate the listed clips of an existing session. The session's saved plan is used
        # so the clip grid stays identical whatever the current widgets say.
        redo_list = [int(v) - 1 for v in _parse_cuts(redo_clips)] if redo_clips.strip() else []
        redo = bool(redo_list)
        if redo:
            if not redo_session.strip():
                raise ValueError("redo_clips が指定されていますが redo_session（フォルダ名）が空です")
            sdir_redo = os.path.join(_chains_root(), redo_session.strip())
            if not os.path.isdir(sdir_redo):
                raise FileNotFoundError(f"redo_session not found: {sdir_redo}")
            plan_path = os.path.join(sdir_redo, "plan.json")
            if os.path.isfile(plan_path):
                with open(plan_path, encoding="utf-8") as f:
                    plan = json.load(f)["plan"]
            else:
                logging.warning("[LTX Chain] redo: plan.json not found, using the current settings (must match the original run)")
            bad = [i + 1 for i in redo_list if i < 0 or i >= len(plan)]
            if bad:
                raise ValueError(f"redo_clips: clip {bad} does not exist (session has {len(plan)} clips)")
        total_chunks = len(plan)
        redo_index = min(chain_iter, len(redo_list) - 1) if redo else 0
        n = redo_list[redo_index] if redo else min(chain_iter, total_chunks - 1)
        cur = plan[n]
        scene_index, scene_start = cur["scene"], cur["scene_start"]

        # Output size: aspect of the reference image, ~generation_width*height pixels, multiples of
        # 64 so the half-resolution stage-1 latent grid divides it exactly. Every later clip is fed
        # its predecessor's frames at exactly this size -> no per-clip resize/crop (= no zoom drift).
        preset = _parse_resolution(resolution)
        if preset is not None:
            out_w, out_h = preset  # explicit size; the image is centre-cropped to this aspect below
        else:
            out_w, out_h = _target_size(image.shape[2], image.shape[1], generation_width, generation_height)
        image = _fit_image(image, out_w, out_h)

        # Scenes: the first clip of a scene (except clip 0, which starts from the input image) is
        # generated from the MSR references alone, so the previous clip's frames are not pinned and
        # the seam is a hard cut.
        state = json.loads(chain_state) if chain_state else {}
        if redo:
            session = redo_session.strip()
            prev = os.path.join(sdir_redo, f"handoff_{n - 1:03d}.npy") if n > 0 and not scene_start else None
            if prev is None:
                ref = image
                prev_handoff = None
            else:
                if not os.path.isfile(prev):
                    raise FileNotFoundError(f"redo: hand-off frames of clip {n} not found: {prev}")
                ref = torch.from_numpy(np.load(prev).astype(np.float32) / 255.0)
                prev_handoff = prev
        elif n == 0 or not state:
            session = _new_session_name()
            sdir = _session_dir(session)
            ref = image
            prev_handoff = None
            n = 0
            Image.fromarray((image[0] * 255).clamp(0, 255).byte().cpu().numpy()).save(
                os.path.join(sdir, "scene_ref_000.png"), compress_level=1)
            with open(os.path.join(sdir, "plan.json"), "w", encoding="utf-8") as f:
                json.dump({"settings": {"audio_start_sec": audio_start_sec, "chunk_seconds": chunk_seconds, "fps": fps,
                                        "overlap": overlap, "target": target, "cuts": cuts, "clips_per_scene": int(clips_per_scene),
                                        "msr_clips": msr_clips, "generation": [out_w, out_h]},
                           "plan": plan}, f, ensure_ascii=False, indent=1)
        else:
            session = state["session"]
            prev = state["prev_frames"]
            if not os.path.isfile(prev):
                raise FileNotFoundError(f"previous clip's hand-off frames not found: {prev}")
            ref = torch.from_numpy(np.load(prev).astype(np.float32) / 255.0)  # [K, H, W, 3], already colour-corrected
            if ref.shape[1] != out_h or ref.shape[2] != out_w:
                logging.warning(f"[LTX Chain] hand-off frames are {ref.shape[2]}x{ref.shape[1]}, expected {out_w}x{out_h}; refitting")
                ref = _fit_image(ref, out_w, out_h)
            prev_handoff = prev

        start_sec = cur["start"]
        num_seconds = cur["num_seconds"]

        # dissolve source for a scene-start clip = the previous clip's hand-off frames
        xfade_prev = None
        if scene_start and n > 0:
            xfade_prev = os.path.join(sdir_redo, f"handoff_{n - 1:03d}.npy") if redo else prev_handoff
            if not (xfade_prev and os.path.isfile(xfade_prev)):
                xfade_prev = None

        chain = {
            "session": session,
            "index": n,
            "total": total_chunks,
            "chunk_seconds": chunk_seconds,
            "num_seconds": num_seconds,
            "fps": fps,
            "overlap": overlap,
            "handoff": handoff,
            "audio_start_sec": audio_start_sec,
            "target_seconds": target,
            "audio": audio,
            "reference": image,
            "color_match": float(handoff_color_match),
            # Scene change: dissolve from the end of the previous scene into this clip's head so the
            # cut is a cross-fade, not a jump. The previous clip's hand-off frames overlap this
            # clip's first `handoff` frames in time, so the blend keeps the length (and lip sync).
            "prev_handoff": (xfade_prev if scene_crossfade else None) if scene_start else prev_handoff,
            "scene_index": scene_index,
            "scene_start": scene_start,
            "start_sec": float(start_sec),
            "end_sec": float(start_sec + num_seconds),
            "keep_frames": cur["keep_frames"],
            "redo": redo,
            "redo_list": redo_list,
            "redo_index": redo_index,
        }
        # End pin (redo only): the next clip was generated from this clip's old hand-off frames, so
        # the new version must end on the same frame -> pin old handoff[0] at frame -handoff.
        pin_end = False
        end_image = ref[:1]
        attempt = 0
        if redo:
            sdir_r = _session_dir(session)
            attempt = len([f for f in os.listdir(os.path.join(sdir_r, "redo")) if f.startswith(f"clip_{n + 1:03d}_")]) \
                if os.path.isdir(os.path.join(sdir_r, "redo")) else 0
            old_hand = os.path.join(sdir_r, f"handoff_{n:03d}.npy")
            # no pin when the next clip is regenerated too: it will chain from the new hand-off instead
            if n + 1 < total_chunks and not plan[n + 1]["scene_start"] and (n + 1) not in redo_list and os.path.isfile(old_hand):
                end_image = torch.from_numpy(np.load(old_hand)[:1].astype(np.float32) / 255.0)
                pin_end = True
        clip_seed = (int(seed) + n * 1000003 + attempt * 7919) % (2 ** 63)
        info = (f"{'REDO ' if redo else ''}clip {n + 1}/{total_chunks}  audio {start_sec:.2f}s -> {start_sec + num_seconds:.2f}s  "
                f"(hand-off {ref.shape[0]} frames, seed {clip_seed}{', end pinned' if pin_end else ''})  folder={CHAINS_SUBDIR}/{session}")
        logging.info(f"[LTX Chain] {info}")
        # Stage 1 decides the motion: a reference there on clips > 0 snaps the pose back to the
        # reference picture. Stage 2 only refines detail, so the reference is safe on every clip.
        # A scene-start clip has no start frame, so it needs the references at stage 1 as well.
        use_msr_stage1 = (msr_clips == "all") or n == 0 or scene_start
        use_msr_stage2 = (msr_clips in ("all", "stage2_all")) or n == 0 or scene_start
        return (ref, float(start_sec), num_seconds, n, total_chunks, chain, info, out_w, out_h,
                use_msr_stage1, use_msr_stage2, scene_index, scene_start, clip_seed, end_image, pin_end, bool(face_anchor))


class LTXChainStep:
    """Saves the finished clip, then re-queues the workflow for the next one (or builds the final video)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "chain": (CHAIN_TYPE,),
                "final_name": ("STRING", {"default": "final",
                                          "tooltip": "連結した完成動画のファイル名（拡張子なし）"}),
                "final_crf": ("INT", {"default": 16, "min": 0, "max": 51}),
                "auto_continue": ("BOOLEAN", {"default": True,
                                              "tooltip": "オフにすると次のクリップを自動投入しません（テスト用）"}),
            },
            "optional": {
                "chunk_audio": ("AUDIO", {"tooltip": "このクリップ分の音声（TrimAudioDuration の出力）。つなぐとクリップ mp4 に音声が付きます"}),
                "handoff_images": ("IMAGE", {"tooltip": "任意。次クリップの参照（hand-off）に使うフレーム。ReActor で顔を直した「最後の数フレーム」をつなぐと、保存するクリップは元のままで、次クリップだけ正しい顔から生成されます（顔ドリフトのリセット）。末尾 handoff 枚だけ使用"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "LTX Chain"

    def run(self, images, chain, final_name, final_crf, auto_continue, chunk_audio=None, handoff_images=None,
            prompt=None, extra_pnginfo=None, unique_id=None):
        n, total, fps = chain["index"], chain["total"], chain["fps"]
        handoff = chain.get("handoff", 1)
        session = chain["session"]
        sdir = _session_dir(session)
        subfolder = f"{CHAINS_SUBDIR}/{session}"

        # Colour drift / seam step: normalise every frame's global colour statistics to the
        # reference image (see _normalize_color). Frame 0 of the first clip IS the reference.
        images = images.float()
        strength = chain.get("color_match", 0.0)
        scene_index = chain.get("scene_index", 0)
        scene_ref_path = os.path.join(sdir, f"scene_ref_{scene_index:03d}.png")
        if chain.get("scene_start"):
            # new scene: this clip's own first frame becomes the colour reference for the scene
            Image.fromarray((images[0] * 255).clamp(0, 255).byte().cpu().numpy()).save(scene_ref_path, compress_level=1)
        if os.path.isfile(scene_ref_path):
            color_ref = torch.from_numpy(np.asarray(Image.open(scene_ref_path).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
        else:
            color_ref = chain["reference"]
        skip = 1 if (n == 0 or chain.get("scene_start")) else 0
        if strength > 0 and images.shape[0] > skip:
            images = torch.cat([images[:skip], _normalize_color(images[skip:], color_ref, strength)], dim=0)

        # hand-off frames for the next clip (lossless): the last 1 + overlap frames.
        # On a redo the old hand-off stays: the following clip was built on it (and the new clip
        # was pinned to end on it), so the seam remains continuous.
        k = min(handoff, images.shape[0])
        hand_src = images
        if handoff_images is not None and handoff_images.shape[0] > 0:
            # identity anchor: the next clip continues from face-corrected frames, the saved clip
            # stays untouched. Same colour normalisation as the clip so the seam colour matches.
            hs = handoff_images.float().cpu()
            if hs.shape[1:3] != images.shape[1:3]:
                hs = _fit_image(hs, images.shape[2], images.shape[1])
            if strength > 0:
                hs = _normalize_color(hs, color_ref, strength)
            hand_src = hs
            k = min(handoff, hs.shape[0])
            logging.info(f"[LTX Chain] clip {n + 1}: hand-off taken from corrected frames ({k} of {hs.shape[0]})")
        hand = (hand_src[-k:] * 255).clamp(0, 255).byte().cpu().numpy()
        hand_path = os.path.join(sdir, f"handoff_{n:03d}.npy")
        redo = bool(chain.get("redo"))
        next_redone = redo and (n + 1) in chain.get("redo_list", [])
        if not redo or next_redone or not os.path.isfile(hand_path):
            np.save(hand_path, hand)
            Image.fromarray(hand[-1]).save(os.path.join(sdir, f"last_frame_{n:03d}.png"), compress_level=1)

        # Seam: the first `handoff` frames of this clip re-create the tail of the previous clip.
        # Dissolve from the previous clip's real tail into this clip's version of it, so any small
        # difference in pose/detail is spread over the overlap instead of landing on one frame.
        frames = images
        prev_path = chain.get("prev_handoff")
        if n > 0 and prev_path and os.path.isfile(prev_path):
            prev = torch.from_numpy(np.load(prev_path).astype(np.float32) / 255.0)
            k = min(prev.shape[0], handoff, images.shape[0])
            if k > 1:
                w = torch.linspace(0.0, 1.0, k).view(-1, 1, 1, 1)
                head = prev[-k:] * (1.0 - w) + images[:k] * w
                frames = torch.cat([head, images[k:]], dim=0)
        # every clip but the last of its scene hands its tail to the next clip (which starts with
        # the dissolve); the last clip of a scene / of the run is cut exactly at the boundary
        keep = chain.get("keep_frames")
        if keep is not None:
            frames = frames[:max(1, min(int(keep), frames.shape[0]))]
        elif n + 1 < total and frames.shape[0] > handoff:
            frames = frames[:-handoff]
        clip_path = os.path.join(sdir, f"clip_{n + 1:03d}.mp4")
        if redo and os.path.isfile(clip_path):
            bdir = os.path.join(sdir, "redo")
            os.makedirs(bdir, exist_ok=True)
            k_old = len([f for f in os.listdir(bdir) if f.startswith(f"clip_{n + 1:03d}_")])
            os.replace(clip_path, os.path.join(bdir, f"clip_{n + 1:03d}_{k_old + 1:02d}.mp4"))
        if chunk_audio is not None and chunk_audio["waveform"].shape[-1] < chunk_audio["sample_rate"] // 10:
            chunk_audio = None  # < 0.1 s of audio: the encoder chokes on it, save the clip silent
        if frames.shape[0] == 0:
            logging.warning(f"[LTX Chain] clip {n + 1}/{total} produced no new frames; skipping it")
        else:
            InputImpl.VideoFromComponents(
                Types.VideoComponents(images=frames, audio=chunk_audio, frame_rate=Fraction(fps))
            ).save_to(clip_path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=CHUNK_CRF)
            logging.info(f"[LTX Chain] saved clip {n + 1}/{total}: {clip_path} ({frames.shape[0]} frames)")

        if redo:
            rl, ri = chain["redo_list"], chain["redo_index"]
            if ri + 1 < len(rl):
                if auto_continue:
                    self._requeue(prompt, extra_pnginfo, chain, hand_path)
                    info = f"REDO clip {n + 1} done -> queued clip {rl[ri + 1] + 1} ({ri + 2}/{len(rl)})"
                else:
                    info = f"REDO clip {n + 1} done (auto_continue off)"
                ui_files = [{"filename": os.path.basename(clip_path), "subfolder": subfolder, "type": "output"}]
                return {"ui": {"images": ui_files, "animated": (True,), "text": [info]}, "result": (info,)}
            final_path = self._concat(chain, sdir, final_name, final_crf)
            info = f"REDO done ({', '.join(str(i + 1) for i in rl)}) -> {subfolder}/{os.path.basename(final_path)}  (State の redo_clips を空に戻してください)"
            logging.info(f"[LTX Chain] {info}")
            ui_files = [{"filename": os.path.basename(final_path), "subfolder": subfolder, "type": "output"}]
            return {"ui": {"images": ui_files, "animated": (True,), "text": [info]}, "result": (info,)}

        if n + 1 < total:
            if auto_continue:
                self._requeue(prompt, extra_pnginfo, chain, hand_path)
                info = f"clip {n + 1}/{total} done -> queued clip {n + 2}"
            else:
                info = f"clip {n + 1}/{total} done (auto_continue off)"
            ui_files = [{"filename": os.path.basename(clip_path), "subfolder": subfolder, "type": "output"}]
            return {"ui": {"images": ui_files, "animated": (True,), "text": [info]}, "result": (info,)}

        final_path = self._concat(chain, sdir, final_name, final_crf)
        info = f"all {total} clips done -> {subfolder}/{os.path.basename(final_path)}"
        logging.info(f"[LTX Chain] {info}")
        ui_files = [{"filename": os.path.basename(final_path), "subfolder": subfolder, "type": "output"}]
        return {"ui": {"images": ui_files, "animated": (True,), "text": [info]}, "result": (info,)}

    # -- helpers -----------------------------------------------------------

    def _requeue(self, prompt, extra_pnginfo, chain, hand_path):
        if prompt is None:
            raise RuntimeError("hidden PROMPT missing; cannot re-queue")
        new_prompt = copy.deepcopy(prompt)
        state_ids = [k for k, v in new_prompt.items() if v.get("class_type") == STATE_NODE_CLASS]
        if not state_ids:
            raise RuntimeError("LTXChainState node not found in the prompt")
        for sid in state_ids:
            if chain.get("redo"):
                new_prompt[sid]["inputs"]["chain_iter"] = chain["redo_index"] + 1
                new_prompt[sid]["inputs"]["chain_state"] = json.dumps({"session": chain["session"], "redo": True})
            else:
                new_prompt[sid]["inputs"]["chain_iter"] = chain["index"] + 1
                new_prompt[sid]["inputs"]["chain_state"] = json.dumps(
                    {"session": chain["session"], "prev_frames": hand_path})

        server = PromptServer.instance
        body = {"prompt": new_prompt, "extra_data": {}}
        if extra_pnginfo is not None:
            body["extra_data"]["extra_pnginfo"] = extra_pnginfo
        if server.client_id:
            body["client_id"] = server.client_id
        host = server.address if server.address not in (None, "", "0.0.0.0", "::") else "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        url = f"http://{host}:{server.port}/prompt"
        data = json.dumps(body).encode("utf-8")

        def post():
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                logging.info(f"[LTX Chain] queued clip {chain['index'] + 2}/{chain['total']}: {resp.get('prompt_id')}")
            except Exception as e:  # noqa: BLE001
                logging.error(f"[LTX Chain] failed to queue next clip: {e}")

        # Post from a thread so this node returns and the current prompt finishes first.
        threading.Timer(0.5, post).start()

    def _concat(self, chain, sdir, final_name, crf):
        fps = chain["fps"]
        total = chain["total"]
        clip_paths = [os.path.join(sdir, f"clip_{i + 1:03d}.mp4") for i in range(total)]
        missing = [p for p in clip_paths if not os.path.isfile(p)]
        if missing:
            logging.warning(f"[LTX Chain] concatenating without missing clip files: {[os.path.basename(m) for m in missing]}")
            clip_paths = [p for p in clip_paths if os.path.isfile(p)]
        if not clip_paths:
            raise FileNotFoundError(f"no clip files found in {sdir}")

        safe = re.sub(r'[\\/:*?"<>|]+', "_", final_name.strip()) or "final"
        final_path = os.path.join(sdir, f"{safe}.mp4")
        k = 2
        while os.path.exists(final_path):
            final_path = os.path.join(sdir, f"{safe}_{k}.mp4")
            k += 1

        # count frames so the audio can be trimmed to exactly the video length
        n_frames = 0
        for p in clip_paths:
            with av.open(p) as c:
                n_frames += c.streams.video[0].frames or sum(1 for _ in c.decode(video=0))
        video_len = n_frames / fps

        audio = chain["audio"]
        sr = int(audio["sample_rate"])
        wav = audio["waveform"][0]
        s0 = int(round(chain["audio_start_sec"] * sr))
        s1 = min(wav.shape[-1], s0 + int(math.ceil(video_len * sr)))
        wav = wav[:, s0:s1].float().cpu().contiguous().numpy()
        layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(wav.shape[0], "stereo")

        with av.open(final_path, mode="w") as out:
            vs = None
            astream = out.add_stream("aac", rate=sr, layout=layout)
            idx = 0
            for p in clip_paths:
                with av.open(p) as c:
                    for fr in c.decode(video=0):
                        if vs is None:
                            vs = out.add_stream("libx264", rate=Fraction(fps))
                            vs.width, vs.height = fr.width, fr.height
                            vs.pix_fmt = "yuv420p"
                            vs.options = {"crf": str(crf), "preset": "medium"}
                        nf = fr.reformat(format="yuv420p")
                        nf.pts = idx
                        nf.time_base = Fraction(1, fps)
                        idx += 1
                        out.mux(vs.encode(nf))
            out.mux(vs.encode(None))

            if wav.shape[-1] > 0:
                af = av.AudioFrame.from_ndarray(wav, format="fltp", layout=layout)
                af.sample_rate = sr
                af.pts = 0
                out.mux(astream.encode(af))
            out.mux(astream.encode(None))

        logging.info(f"[LTX Chain] final video: {final_path} ({idx} frames, {idx / fps:.2f}s)")
        return final_path


class LTXChainPromptSwitch:
    """Picks the prompt for this clip: the intro prompt while the vocals have not started yet,
    the singing prompt from the clip in which the vocals begin."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": (CHAIN_TYPE,),
                "vocal_start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                                              "tooltip": "mp3 上で歌が始まる秒。それより前に終わるクリップは prompt_intro、それ以降は prompt_singing"}),
                "prompt_intro": ("STRING", {"multiline": True, "default": ""}),
                "prompt_singing": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "vocals": ("AUDIO", {"tooltip": "このクリップの歌声だけの音声（MelBandRoFormer の vocals）。つなぐと、クリップ内に歌がほとんど無いときは自動で prompt_intro を使います"}),
                "vocal_threshold_db": ("FLOAT", {"default": -40.0, "min": -90.0, "max": 0.0, "step": 1.0,
                                                 "tooltip": "歌声とみなす音量（dBFS）。これより大きい区間を「歌あり」と数えます"}),
                "min_vocal_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "クリップの中で「歌あり」の区間がこの割合未満なら prompt_intro（歌わない）にします"}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("prompt", "is_singing", "vocal_ratio")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    @staticmethod
    def _vocal_ratio(vocals, threshold_db, window_sec=0.1):
        """Fraction of `window_sec` windows whose RMS is above threshold_db (dBFS)."""
        wav = vocals["waveform"]
        if wav.ndim == 3:
            wav = wav[0]
        mono = wav.float().mean(dim=0)
        win = max(1, int(vocals["sample_rate"] * window_sec))
        n = mono.shape[0] // win
        if n == 0:
            return 0.0
        rms = mono[:n * win].view(n, win).pow(2).mean(dim=1).sqrt()
        db = 20.0 * torch.log10(rms.clamp_min(1e-8))
        return float((db > threshold_db).float().mean())

    def run(self, chain, vocal_start_sec, prompt_intro, prompt_singing, vocals=None,
            vocal_threshold_db=-40.0, min_vocal_ratio=0.2):
        if "start_sec" in chain:
            start, end = chain["start_sec"], chain["end_sec"]
        else:
            step = chain["chunk_seconds"] - chain.get("overlap", 0) / chain["fps"]
            start = chain["audio_start_sec"] + chain["index"] * step
            end = start + chain["num_seconds"]
        singing = end > vocal_start_sec + 1e-6
        ratio = -1.0
        if vocals is not None:
            ratio = self._vocal_ratio(vocals, vocal_threshold_db)
            singing = singing and ratio >= min_vocal_ratio
        prompt = prompt_singing if singing or not prompt_intro.strip() else prompt_intro
        msg = (f"clip {chain['index'] + 1}/{chain['total']}: {'singing' if singing else 'intro'} prompt "
               f"({start:.1f}-{end:.1f}s, vocals from {vocal_start_sec:.1f}s, vocal ratio {ratio:.2f})")
        logging.info("[LTX Chain] " + msg)
        try:
            with open(os.path.join(_session_dir(chain["session"]), "prompts.txt"), "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass
        return (prompt, singing, ratio)


class LTXChainScenePrompt:
    """Prompt for scene mode: the scene/camera block for this clip's scene (State.clips_per_scene)
    plus the singing / not-singing action block (vocal detection as in Prompt Switch)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": (CHAIN_TYPE,),
                "scenes": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "シーン（カメラアングル・場所）の説明を「---」だけの行で区切って並べる。State の clips_per_scene クリップごとに上から順に使い、最後まで行ったら先頭に戻る。1 つ目は最初の画像のシーン"}),
                "action_singing": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "歌っているクリップで、シーン説明の後ろに足す文章"}),
                "action_intro": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "歌っていないクリップで足す文章（空なら常に action_singing）"}),
                "vocal_start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                                              "tooltip": "mp3 上で歌が始まる秒。それより前に終わるクリップは action_intro"}),
            },
            "optional": {
                "vocals": ("AUDIO", {"tooltip": "このクリップの歌声だけの音声（MelBandRoFormer の vocals）。歌がほとんど無いクリップは action_intro"}),
                "vocal_threshold_db": ("FLOAT", {"default": -40.0, "min": -90.0, "max": 0.0, "step": 1.0}),
                "min_vocal_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT", "STRING")
    RETURN_NAMES = ("prompt", "is_singing", "scene_index", "info")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    @staticmethod
    def _split_scenes(text):
        blocks, cur = [], []
        for line in text.splitlines():
            if line.strip().startswith("---"):
                blocks.append("\n".join(cur).strip()); cur = []
            else:
                cur.append(line)
        blocks.append("\n".join(cur).strip())
        return [b for b in blocks if b]

    def run(self, chain, scenes, action_singing, action_intro, vocal_start_sec, vocals=None,
            vocal_threshold_db=-40.0, min_vocal_ratio=0.2):
        blocks = self._split_scenes(scenes)
        if not blocks:
            raise ValueError("Scene Prompt: 'scenes' is empty")
        idx = chain.get("scene_index", 0)
        scene_text = blocks[idx % len(blocks)]

        if "start_sec" in chain:
            start, end = chain["start_sec"], chain["end_sec"]
        else:
            step = chain["chunk_seconds"] - chain.get("overlap", 0) / chain["fps"]
            start = chain["audio_start_sec"] + chain["index"] * step
            end = start + chain["num_seconds"]
        singing = end > vocal_start_sec + 1e-6
        ratio = -1.0
        if vocals is not None:
            ratio = LTXChainPromptSwitch._vocal_ratio(vocals, vocal_threshold_db)
            singing = singing and ratio >= min_vocal_ratio
        action = action_singing if singing or not action_intro.strip() else action_intro
        prompt = scene_text + ("\n\n" + action.strip() if action.strip() else "")

        msg = (f"clip {chain['index'] + 1}/{chain['total']}: scene {idx % len(blocks) + 1}/{len(blocks)}"
               f"{' (new scene)' if chain.get('scene_start') else ''}, {'singing' if singing else 'intro'} "
               f"({start:.1f}-{end:.1f}s, vocal ratio {ratio:.2f})")
        logging.info("[LTX Chain] " + msg)
        try:
            with open(os.path.join(_session_dir(chain["session"]), "prompts.txt"), "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass
        return (prompt, singing, idx % len(blocks), msg)


class LTXChainSceneCuts:
    """Cut-line editor. The waveform UI lives in web/scene_cuts.js and writes the times into `cuts`;
    this node just normalises the text and passes it on to State.scene_cuts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cuts": ("STRING", {"default": "", "multiline": False,
                                    "tooltip": "シーンの切り替え位置（秒、カンマ区切り）。上の波形をクリックすると自動で入ります。手で編集も可"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "LoadAudio をつなぐと波形が表示されます（音声そのものは使いません）"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("scene_cuts", "scene_count")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    def run(self, cuts, audio=None):
        vals = _parse_cuts(cuts)
        return (", ".join(f"{v:g}" for v in vals), len(vals) + 1)


class LTXChainAutoScenes:
    """Auto-writes the whole prompt from the input image (Florence-2 caption) for a singing video.

    The image is analysed once and cached by its content hash, so swapping the image + mp3 and
    pressing Run is enough: the character/appearance/setting are read from the picture and combined
    with a fixed set of singing camera angles. `caption` is a lazy input (Florence-2 Run), so on
    cached clips the vision model is never loaded."""

    # Two sets of camera templates: with a studio microphone in frame, and without one. Any mention of
    # "microphone" in the prompt makes LTX add one, so the no-mic set never uses the word at all.
    CAMERAS_MIC = [
        "Locked medium shot from the waist up, camera level and still, 85mm lens, the microphone in the "
        "right third of the frame. Soft key light on the face, the background gently out of focus. Camera locked.",
        "Close-up of the singer's face and the microphone from a three-quarter angle, 100mm lens, shallow "
        "depth of field, the pop filter softly blurred in a corner, the eyes and expression in crisp focus. Camera locked.",
        "Medium-full shot, the camera at chest height angled very slightly upward so the singer looks tall and "
        "elegant, 50mm lens, framed from the head to mid-thigh, standing upright at the microphone. The framing "
        "stays exactly the same for the whole shot, camera locked, no zoom and no push-in.",
        "Profile shot from the side, chest up, the microphone between the singer and the camera, 85mm lens, "
        "soft backlight outlining the face and hair. Camera locked.",
        "Low angle from just below the microphone looking slightly up at the singer, 50mm lens, the microphone "
        "and pop filter large in the lower foreground. Camera locked.",
    ]
    CAMERAS_NOMIC = [
        "Locked medium shot from the waist up, camera level and still, 85mm lens, the singer slightly off-centre "
        "in the frame. Soft key light on the face, the background gently out of focus. Camera locked.",
        "Close-up of the singer's face from a three-quarter angle, 100mm lens, shallow depth of field, the "
        "background softly blurred, the eyes and expression in crisp focus. Camera locked.",
        "Medium-full shot, the camera at chest height angled very slightly upward so the singer looks tall and "
        "elegant, 50mm lens, framed from the head to mid-thigh, standing upright and facing the camera. The framing "
        "stays exactly the same for the whole shot, camera locked, no zoom and no push-in.",
        "Profile shot from the side, chest up, 85mm lens, soft backlight outlining the face and hair. Camera locked.",
        "Low angle from below looking slightly up at the singer, 50mm lens, the face large in the upper part of "
        "the frame. Camera locked.",
    ]
    CAMERAS = CAMERAS_MIC  # backwards-compatible alias
    # how wide each built-in angle is (index-aligned with CAMERAS_MIC / CAMERAS_NOMIC) and the
    # face height each framing leaves in the frame (fraction of the output height, measured on
    # this project's renders). The face is what breaks first when the shot gets wide: stage 1 runs
    # at half resolution and the VAE packs 32 px into one latent cell, so a face under ~120 px of
    # output height (~2 latent cells at stage 1) can no longer be held even with MSR references.
    ANGLE_FRAMING = ["waist-up", "close-up", "mid-thigh", "chest-up", "close-up"]
    FRAMING_RANK = {"close-up": 0, "chest-up": 1, "waist-up": 2, "mid-thigh": 3, "full body": 4, "wide": 5}
    FRAMING_FACE = {"close-up": 0.45, "chest-up": 0.30, "waist-up": 0.22, "mid-thigh": 0.15, "full body": 0.10, "wide": 0.06}
    FRAMING_LIMITS = ["close-up", "chest-up", "waist-up", "mid-thigh", "full body", "wide"]
    ACTION_SINGING = ("The singer performs the song into the microphone with precise lip sync to the vocals: the "
                      "mouth shapes match every word, natural jaw and lip movement, breaths between phrases, the mouth "
                      "closed during instrumental passages. Restrained emotion in the eyes and brows, the head moving "
                      "softly with the melody, the hands gesturing gently. Single continuous take, photorealistic, crisp "
                      "focus on the face, fine skin and fabric texture, steady exposure and white balance. "
                      "Audio: the vocal performance over the song with quiet room tone.")
    ACTION_INTRO = ("The singer is not singing yet: the lips stay closed, listening to the music, swaying gently to the "
                    "rhythm, the head nodding softly on the beat, breathing calmly, glancing at the microphone while "
                    "waiting for the cue. Single continuous take, photorealistic, crisp focus on the face, steady "
                    "exposure and white balance. Audio: the instrumental of the song with quiet room tone.")
    ACTION_INTRO_NOMIC = ("The singer is not singing yet: the lips stay closed, listening to the music, swaying gently to "
                          "the rhythm, the head nodding softly on the beat, breathing calmly, glancing at the camera while "
                          "waiting for the cue. Single continuous take, photorealistic, crisp focus on the face, steady "
                          "exposure and white balance. Audio: the instrumental of the song with quiet room tone.")
    MIC_WORDS = ("microphone", " mic ", " mic,", " mic.", " mics ", "mic stand")

    # gender: detected from the caption, or forced (then the caption's own words are rewritten too,
    # since Florence sometimes reads a short-haired woman as "a man" and the prompt would fight itself)
    FEMALE_WORDS = ("woman", "women", "girl", "girls", "female", "lady", "ladies", "she", "her", "hers",
                    "actress", "schoolgirl", "businesswoman")
    MALE_WORDS = ("man", "men", "boy", "boys", "male", "gentleman", "gentlemen", "he", "his", "him",
                  "guy", "guys", "schoolboy", "businessman")
    TO_FEMALE = {"man": "woman", "men": "women", "boy": "girl", "boys": "girls", "male": "female",
                 "gentleman": "lady", "gentlemen": "ladies", "he": "she", "his": "her", "him": "her",
                 "guy": "woman", "guys": "women", "schoolboy": "schoolgirl", "businessman": "businesswoman"}
    TO_MALE = {"woman": "man", "women": "men", "girl": "boy", "girls": "boys", "female": "male",
               "lady": "gentleman", "ladies": "gentlemen", "she": "he", "her": "his", "hers": "his",
               "actress": "actor", "schoolgirl": "schoolboy", "businesswoman": "businessman"}

    # emotion / facial expression of the performance: (while singing, while listening / not singing)
    EMOTIONS = {
        "none": ("", ""),
        "happy": ("a happy, warm mood: bright eyes, a light smile between phrases, cheerful and relaxed",
                  "a happy, warm expression with a light smile"),
        "joyful": ("a joyful, radiant mood: eyes lit up, an open smile between phrases, lively and uplifted",
                   "a joyful, radiant expression, smiling"),
        "tender": ("a tender, loving mood: soft eyes, a gentle warm expression, intimate and caring",
                   "a tender, gentle expression with soft eyes"),
        "sad": ("a sad, sorrowful mood: heavy eyes, the corners of the mouth turned down, brows drawn "
                "together, holding back tears, never smiling",
                "a sad, sorrowful expression with heavy eyes, the mouth turned down, not smiling"),
        "melancholic": ("a melancholic, wistful mood: distant eyes, a faint bittersweet expression, quiet longing, "
                        "the lips relaxed and unsmiling",
                        "a melancholic, wistful expression with distant eyes, not smiling"),
        "nostalgic": ("a nostalgic, reflective mood: a soft faraway look, eyes slightly glazed with memory, a quiet "
                      "bittersweet expression, the lips relaxed and unsmiling",
                      "a nostalgic, reflective expression with a faraway look, the lips closed and unsmiling"),
        "passionate": ("a passionate, intense mood: burning eyes, brows knitting on the strong notes, deeply "
                       "committed to every word, serious and not smiling",
                       "a passionate, intense expression with burning eyes, serious"),
        "serene": ("a serene, peaceful mood: calm relaxed eyes, a tranquil neutral expression, unhurried, "
                   "the lips relaxed",
                   "a serene, peaceful expression with calm eyes and relaxed neutral lips"),
        "playful": ("a playful, flirty mood: sparkling eyes, a mischievous smile, small teasing glances at the camera",
                    "a playful expression with sparkling eyes and a mischievous smile"),
        "confident": ("a confident, powerful mood: a direct steady gaze, the chin slightly raised, self-assured",
                      "a confident expression with a direct steady gaze"),
        "serious": ("a serious, earnest mood: a composed, focused face, steady sincere eyes, the brows calm, the lips "
                    "relaxed and unsmiling, dignified and fully concentrated on the song",
                    "a serious, composed expression with steady focused eyes, not smiling"),
        "dreamy": ("a dreamy, floating mood: half-lidded eyes, a soft faraway expression, gently lost in the music, "
                   "the lips relaxed and unsmiling",
                   "a dreamy expression with half-lidded eyes, lost in the music, not smiling"),
        "angry": ("an angry, defiant mood: hard eyes, a tightened jaw, furrowed brows, fierce and intense, "
                  "never smiling",
                  "an angry, defiant expression with hard eyes and furrowed brows, not smiling"),
    }
    # moods where a smile in the image caption must not leak into every shot
    NO_SMILE = ("sad", "melancholic", "nostalgic", "passionate", "serene", "dreamy", "angry", "serious")
    NO_SMILE_WORDS = ("sad", "sorrow", "melanchol", "nostalg", "wistful", "angry", "anger", "serious", "grief",
                      "tear", "cry", "lonely", "somber", "sombre", "gloomy", "solemn", "not smiling", "no smile",
                      "unsmiling")

    # look (colour grade / film look), lighting and setting presets. They are composed with the free
    # `style` text into the tail that every scene prompt ends with; `setting` also rewrites the
    # "setting stays the same as the reference" sentence, since it deliberately replaces the photo's.
    LOOKS = {
        "custom (style text only)": "",
        "natural color film": "photorealistic, cinematic, natural film-like color, realistic skin and fabric texture, "
                              "subtle film grain",
        "warm cinematic": "photorealistic, cinematic, warm amber color grade, golden highlights, soft contrast, "
                          "realistic skin and fabric texture, subtle film grain",
        "cool cinematic": "photorealistic, cinematic, cool teal-and-blue color grade, clean highlights, realistic "
                          "skin and fabric texture, subtle film grain",
        "teal and orange": "photorealistic, cinematic teal-and-orange color grade, warm skin tones against cool "
                           "shadows, realistic skin and fabric texture, subtle film grain",
        "black and white": "photorealistic, cinematic black-and-white, rich monochrome tones with deep blacks and soft "
                           "highlights, fine film grain, realistic skin and fabric texture",
        "vintage 70s film": "photorealistic, vintage 1970s film look, faded warm colors, soft halation, gentle grain, "
                            "slightly lifted blacks, realistic skin and fabric texture",
        "90s music video": "photorealistic, 1990s music video look, punchy saturated color, slight softness, film "
                           "grain, realistic skin and fabric texture",
        "high-key clean": "photorealistic, clean bright high-key look, soft even light, low contrast, pastel-leaning "
                          "color, realistic skin and fabric texture",
        "moody low-key": "photorealistic, moody low-key cinematic look, deep shadows, restrained color, high "
                         "contrast, fine film grain, realistic skin and fabric texture",
        "neon night": "photorealistic, cinematic neon-lit night look, magenta and cyan highlights, glossy "
                      "reflections, deep shadows, realistic skin and fabric texture",
        "glossy commercial": "photorealistic, glossy high-end commercial look, crisp detail, polished color, clean "
                             "highlights, realistic skin and fabric texture",
        "documentary": "photorealistic, natural documentary look, unpolished true-to-life color, available light, "
                       "realistic skin and fabric texture",
    }
    LIGHTINGS = {
        "auto (from photo)": "",
        "soft key light": "soft key light on the face, gentle fill, the background a little darker than the face",
        "warm tungsten studio": "warm amber tungsten studio lighting, soft key light on the face, warm practical "
                                "lights glowing in the background",
        "studio softbox": "large softbox studio lighting, even soft light on the face, clean gentle shadows",
        "golden hour": "golden hour sunlight, a warm low sun from the side, long soft shadows",
        "overcast daylight": "soft overcast daylight, diffuse even light, no hard shadows",
        "window light": "soft natural window light from one side, gentle falloff into shadow",
        "dramatic side light": "dramatic single-source side light, half of the face in shadow, strong contrast",
        "backlight rim": "backlight with a rim of light outlining the hair and shoulders, soft fill on the face",
        "stage spotlight": "a stage spotlight from above, the singer bright against a dark background, haze in "
                           "the beam",
        "neon": "neon lights from the sides, magenta and cyan colored light on the face and hair",
        "candles / practicals": "warm practical lamps and candles, a soft flickering glow on the face",
    }
    SETTINGS = {
        "auto (from photo)": "",
        "recording studio": "a professional recording studio with a mixing console, studio monitor speakers and "
                            "acoustic panels softly out of focus in the background",
        "plain studio backdrop": "a plain seamless studio backdrop, nothing else in the background",
        "white cyclorama": "a bright white cyclorama studio, seamless white all around",
        "concert stage": "a concert stage at night with stage lights, haze and a dark audience area behind",
        "small club stage": "a small live-house club stage with a brick wall, amps and colored stage lights",
        "empty theater": "an empty old theater with red velvet seats out of focus behind",
        "rooftop city": "a city rooftop with the skyline and city lights in the background",
        "street at night": "a city street at night with shop lights, wet asphalt reflections and passing lights",
        "neon alley": "a narrow alley lit by neon signs, wet ground reflections",
        "bedroom": "a cozy bedroom with a window, warm lamp light and posters on the wall",
        "loft apartment": "a bright loft apartment with large windows and exposed brick",
        "rainy window": "indoors by a large rain-streaked window, blurred city lights outside",
        "classic bar": "a classic dim bar with a wooden counter and bottles glowing behind",
        "abandoned warehouse": "a large abandoned warehouse with dusty beams of light through high windows",
        "forest": "a forest clearing with soft light filtering through the trees",
        "beach sunset": "a beach at sunset with the sea and the sky behind",
    }

    # weather / atmosphere: written as a continuous, visible effect so it does not fade out mid-clip.
    # `weather_when` decides which camera blocks get it (scenes cycle through the blocks).
    WEATHERS = {
        "none": "",
        "light rain": "light rain falling steadily through the whole shot, fine drops visible in the air and "
                      "glistening on surfaces, a damp sheen on the hair and clothes",
        "heavy rain": "heavy rain pouring down through the whole shot, thick streaks of rain in the air, water "
                      "running off surfaces, wet hair and soaked clothes",
        "light snow": "light snow falling gently and continuously through the whole shot, soft flakes drifting "
                      "in the air and settling on the hair and shoulders",
        "heavy snow": "heavy snowfall through the whole shot, dense flakes filling the air, snow settling on the "
                      "hair, shoulders and the ground",
        "fog": "thick soft fog filling the scene, the background fading into mist, diffuse light",
        "mist": "a thin veil of mist in the air, soft haze softening the background",
        "wind": "a strong wind through the whole shot, the hair and clothes blowing and fluttering continuously",
        "storm": "a storm through the whole shot: heavy rain, gusts of wind whipping the hair and clothes, "
                 "occasional distant lightning flashes",
        "falling petals": "cherry blossom petals drifting and falling continuously through the air",
        "falling leaves": "autumn leaves drifting and falling continuously through the air",
        "dust in sunbeams": "fine dust motes floating in visible beams of light",
        "sparks / embers": "glowing embers and sparks drifting up through the air",
        "confetti": "confetti falling continuously from above, fluttering through the air",
        "haze with light rays": "atmospheric haze with visible rays of light cutting through the air",
    }
    WEATHER_WHEN = {
        "all scenes": lambda i, n: True,
        "every other scene": lambda i, n: i % 2 == 0,
        "every third scene": lambda i, n: i % 3 == 0,
        "first scene only": lambda i, n: i == 0,
        "last scene only": lambda i, n: i == n - 1,
        "all but the first scene": lambda i, n: i > 0,
    }

    # VFX: one-off spectacle effects, placed on some of the camera blocks like the weather.
    # NOTE: flash-type effects (lightning, strobe, camera flashes) fight Step's per-frame colour
    # normalisation - the tooltip tells the user to lower handoff_color_match for those.
    VFX = {
        "none": "",
        "money rain (banknotes)": "banknotes raining down from above continuously, bills tumbling and fluttering "
                                  "through the air all around the singer and piling on the ground",
        "lightning": "sudden bright lightning flashes lighting up the whole scene at intervals, the background "
                     "flaring white for an instant, then dark again",
        "fireworks": "colorful fireworks bursting in the sky behind the singer, sparks raining down",
        "pyro sparks": "pyrotechnic spark fountains shooting up behind the singer, bright golden sparks",
        "stage fog": "low stage fog rolling across the floor and around the singer's feet",
        "smoke": "thick colored smoke drifting through the scene behind the singer",
        "bubbles": "soap bubbles floating through the air all around the singer",
        "gold glitter": "glittering gold dust sparkling and drifting through the air",
        "laser beams": "colored laser beams sweeping through haze behind the singer",
        "strobe lights": "strobe lights flashing rapidly in the background",
        "camera flashes": "paparazzi camera flashes popping from the darkness around the singer",
        "feathers": "white feathers floating slowly down through the air",
        "paper sheets flying": "sheets of paper swirling and flying through the air in the wind",
        "balloons": "colorful balloons drifting up past the singer",
        "flames": "flames flickering along the ground and the edges of the frame behind the singer",
        "lens flare": "anamorphic lens flares streaking horizontally across the frame from the lights",
        "light leaks": "warm film light leaks washing over the edges of the frame",
        "floating lanterns": "glowing paper lanterns floating up into the night sky behind the singer",
        "rain of rose petals": "red rose petals raining down from above all around the singer",
        "shattering glass": "shards of glass flying through the air in slow motion around the singer",
    }
    # each entry -> the set of block indices that get the effect (n blocks, seeded rng)
    VFX_WHEN = {
        "all scenes": lambda n, r: set(range(n)),
        "every other scene": lambda n, r: {i for i in range(n) if i % 2 == 0},
        "every third scene": lambda n, r: {i for i in range(n) if i % 3 == 0},
        "first scene only": lambda n, r: {0},
        "last scene only": lambda n, r: {n - 1},
        "all but the first scene": lambda n, r: set(range(1, n)),
        "random (about half)": lambda n, r: set(r.sample(range(n), max(1, round(n * 0.5)))),
        "random (about a third)": lambda n, r: set(r.sample(range(n), max(1, round(n / 3)))),
        "random (one scene)": lambda n, r: {r.randrange(n)},
    }

    # height / body build: the adjective that goes into the "(a tall, slim 25-year-old Japanese woman)" tag,
    # plus a short body sentence so full-body angles keep the proportions
    # (tag adjective, body description). "tall" alone means nothing in a lone shot with no reference
    # object, so the description spells out the PROPORTIONS (leg length, head-to-body ratio) - that is
    # what keeps a full-body walking shot from collapsing into a big-head / short-leg figure.
    HEIGHTS = {
        "none": ("", ""),
        "petite": ("petite", "a petite, small, delicate frame with a short stature"),
        "short": ("short", "a short stature with a compact frame"),
        "average height": ("average-height", "an average adult height with normal adult proportions"),
        "tall": ("tall", "a tall, long-legged adult frame with a long torso, the head small in proportion to the "
                         "body, adult proportions about eight heads tall"),
        "very tall": ("very tall", "a very tall, towering, long-legged adult frame with a long torso and long arms, "
                                   "the head small in proportion to the body, about eight and a half heads tall, "
                                   "never child-like, stubby or shrunken proportions"),
    }
    BUILDS = {
        "none": ("", ""),
        "very slim": ("very slim", "a very slim, narrow, bony frame"),
        "slim": ("slim", "a slim, lean frame"),
        "athletic": ("athletic, toned", "an athletic, toned frame with defined shoulders"),
        "average": ("average-build", "an average build"),
        "curvy": ("curvy", "a curvy, full-figured frame"),
        "chubby": ("chubby", "a soft, chubby, rounded frame"),
        "plump": ("plump", "a plump, heavy-set frame with a round belly"),
        "heavy": ("heavy-set", "a heavy, large-bodied frame with a big belly and thick limbs"),
    }
    # caption words that would contradict a forced height / build (removed from the caption)
    HEIGHT_WORDS = ("tall", "short", "petite", "towering", "tiny", "diminutive", "statuesque")
    BUILD_WORDS = ("slim", "slender", "thin", "skinny", "lean", "athletic", "toned", "muscular", "fit", "curvy",
                   "voluptuous", "chubby", "plump", "overweight", "fat", "heavy", "heavyset", "heavy-set", "stocky",
                   "stout", "portly", "obese", "full-figured", "plus-size", "petite")

    # camera movement applied to every angle (replaces "Camera locked." in the built-in templates).
    # NOTE: push-in / pull-out / crane accumulate across the clips of one scene (each clip continues
    # from the previous frames), so they are written as very slow and subtle.
    CAMERA_MOVES = {
        "auto": None,   # locked, or the follow-camera a moving `motion` asks for
        "locked": "Camera locked, no movement, no zoom.",
        "slow push-in": "The camera pushes in very slowly and smoothly toward the singer, a subtle gentle dolly-in, "
                        "no cut.",
        "slow pull-out": "The camera pulls back very slowly and smoothly, a subtle gentle dolly-out, the singer "
                         "staying centred, no cut.",
        "dolly left": "The camera glides slowly and smoothly sideways to the left on a dolly, the singer staying "
                      "centred and the same size, gentle parallax in the background.",
        "dolly right": "The camera glides slowly and smoothly sideways to the right on a dolly, the singer staying "
                       "centred and the same size, gentle parallax in the background.",
        "orbit": "The camera orbits slowly around the singer in a smooth continuous arc, the singer staying centred "
                 "and the same size in the frame.",
        "crane up": "The camera rises slowly and smoothly on a crane, tilting down a little to keep the singer "
                    "framed, no zoom.",
        "crane down": "The camera descends slowly and smoothly on a crane, tilting up a little to keep the singer "
                      "framed, no zoom.",
        "handheld": "Handheld camera with subtle natural sway and small organic micro-movements, documentary feel, "
                    "the singer staying in frame.",
        "follow": "The camera follows the singer smoothly, keeping the singer the same size and position in the "
                  "frame.",
        "mix (varies per scene)": None,  # a different move for each angle, see CAMERA_MIX
    }
    # "mix": no push-in / pull-out - they keep going clip after clip inside one scene, and a figure that
    # shrinks in frame is redrawn with child-like proportions. Both stay available as explicit choices.
    CAMERA_MIX = ["locked", "dolly left", "handheld", "orbit", "locked", "dolly right", "crane up", "handheld",
                  "locked", "crane down"]
    # a walking / strolling singer: only cameras that travel with the singer (orbit / crane contradict it)
    CAMERA_MIX_MOVING = ["follow", "handheld", "dolly left", "follow", "dolly right", "handheld"]
    MOVING_MOTIONS = ("walking toward camera", "running toward camera", "jogging toward camera", "walking sideways",
                      "strolling", "dancing")

    # how the singer moves: (while singing, while listening, camera note or None)
    # the camera note replaces "Camera locked." in the built-in angles so the framing doesn't fight the motion
    MOTIONS = {
        "none": ("", "", None),
        "standing still": (
            "The singer stays standing in exactly the same spot for the whole shot, feet planted, no walking and "
            "no stepping, the body only breathing and swaying very slightly",
            "standing in the same spot, feet planted", None),
        "gentle sway": (
            "The singer stays in the same spot, swaying gently from side to side with the rhythm, shifting weight "
            "from foot to foot, no walking, already swaying at the very first frame",
            "swaying gently in place with the rhythm, already swaying at the first frame", None),
        "hand gestures": (
            "The singer stays in the same spot and expresses the song with the hands: flowing, expressive hand and "
            "arm gestures that follow the lyrics, no walking",
            "standing in place, the hands moving gently with the music", None),
        "light dance in place": (
            "The singer dances lightly in place to the song while singing: small dance steps, hip sway and arm "
            "movements in time with the beat, staying in the same spot, the face natural and the lips in sync, "
            "already dancing at the very first frame, no standing start",
            "moving lightly to the beat in place, small dance steps, already moving at the first frame", None),
        "dancing": (
            "The singer dances to the song while singing: the whole body moves with the rhythm, steps, turns and "
            "arm movements in time with the beat, energetic but graceful, the face natural and the lips in sync, "
            "already mid-dance at the very first frame, no pause and no standing start",
            "dancing to the beat, the whole body moving with the rhythm, already dancing at the first frame",
            "Camera locked, the dancing stays within the frame."),
        "walking toward camera": (
            "The singer walks slowly and steadily toward the camera while singing, natural relaxed steps, the arms "
            "swinging lightly, already mid-stride at the very first frame, no pause and no standing start",
            "walking slowly toward the camera with natural relaxed steps, already mid-stride at the first frame",
            "The camera glides smoothly backward at the same pace as the singer, so the singer stays the same size "
            "and position in the frame, no zoom."),
        "running toward camera": (
            "The singer runs straight toward the camera at a steady pace while singing, arms pumping, the clothes "
            "and hair moving with the run, already mid-stride at the very first frame, no pause and no standing start",
            "running steadily toward the camera, already mid-stride at the first frame",
            "The camera races backward at exactly the same speed as the runner, so the singer stays the same size "
            "and centred in the frame, a slight handheld shake, the street rushing past on both sides, no zoom."),
        "jogging toward camera": (
            "The singer jogs lightly toward the camera while singing, relaxed steady strides, the arms swinging, "
            "already mid-stride at the very first frame, no standing start",
            "jogging lightly toward the camera, already mid-stride at the first frame",
            "The camera glides backward at the same pace as the jogger, keeping the singer the same size and "
            "centred in the frame, a subtle handheld feel, no zoom."),
        "walking sideways": (
            "The singer walks slowly along the scene from one side to the other while singing, natural relaxed steps, "
            "already mid-stride at the very first frame, no pause and no standing start",
            "walking slowly across the scene with natural relaxed steps, already mid-stride at the first frame",
            "The camera tracks sideways smoothly at the same pace, keeping the singer centred and the same size in "
            "the frame."),
        "strolling": (
            "The singer strolls slowly through the scene while singing, natural relaxed steps, looking around "
            "occasionally and back to the camera, already mid-stride at the very first frame, no standing start",
            "strolling slowly through the scene with natural relaxed steps, already mid-stride at the first frame",
            "The camera follows smoothly, keeping the singer the same size in the frame."),
        "sitting": (
            "The singer stays seated in the same position for the whole shot, the upper body relaxed, the hands "
            "resting, no standing up",
            "seated in the same position, relaxed", None),
    }

    # How strongly the singer performs. LTX tends to over-open the mouth, so the default is toned down.
    PERFORMANCE = {
        "subtle": "a calm, subtle, intimate delivery: the mouth opens only a little and naturally, small relaxed lip "
                  "movements, a soft gentle expression, barely any head or body movement, understated and never "
                  "exaggerated, no wide-open mouth and no grimacing",
        "restrained": "a restrained, controlled delivery: natural moderate mouth movement, quiet emotion in the eyes and "
                      "brows, the head moving only softly with the melody, small gentle gestures, not theatrical, "
                      "avoiding a wide-open mouth or exaggerated faces",
        "natural": "a natural, believable delivery: normal mouth movement matching the words, genuine but not overacted "
                   "emotion, the head moving softly with the melody, gentle hand gestures",
        "expressive": "an emotional, expressive delivery with strong feeling, fuller mouth movement and more dynamic "
                      "gestures, while keeping the face natural",
    }

    def _sing(self, ns, clause, mic=True, emo="", mot=""):
        common = ("Single continuous take, photorealistic, crisp focus on the face"
                  f"{'s' if ns > 1 else ''}, fine skin and fabric texture, steady exposure and white balance. "
                  "Audio: the vocal performance over the song with quiet room tone.")
        into = "into the microphone" if mic else "directly to the camera"
        emo = f" The performance carries {emo}." if emo else ""
        if mot and ns > 1:
            mot = mot.replace("The singer stays", "The singers stay").replace("The singer walks", "The singers walk") \
                     .replace("The singer strolls", "The singers stroll").replace("The singer dances", "The singers dance") \
                     .replace("The singer runs", "The singers run").replace("The singer jogs", "The singers jog")
        emo = emo + (f" {mot}." if mot else "")
        if ns == 1:
            return (f"The singer performs the song {into} with precise lip sync to the vocals: the mouth "
                    "shapes match every word, breaths between phrases, the mouth closed during instrumental passages. "
                    f"{clause}.{emo} " + common)
        who = "The two singers" if ns == 2 else f"The {self._num_word(ns)} singers"
        together = ("trade lines and harmonise" if ns == 2 else "sing together as an ensemble")
        return (f"{who} perform the song together and {together}: each one lip-syncs their own part with precise timing "
                "to the vocals, mouths matching every word, breaths between phrases, mouths closed during instrumental "
                f"passages. {clause}.{emo} " + common)

    def _intro(self, ns, mic=True, emo="", mot=""):
        at = "at the microphone" if mic else "at the camera"
        mot = f" Movement: {mot}." if mot else ""
        if ns == 1:
            emo = (f" The face keeps {emo}." if emo else "") + mot
            return ("The singer is not singing yet: the lips stay closed, listening to the music, swaying gently to the "
                    f"rhythm, the head nodding softly on the beat, breathing calmly, glancing {at} while "
                    f"waiting for the cue.{emo} Single continuous take, photorealistic, crisp focus on the face, steady "
                    "exposure and white balance. Audio: the instrumental of the song with quiet room tone.")
        who = "The two singers" if ns == 2 else f"The {self._num_word(ns)} singers"
        emo = (f" Their faces keep {emo}." if emo else "") + mot
        return (f"{who} are not singing yet: lips closed, listening to the music, swaying gently to the rhythm, heads "
                f"nodding softly on the beat, glancing at each other and {at} while waiting for the cue.{emo} "
                "Single continuous take, photorealistic, steady exposure. Audio: the instrumental of the song.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "解析する開始画像（LoadImage）。この見た目・服・場所を全シーンに反映します"}),
                "caption": ("STRING", {"forceInput": True, "lazy": True,
                                       "tooltip": "Florence-2 Run（task = more_detailed_caption）の caption 出力。キャッシュがある画像では評価されません"}),
                "num_scenes": ("INT", {"default": 4, "min": 1, "max": 5,
                                       "tooltip": "作るシーン（カメラアングル）の種類数。State の clips_per_scene / scene_cuts と組み合わせて切り替わります"}),
                "num_singers": ("INT", {"default": 1, "min": 1, "max": 8,
                                        "tooltip": "歌っている人数。1=ソロ / 2=デュエット（掛け合い＋ハモリ）/ 3人以上=グループで一緒に。人物説明と歌い方を人数に合わせて書き分けます。全員が1枚の画像に写っている想定（別々の顔を固定したいときは MSR の REF スロットを使う）"}),
                "ethnicity": ("STRING", {"default": "",
                                         "tooltip": "国籍・人種（例: Japanese / Korean / Japanese and French mixed）。指定すると人物説明に明記します。空なら画像の見た目のまま（Florence の説明を尊重）"}),
                "age": ("STRING", {"default": "",
                                   "tooltip": "年齢（例: 25 / 20代前半 は early 20s / around 30）。指定すると人物説明に明記し「毎フレーム同じ年齢・老けない」を追加。長尺で顔が老けていくのを防ぎます。空なら画像のまま。数字だけなら自動で '25-year-old' に整形"}),
                "performance": (["subtle", "restrained", "natural", "expressive"], {"default": "restrained",
                                "tooltip": "歌い方の強さ。subtle=控えめ（口を大きく開けない）/ restrained=抑えめ（既定・大げさ防止）/ natural=自然 / expressive=感情的。LTX は口を開けすぎる癖があるので既定は抑えめ"}),
                "style": ("STRING", {"multiline": True, "default": "photorealistic, cinematic, natural film-like color",
                                     "tooltip": "全シーン共通で足したい雰囲気（自由記述）。下の look / lighting / setting プリセットの後ろに足されます。プリセットを使うならここは空か短い補足で OK。※全シーンに付くので、カメラアングルはここに書かない"}),
                "extra_cameras": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "任意。自分で足したいカメラアングルを1行に1つ（例: profile shot from the side, 85mm, the microphone between her and the camera）。ここに書いた分がシーンのローテーションに追加され、時々そのアングルになります。空なら既定の雛形だけ"}),
                "microphone": (["auto", "yes", "no"], {"default": "auto",
                               "tooltip": "マイクをプロンプトに入れるか。auto=画像解析（Florence）の説明にマイクがあれば入れる、無ければ入れない / yes=常にスタジオマイクの前で歌う / no=マイク無し（カメラに向かって歌う）。以前は常にマイク入りだったので、画像に無いマイクが勝手に足される場合は no または auto に"}),
                "gender": (["auto", "female", "male"], {"default": "auto",
                           "tooltip": "歌い手の性別。auto=画像解析（Florence）の説明から判定（woman/girl/she → female、man/boy/he → male）。female / male を指定すると人物説明に明記し、Florence の説明文にある性別の語も書き換えます（短髪の女性が man と読まれた時などの修正用）"}),
                "emotion": (["none", "happy", "joyful", "tender", "sad", "melancholic", "nostalgic", "passionate",
                             "serene", "playful", "confident", "serious", "dreamy", "angry"], {"default": "none",
                            "tooltip": "歌っている時の感情・表情。none=指定なし（performance の強さだけ）/ happy=幸せ / joyful=喜び / tender=優しい / sad=悲しい / melancholic=憂い / nostalgic=懐かしむ / passionate=情熱的 / serene=穏やか / playful=茶目っ気 / confident=自信 / serious=真面目・真剣 / dreamy=夢見心地 / angry=怒り。歌唱中と歌っていない区間（間奏）の表情の両方に反映。sad〜angry の非笑顔系では写真説明の smile 語も自動除去。下の emotion_custom に書くとそちらが優先"}),
                "emotion_custom": ("STRING", {"default": "",
                                   "tooltip": "任意。感情・表情を自由記述（英語推奨。例: a shy, bashful mood with a small nervous smile）。空なら上の emotion を使用"}),
                "motion": (["none", "standing still", "gentle sway", "hand gestures", "light dance in place", "dancing",
                            "walking toward camera", "jogging toward camera", "running toward camera", "walking sideways",
                            "strolling", "sitting"], {"default": "none",
                           "tooltip": "人物の動き。none=指定なし / standing still=その場に立ったまま / gentle sway=その場で軽く揺れる / hand gestures=その場で手振り / light dance in place=その場で軽く踊る / dancing=踊る / walking toward camera=カメラに向かって歩く（カメラは後退追従） / jogging・running toward camera=カメラに向かって走る（カメラは同速で後退、手持ち感） / walking sideways=横に歩く（カメラ横追従） / strolling=散歩（カメラ追従） / sitting=座ったまま。歩く・踊る時は microphone=no 推奨。下の motion_custom に書くとそちらが優先"}),
                "motion_custom": ("STRING", {"default": "",
                                  "tooltip": "任意。動きを自由記述（英語推奨。例: The singer slowly turns around and walks away from the camera）。空なら上の motion を使用"}),
                "height": (["none", "petite", "short", "average height", "tall", "very tall"], {"default": "none",
                           "tooltip": "背の高さ。none=指定なし（写真のまま）/ petite=小柄 / short=低め / average height=平均 / tall=高い / very tall=とても高い。指定すると人物説明に明記し、写真説明にある背の語（tall/short 等）は除去。膝上・全身の構図で効きます"}),
                "build": (["none", "very slim", "slim", "athletic", "average", "curvy", "chubby", "plump", "heavy"], {"default": "none",
                          "tooltip": "体型。none=指定なし（写真のまま）/ very slim=とても痩せている / slim=痩せている / athletic=引き締まった / average=普通 / curvy=グラマー / chubby=ぽっちゃり / plump=太め / heavy=太っている。指定すると人物説明に明記し、写真説明にある体型の語（slim/plump 等）は除去。全シーンで体型を固定する文も追加"}),
                "camera": (["auto", "locked", "slow push-in", "slow pull-out", "dolly left", "dolly right", "orbit",
                            "crane up", "crane down", "handheld", "follow", "mix (varies per scene)"], {"default": "auto",
                           "tooltip": "カメラワーク（全アングル共通）。auto=固定（歩く系 motion のときは追従）/ locked=固定 / slow push-in=ゆっくり寄る / slow pull-out=ゆっくり引く / dolly left・right=横にスライド / orbit=人物の周りを回る / crane up・down=上昇・下降 / handheld=手持ちの揺れ / follow=人物を追う / mix=シーンごとに違う動きを順番に割り当て（固定 / 横ドリー / 手持ち / オービット / クレーン。歩く系 motion のときは追従 / 手持ち / 横ドリーのみ）。※push-in / pull-out は同じシーン内でクリップをまたいで蓄積し、人物が小さくなると体型も崩れるので、明示的に選んだときだけ・clips_per_scene=1 で。下の camera_custom に書くとそちらが優先"}),
                "camera_custom": ("STRING", {"default": "",
                                  "tooltip": "任意。カメラワークを自由記述（英語推奨。例: slow lateral dolly from left to right, 50mm, the singer stays centred）。全アングルの「Camera locked.」と置き換わります。空なら上の camera を使用"}),
                "outfit": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "任意。服装の補足（英語推奨）。参照画像に写っていない部分を明示するのに使う。例: full-length dark blue denim jeans down to the ankles, white sneakers。ここに書いた文が人物説明に足され、全シーンで固定されます。参照が太ももで切れているとモデルは膝下を勝手に補う（短パンになる等）ので、脚・靴まで書くのがコツ。jeans/pants を書くと説明文中の shorts は除去"}),
                "look": (list(cls.LOOKS.keys()), {"default": "custom (style text only)",
                         "tooltip": "画調のプリセット。natural color film=自然なカラー / warm cinematic=暖色シネマ / cool cinematic=寒色シネマ / teal and orange / black and white=白黒 / vintage 70s film / 90s music video / high-key clean=明るく清潔 / moody low-key=暗く重厚 / neon night / glossy commercial=広告風 / documentary。custom = プリセットなし（style 欄の文だけ）。選んだ文の後ろに style 欄の文が足されます"}),
                "lighting": (list(cls.LIGHTINGS.keys()), {"default": "auto (from photo)",
                             "tooltip": "照明のプリセット。auto=写真の説明に任せる / soft key light / warm tungsten studio=暖色スタジオ / studio softbox / golden hour / overcast daylight / window light / dramatic side light / backlight rim / stage spotlight / neon / candles"}),
                "setting": (list(cls.SETTINGS.keys()), {"default": "auto (from photo)",
                            "tooltip": "舞台（背景）のプリセット。auto=写真の背景のまま。それ以外を選ぶと「写真の背景の代わりにこの舞台」として全シーンに入ります（recording studio / plain backdrop / white cyclorama / concert stage / club stage / theater / rooftop / street at night / neon alley / bedroom / loft / rainy window / bar / warehouse / forest / beach）"}),
                "weather": (list(cls.WEATHERS.keys()), {"default": "none",
                            "tooltip": "天候・空気感。none / light rain=小雨 / heavy rain=大雨 / light snow=小雪 / heavy snow=大雪 / fog=濃霧 / mist=薄い靄 / wind=強風（髪・服がなびく） / storm=嵐 / falling petals=花びら / falling leaves=落ち葉 / dust in sunbeams=光の中の埃 / sparks / confetti / haze with light rays。「shot 全体で続く」と書くのでクリップ途中で消えにくい。屋内の setting と雨雪は矛盾するので注意"}),
                "weather_when": (list(cls.WEATHER_WHEN.keys()), {"default": "all scenes",
                                 "tooltip": "天候をどのシーン（カメラアングルのブロック）に付けるか。all scenes=全シーン / every other scene=1つおき / every third scene=2つおき / first scene only / last scene only / all but the first scene。シーン切替は新規生成＋ディゾルブなので、シーン間で天候が変わっても破綻しません（同じシーン内は hand-off で粒子が続く）"}),
                "vfx": (list(cls.VFX.keys()), {"default": "none",
                        "tooltip": "VFX（演出効果）。money rain=お札が降る / lightning=雷 / fireworks=花火 / pyro sparks=火花 / stage fog=足元の霧 / smoke / bubbles / gold glitter / laser beams / strobe lights / camera flashes / feathers / paper sheets flying / balloons / flames / lens flare / light leaks / floating lanterns / rain of rose petals / shattering glass。※ lightning・strobe・camera flashes は State の handoff_color_match（毎フレームの色正規化）に打ち消されやすいので 0.3 以下に。下の vfx_custom に書くとそちらが優先"}),
                "vfx_when": (list(cls.VFX_WHEN.keys()), {"default": "random (about a third)",
                             "tooltip": "VFX をどのシーンに付けるか。all / 1つおき / 2つおき / 最初だけ / 最後だけ / 最初以外 / random（約半分・約1/3・1シーンだけ）。random は画像とシーン数から決まる固定の並びなので、同じ設定なら redo でも同じシーンに出ます"}),
                "vfx_custom": ("STRING", {"default": "",
                               "tooltip": "任意。効果を自由記述（英語推奨。例: paper talismans (ofuda) falling from above / a dragon made of smoke coiling behind the singer）。空なら上の vfx を使用"}),
                "framing_limit": (cls.FRAMING_LIMITS, {"default": "wide",
                                  "tooltip": "一番引いた構図をどこまで許すか（ズームアウトの上限）。これより広い内蔵アングルはローテーションから外れます（close-up=顔アップまで / chest-up=胸上まで / waist-up=腰上まで / mid-thigh=膝上まで / full body / wide=制限なし）。顔の崩れは「出力フレーム内の顔の画素数」で決まり、目安は顔の高さ ≥120px 安定 / 90〜120px 境界 / <90px 崩れやすい（Stage 1 は半分の解像度、VAE は 32px=1セル）。State の解像度と合わせた推定値がノード下部に表示されます。extra_cameras の自作アングルは対象外"}),
            },
            "optional": {
                "chain": (CHAIN_TYPE, {"tooltip": "任意。ログ用"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("scenes", "action_singing", "action_intro", "character")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    @staticmethod
    def _key(image, num_scenes, style):
        h = hashlib.sha1(np.ascontiguousarray((image[:1] * 255).byte().cpu().numpy())).hexdigest()[:12]
        return hashlib.sha1(f"{h}|{num_scenes}|{style}".encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _num_word(n):
        return ["", "", "two", "three", "four", "five", "six", "seven", "eight"][n] if 0 <= n <= 8 else str(n)

    @classmethod
    def _cache_path(cls, image, num_scenes, style, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras="", microphone="auto", gender="auto", emotion="none", emotion_custom="", motion="none", motion_custom="", height="none", build="none", camera="auto", camera_custom="", outfit="", look="custom (style text only)", lighting="auto (from photo)", setting="auto (from photo)", weather="none", weather_when="all scenes", vfx="none", vfx_when="random (about a third)", vfx_custom="", framing_limit="wide"):
        d = os.path.join(_chains_root(), "_autoprompt")
        os.makedirs(d, exist_ok=True)
        key = f'{style}|n{num_singers}|e{ethnicity}|p{performance}|a{age}|c{extra_cameras}|m{microphone}|g{gender}|x{emotion}|xc{emotion_custom}|v{motion}|vc{motion_custom}|h{height}|b{build}|k{camera}|kc{camera_custom}|o{outfit}|l{look}|li{lighting}|s{setting}|w{weather}|ww{weather_when}|f{vfx}|fw{vfx_when}|fc{vfx_custom}|fr{framing_limit}'
        return os.path.join(d, f"scenes_{cls._key(image, num_scenes, key)}.json")

    @staticmethod
    def _age_phrase(age):
        a = (age or "").strip()
        if not a:
            return ""
        return f"{a}-year-old" if a.isdigit() else a

    @staticmethod
    def _clean_caption(caption):
        t = " ".join((caption or "").split()).strip()
        for pre in ("The image shows ", "The image is ", "This image shows ", "This image is ",
                    "In this image, ", "In the image, ", "The image depicts ", "This is ", "The photo shows ",
                    "It shows ", "Here is ", "A photo of ", "An image of "):
            if t.lower().startswith(pre.lower()):
                t = t[len(pre):]
                break
        # drop a leading framing phrase so "close-up portrait of ..." doesn't bias every shot to a close-up
        low = t.lower()
        for art in ("a ", "an ", "the "):
            if low.startswith(art):
                t = t[len(art):]; low = t.lower(); break
        for fr in ("close-up portrait of ", "close up portrait of ", "cropped portrait of ", "close-up of ",
                   "close up of ", "portrait of ", "headshot of ", "cropped photo of ", "cropped image of ",
                   "photo of ", "picture of ", "image of ", "shot of "):
            if low.startswith(fr):
                t = t[len(fr):]; break
        t = t.rstrip(". ").strip()
        return t[0].lower() + t[1:] if t else t

    def check_lazy_status(self, image, caption, num_scenes, style, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras="", microphone="auto", gender="auto", emotion="none", emotion_custom="", motion="none", motion_custom="", height="none", build="none", camera="auto", camera_custom="", outfit="", look="custom (style text only)", lighting="auto (from photo)", setting="auto (from photo)", weather="none", weather_when="all scenes", vfx="none", vfx_when="random (about a third)", vfx_custom="", framing_limit="wide", chain=None):
        if os.path.isfile(self._cache_path(image, num_scenes, style, num_singers, ethnicity, performance, age, extra_cameras, microphone, gender, emotion, emotion_custom, motion, motion_custom, height, build, camera, camera_custom, outfit, look, lighting, setting, weather, weather_when, vfx, vfx_when, vfx_custom, framing_limit)):
            return []
        return ["caption"]

    @staticmethod
    def _apply_camera(cam, note):
        """Put the camera move at the head of the shot line and remove every 'locked / still' word from the
        template, so the angle no longer contradicts the move (LTX weighs the start of a sentence most)."""
        for l in ("The framing stays exactly the same for the whole shot, camera locked, no zoom and no push-in.",
                  "Camera locked, the dancing stays within the frame.", "Camera locked.", "camera locked,",
                  "camera level and still,", ", camera level and still", "no zoom and no push-in.",
                  "The framing stays exactly the same for the whole shot,"):
            cam = cam.replace(l, "")
        cam = cam.replace("Locked medium shot", "Medium shot").replace("Locked ", "").replace("locked ", "")
        cam = re.sub(r"\s{2,}", " ", cam).replace(" ,", ",").replace(" .", ".").strip().strip(",").strip()
        if cam and not cam.endswith("."):
            cam += "."
        if cam:
            cam = cam[0].upper() + cam[1:]
        return (note.rstrip() + " " + cam).strip()

    @staticmethod
    def _strip_words(desc, words):
        """Drop contradicting adjectives from the caption ("a slim woman" -> "a woman")."""
        pat = r",?\s*\b(?:" + "|".join(re.escape(w) for w in words) + r")\b(?:\s+and\s+(?=\w))?,?\s*"
        t = re.sub(pat, " ", desc, flags=re.I)
        t = re.sub(r"\s{2,}", " ", t); t = re.sub(r"\s+,", ",", t); t = re.sub(r",\s*,", ",", t)
        t = re.sub(r"\ba\s+(?=[aeiouAEIOU])", "an ", t); t = re.sub(r"\ban\s+(?=[^aeiouAEIOU\W])", "a ", t)
        return t.strip().rstrip(",")

    @classmethod
    def _detect_gender(cls, desc):
        """'female' / 'male' / '' from the caption: whichever gendered word appears first wins."""
        words = re.findall(r"[a-z]+", desc.lower())
        for w in words:
            if w in cls.FEMALE_WORDS:
                return "female"
            if w in cls.MALE_WORDS:
                return "male"
        return ""

    @classmethod
    def _force_gender(cls, desc, gender):
        """Rewrite the caption's gendered words so it agrees with the forced gender."""
        table = cls.TO_FEMALE if gender == "female" else cls.TO_MALE
        def sub(m):
            w = m.group(0); r = table.get(w.lower())
            if r is None:
                return w
            return r.capitalize() if w[:1].isupper() else r
        return re.sub(r"[A-Za-z]+", sub, desc)

    @classmethod
    def _emotion(cls, emotion, emotion_custom):
        """(singing clause, listening clause) for the chosen emotion; custom text wins."""
        c = " ".join((emotion_custom or "").split()).strip().rstrip(".")
        if c:
            return c, c
        return cls.EMOTIONS.get(emotion, cls.EMOTIONS["none"])

    @classmethod
    def _no_smile(cls, emotion, emotion_custom):
        c = (emotion_custom or "").strip().lower()
        if c:
            return any(w in c for w in cls.NO_SMILE_WORDS)
        return emotion in cls.NO_SMILE

    @staticmethod
    def _strip_smile(desc):
        """Remove 'smiling' / 'with a big smile' from the caption: the character block is repeated in every
        shot, so a smile written there beats any sad/nostalgic mood asked for in the action."""
        t = desc
        t = re.sub(r",\s*(?:smiling|laughing|grinning|beaming)\s+and\s+", ", ", t, flags=re.I)  # ", laughing and holding"
        t = re.sub(r",?\s*(?:and\s+|while\s+)?(?:is\s+|are\s+)?(?:smiling|laughing|grinning|beaming)"
                   r"(?:\s+(?:broadly|widely|brightly|warmly|happily|softly|gently|cheerfully))?"
                   r"(?:\s+(?:at|into|toward|towards)\s+the\s+(?:camera|viewer))?", "", t, flags=re.I)
        t = re.sub(r",?\s*(?:and\s+)?(?:with|has|have|wears|wearing|flashing|showing)\s+(?:a|an)?\s*(?:\w+\s+){0,2}"
                   r"(?:smile|grin)(?:\s+on\s+(?:her|his|their)\s+face)?", "", t, flags=re.I)
        t = re.sub(r"\b(?:smiling|grinning|laughing|beaming)\s+", "", t, flags=re.I)  # "a smiling woman"
        # sentences that were only about the smile ("She is ." / "She ." / "Her smile ...")
        t = re.sub(r"(?:^|(?<=\.\s))(?:She|He|They|Her|His|Their)\s*(?:is|are)?\s*\.\s*", "", t)
        t = re.sub(r"(?:^|\.\s*)(?:She|He|They|Her|His|Their)\s*(?:is|are)?\s*$", "", t)  # caption already lost its final "."
        t = re.sub(r"(?:^|(?<=\.\s))(?:Her|His|Their)\s+(?:smile|grin)[^.]*\.\s*", "", t, flags=re.I)
        t = re.sub(r"\s+,", ",", t); t = re.sub(r",\s*,", ",", t); t = re.sub(r"\s{2,}", " ", t)
        t = re.sub(r",\s*\.", ".", t).strip().rstrip(",")
        return t

    @classmethod
    def _motion(cls, motion, motion_custom):
        """(singing clause, listening clause, camera note) for the chosen motion; custom text wins."""
        c = " ".join((motion_custom or "").split()).strip().rstrip(".")
        if c:
            return c, c, None
        return cls.MOTIONS.get(motion, cls.MOTIONS["none"])

    @classmethod
    def _wants_mic(cls, microphone, desc):
        """yes/no are explicit; auto = only when the image caption actually mentions a microphone."""
        if microphone == "yes":
            return True
        if microphone == "no":
            return False
        low = f" {desc.lower()} "
        return any(w in low for w in cls.MIC_WORDS)

    def _build(self, image, num_scenes, style, caption, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras="", microphone="auto", gender="auto", emotion="none", emotion_custom="", motion="none", motion_custom="", height="none", build="none", camera="auto", camera_custom="", outfit="", look="custom (style text only)", lighting="auto (from photo)", setting="auto (from photo)", weather="none", weather_when="all scenes", vfx="none", vfx_when="random (about a third)", vfx_custom="", framing_limit="wide"):
        desc = self._clean_caption(caption)
        mic = self._wants_mic(microphone, desc)
        hgt, hdesc = self.HEIGHTS.get(height, ("", ""))
        bld, bdesc = self.BUILDS.get(build, ("", ""))
        if hgt:
            desc = self._strip_words(desc, self.HEIGHT_WORDS)
        if bld:
            desc = self._strip_words(desc, self.BUILD_WORDS)
        if self._no_smile(emotion, emotion_custom):
            desc = self._strip_smile(desc)
        mot_sing, mot_listen, cam_note = self._motion(motion, motion_custom)
        if gender in ("female", "male"):
            desc = self._force_gender(desc, gender)
            g = gender
        else:
            g = self._detect_gender(desc)
        emo_sing, emo_listen = self._emotion(emotion, emotion_custom)
        ns = max(1, int(num_singers))
        eth = ethnicity.strip()
        agep = self._age_phrase(age)
        # keep the face from drifting older over a long video
        age_lock = (" The singer looks exactly the same age in every frame and never ages." if ns == 1
                    else " They look exactly the same age in every frame and never age.") if agep else ""
        clause = self.PERFORMANCE.get(performance, self.PERFORMANCE["restrained"])
        if mot_sing and motion not in ("standing still", "sitting"):
            clause = (clause.replace("barely any head or body movement", "barely any head movement")
                            .replace("the head moving only softly with the melody, small gentle gestures",
                                     "the head moving only softly with the melody"))
        art = lambda t: ("an" if t[:1] and t[:1].lower() in "aeiou" else "a")  # "" -> "a" (a duet)
        # only lock "glasses" when the reference actually has eyewear, otherwise the word itself
        # makes the model add glasses to a person who has none
        has_eyewear = any(w in desc.lower() for w in ("glass", "eyewear", "sunglass", "spectacle", "goggle"))
        feat = "face, hair, glasses, clothing" if has_eyewear else "face, hair, clothing"
        # outfit: what the reference does not show (legs, shoes) has to be spelled out, or the model
        # invents it - a photo cropped at the thigh comes back as shorts. Words that contradict the
        # given outfit are dropped from the caption.
        outfit_txt = " ".join((outfit or "").split()).strip().rstrip(".")
        if outfit_txt:
            low = outfit_txt.lower()
            if any(w in low for w in ("jeans", "pants", "trousers", "slacks", "long skirt", "full-length", "ankle")):
                desc = re.sub(r",?\s*(?:and\s+)?(?:denim\s+|jean\s+|short\s+)?(?:shorts|short pants|hot pants|cut-offs|cutoffs)\b", "", desc, flags=re.I)
                desc = re.sub(r"\s{2,}", " ", desc).replace(" ,", ",").strip()
            outfit_lock = (f" Outfit: {outfit_txt}. The full outfit, including the legs and shoes, stays exactly "
                           f"the same in every shot and in every framing.")
        else:
            outfit_lock = ""
        # setting preset: the scene deliberately replaces the photo's background
        set_txt = self.SETTINGS.get(setting, "")
        set_lock = (f" Setting: {set_txt}, replacing the background of the reference photo; the same setting in "
                    f"every shot.") if set_txt else ""
        # "(a 25-year-old Japanese woman)" — the noun follows the gender; only written out when
        # the user forced a gender or gave age/ethnicity, so an unlabelled caption stays untouched
        noun = {"female": "woman", "male": "man"}.get(g, "person")
        gtag = g if gender in ("female", "male") else ""
        body = ", ".join(x for x in (hgt, bld) if x)  # "tall, slim"
        body = (body + " ") if body else ""
        # keep the proportions from drifting across the video (full-body angles especially)
        if hgt or bld:
            what = ", ".join(x for x in ("height" if hgt else "", "body shape" if bld else "") if x)
            bodyd = "; ".join(x for x in (hdesc, bdesc) if x)
            whose = "The singer's" if ns == 1 else "Each singer's"
            body_lock = (f" {whose} body: {bodyd}. The {what} and proportions stay "
                         f"exactly the same in every shot and in every framing, including full-body shots.")
        else:
            body_lock = ""
        if ns == 1:
            who = body + " ".join(x for x in (agep, eth) if x)
            who = who.strip()
            tag = f" ({art(who or noun)} {who + ' ' if who else ''}{noun})" if (who or gtag) else ""
            feat_s = (feat.rsplit(", ", 1)[0] + " and " + feat.rsplit(", ", 1)[1]) if set_txt else feat
            character = (f"Image 1 is the singer{tag}: {desc}. The singer's {feat_s}{'' if set_txt else ' and the setting'} "
                         f"stay exactly the same as in the reference image in every shot.{set_lock}{outfit_lock}{age_lock}{body_lock}")
        else:
            word = self._num_word(ns)
            pre = body + " ".join(x for x in (agep, eth, gtag) if x)
            pre = pre.strip()
            pre = (pre + " ") if pre else ""
            group = f"{art(pre)} {pre}duet" if ns == 2 else f"a group of {word} {pre}singers"
            allof = "Both of them" if ns == 2 else f"All {word} of them"
            character = (f"Image 1 shows the {word} singers ({group}): {desc}. {allof} appear together "
                         f"in every shot; their faces, hair{' and' if set_txt else ','} clothing{'' if set_txt else ' and the setting'} stay exactly the same as in "
                         f"the reference image.{set_lock}{outfit_lock}{age_lock}{body_lock}")
        # camera move: custom text > preset > the follow-camera a moving motion asks for > locked (None)
        cc = " ".join((camera_custom or "").split()).strip()
        if cc:
            cc = cc[0].upper() + cc[1:]
            cam_move = cc if cc.endswith(".") else cc + "."
        elif camera == "mix (varies per scene)":
            cam_move = "mix"
        elif self.CAMERA_MOVES.get(camera):
            cam_move = self.CAMERA_MOVES[camera]
        else:
            cam_move = cam_note
        # the move is repeated as the last sentence of the prompt (the action comes last), where the
        # model also pays close attention - a single mention buried after a long caption gets ignored
        tail_cam = "" if cam_move in (None, "mix") else (" Camera: " + cam_move[0].lower() + cam_move[1:])
        act_sing = self._sing(ns, clause, mic, emo_sing, mot_sing) + tail_cam
        act_intro = self._intro(ns, mic, emo_listen, mot_listen) + tail_cam
        # look + lighting presets + free style text -> the tail of every scene prompt
        parts = [self.LOOKS.get(look, ""), self.LIGHTINGS.get(lighting, ""), style.strip()]
        tail_txt = ", ".join(x.strip().rstrip(",.") for x in parts if x and x.strip())
        tail = ("\n\n" + tail_txt) if tail_txt else ""
        # camera rotation = the first `num_scenes` built-in angles + any custom angles the user added
        extras = [ln.strip() for ln in (extra_cameras or "").replace("\r", "").split("\n") if ln.strip()]
        base = self.CAMERAS_MIC if mic else self.CAMERAS_NOMIC
        limit = self.FRAMING_RANK.get(framing_limit, 99)
        chosen = [c for c, fr in zip(base[:max(1, int(num_scenes))], self.ANGLE_FRAMING)
                  if self.FRAMING_RANK.get(fr, 0) <= limit]
        if not chosen:  # a limit tighter than every selected angle: keep the tightest one
            chosen = [base[1]]
        dropped = max(1, int(num_scenes)) - len(chosen)
        if dropped:
            logging.info(f"[LTX Chain] framing_limit={framing_limit}: {dropped} built-in angle(s) wider than that dropped")
        cams = chosen + extras
        if cam_move == "mix":
            moving = motion in self.MOVING_MOTIONS or bool((motion_custom or "").strip())
            mix = self.CAMERA_MIX_MOVING if moving else self.CAMERA_MIX
            cams = [self._apply_camera(c, self.CAMERA_MOVES[mix[i % len(mix)]]) for i, c in enumerate(cams)]
        elif cam_move:
            cams = [self._apply_camera(c, cam_move) for c in cams]
        if motion == "sitting" and not (motion_custom or "").strip():
            cams = [c.replace("standing upright", "seated") for c in cams]
        if ns > 1:  # phrase every angle for more than one person
            cams = [c.replace("the singer's", "the singers'").replace("the singer ", "the singers ")
                     .replace("at the singer", "at the singers").replace("the singer,", "the singers,")
                    for c in cams]
        # weather goes on the shot line of the chosen blocks only (scenes cycle through the blocks)
        wtxt = self.WEATHERS.get(weather, "")
        when = self.WEATHER_WHEN.get(weather_when, self.WEATHER_WHEN["all scenes"])
        if wtxt:
            cams = [(c.rstrip() + " Weather: " + wtxt + ".") if when(i, len(cams)) else c for i, c in enumerate(cams)]
        # VFX on the chosen blocks; "random" uses a fixed seed from the image + settings so redo
        # runs of the same session land the effect on the same scenes
        vtxt = " ".join((vfx_custom or "").split()).strip().rstrip(".") or self.VFX.get(vfx, "")
        if vtxt:
            import random
            seed_src = f"{self._key(image, num_scenes, vtxt) if image is not None else vtxt}|{len(cams)}"
            rng = random.Random(int(hashlib.sha1(seed_src.encode("utf-8")).hexdigest()[:8], 16))
            picked = self.VFX_WHEN.get(vfx_when, self.VFX_WHEN["all scenes"])(len(cams), rng)
            cams = [(c.rstrip() + " VFX: " + vtxt + ".") if i in picked else c for i, c in enumerate(cams)]
        scenes = [character + "\n\n" + c + tail for c in cams]
        return {"character": character, "scenes": "\n---\n".join(scenes),
                "action_singing": act_sing, "action_intro": act_intro, "microphone": bool(mic),
                "gender": g or "unknown", "emotion": (emotion_custom.strip() or emotion),
                "motion": (motion_custom.strip() or motion), "height": height, "build": build,
                "camera": (camera_custom.strip() or camera), "outfit": outfit_txt,
                "look": look, "lighting": lighting, "setting": setting, "weather": weather, "weather_when": weather_when,
                "vfx": (vfx_custom.strip() or vfx), "vfx_when": vfx_when, "framing_limit": framing_limit}

    def run(self, image, caption, num_scenes, style, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras="", microphone="auto", gender="auto", emotion="none", emotion_custom="", motion="none", motion_custom="", height="none", build="none", camera="auto", camera_custom="", outfit="", look="custom (style text only)", lighting="auto (from photo)", setting="auto (from photo)", weather="none", weather_when="all scenes", vfx="none", vfx_when="random (about a third)", vfx_custom="", framing_limit="wide", chain=None):
        path = self._cache_path(image, num_scenes, style, num_singers, ethnicity, performance, age, extra_cameras, microphone, gender, emotion, emotion_custom, motion, motion_custom, height, build, camera, camera_custom, outfit, look, lighting, setting, weather, weather_when, vfx, vfx_when, vfx_custom, framing_limit)
        clip = (chain.get("index", 0) + 1) if chain else 1
        if os.path.isfile(path):
            data = json.load(open(path, encoding="utf-8"))
            logging.info(f"[LTX Chain] clip {clip}: using cached auto-scenes {os.path.basename(path)}")
        else:
            data = self._build(image, num_scenes, style, caption or "", num_singers, ethnicity, performance, age, extra_cameras, microphone, gender, emotion, emotion_custom, motion, motion_custom, height, build, camera, camera_custom, outfit, look, lighting, setting, weather, weather_when, vfx, vfx_when, vfx_custom, framing_limit)
            json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            logging.info(f"[LTX Chain] clip {clip}: analysed image -> auto-scenes cached ({os.path.basename(path)}), "
                         f"microphone={microphone} -> {'in frame' if data.get('microphone') else 'none'}, "
                         f"gender={gender} -> {data.get('gender')}, emotion={data.get('emotion')}, motion={data.get('motion')}, "
                         f"height={height}, build={build}, camera={data.get('camera')}")
            logging.info(f"[LTX Chain] character: {data['character']}")
            import gc
            import comfy.model_management as mm
            mm.unload_all_models(); gc.collect(); mm.soft_empty_cache()
            logging.info("[LTX Chain] image analysed -> models unloaded to free RAM/VRAM for generation")
        return (data["scenes"], data["action_singing"], data["action_intro"], data["character"])


class LTXChainPromptCache:
    """Runs the (expensive) prompt enhancer only once per session and prompt text.

    `enhanced` is a lazy input: on later clips of the same session the cached text is
    returned and the enhancer upstream is never executed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": (CHAIN_TYPE,),
                "raw_prompt": ("STRING", {"forceInput": True, "tooltip": "Enhancer に入れている元のプロンプト（キャッシュのキーに使う）"}),
                "enhanced": ("STRING", {"forceInput": True, "lazy": True, "tooltip": "Enhancer の出力。キャッシュがあるクリップでは評価されません"}),
                "free_enhancer_after": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Enhancer を実行した直後にモデルを全部アンロードして RAM/VRAM を空ける（Gemma を乗せたまま動画生成に入ってクラッシュするのを防ぐ）"}),
                "use_cache": ("BOOLEAN", {"default": True,
                                          "tooltip": "オン: 同じセッションでは最初の1回だけ Enhancer を実行し、以降はその文章を使い回す"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    @staticmethod
    def _cache_path(chain, raw_prompt):
        key = hashlib.sha1(raw_prompt.encode("utf-8")).hexdigest()[:10]
        return os.path.join(_session_dir(chain["session"]), f"prompt_{key}.txt")

    def check_lazy_status(self, chain, raw_prompt, free_enhancer_after, use_cache, enhanced=None):
        if use_cache and os.path.isfile(self._cache_path(chain, raw_prompt)):
            return []
        return ["enhanced"]

    def run(self, chain, raw_prompt, free_enhancer_after, use_cache, enhanced=None):
        path = self._cache_path(chain, raw_prompt)
        if use_cache and os.path.isfile(path):
            text = open(path, encoding="utf-8").read()
            logging.info(f"[LTX Chain] clip {chain['index'] + 1}: using cached prompt {os.path.basename(path)}")
            return (text,)
        text = enhanced if enhanced is not None else raw_prompt
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"[LTX Chain] clip {chain['index'] + 1}: prompt generated and cached -> {os.path.basename(path)}")
        if free_enhancer_after and enhanced is not None:
            # The enhancer LLM plus the LTX text encoder and transformer do not fit in RAM
            # together on a 32 GB machine; drop everything now so the video models load clean.
            import gc
            import comfy.model_management as mm
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            logging.info("[LTX Chain] enhancer done -> all models unloaded to free RAM/VRAM")
        return (text,)


class LTXChainFaceOutputName:
    """Output name for the ReActor post-process workflow: from the path of a finished video
    (.../LTX2.5Chains/<session>/final.mp4) build `LTX2.5Chains/<session>-<suffix>/<suffix>`, so
    Video Combine writes next to the session folder instead of a generic output/ReActor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "元動画のフルパス（Load Video (Path) と同じ文字列をつなぐ）"}),
                "suffix": ("STRING", {"default": "final_face",
                                      "tooltip": "フォルダ名とファイル名に付ける語。<セッション>-<suffix>/<suffix> になります"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filename_prefix", "session")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    def run(self, video_path, suffix):
        path = (video_path or "").strip().strip('"')
        suffix = (suffix or "final_face").strip() or "final_face"
        parent = os.path.basename(os.path.dirname(path)) if path else ""
        stem = os.path.splitext(os.path.basename(path))[0] if path else ""
        session = parent if parent and parent.lower() != CHAINS_SUBDIR.lower() else (stem or "video")
        prefix = f"{CHAINS_SUBDIR}/{session}-{suffix}/{suffix}"
        logging.info(f"[LTX Chain] face output: {prefix}")
        return (prefix, session)


class LTXChainLastFrames:
    """The clip's last `handoff` frames (from the chain), for the identity anchor: decode -> LastFrames
    -> ReActor (-> Face Keep Mouth) -> Step.handoff_images. ReActor then swaps 9 frames, not 240."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "デコードしたクリップの全フレーム"}),
                "chain": (CHAIN_TYPE, {"tooltip": "State の chain（hand-off 枚数を読む）"}),
            },
            "optional": {
                "extra": ("INT", {"default": 0, "min": 0, "max": 64,
                                  "tooltip": "hand-off 枚数に足して切り出す枚数（通常 0）"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frames",)
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    def run(self, images, chain, extra=0):
        k = min(int(chain.get("handoff", 1)) + int(extra), images.shape[0])
        return (images[-k:],)


_FACE_APP = None


def _face_app():
    """insightface buffalo_l (the same detector ReActor uses), loaded once, CUDA if available."""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        root = os.path.join(folder_paths.models_dir, "insightface")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", root=root, providers=providers, allowed_modules=["detection"])
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        _FACE_APP = app
        logging.info(f"[LTX Chain] face detector loaded ({providers[0]})")
    return _FACE_APP


class LTXChainFaceKeepMouth:
    """Face swap + lip sync: keep the identity from the swapped frame (eyes, nose, cheeks) but put the
    ORIGINAL mouth/chin back, so the swapper's habit of closing the mouth (and its frame-to-frame jumps
    when the mouth opens wide) never reaches the output. Per frame: detect the face, build a soft ellipse
    around the mouth from the 5 landmarks, blend original over swapped inside it. Frames with no face
    reuse the previous mask so nothing pops."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "swapped": ("IMAGE", {"tooltip": "ReActor の SWAPPED_IMAGE（顔を入れ替えた後）"}),
                "original": ("IMAGE", {"tooltip": "ReActor の ORIGINAL_IMAGE（入れ替え前。口の動きはこちらを使う）"}),
                "mouth_width": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                                          "tooltip": "口の幅（口角の距離）の何倍を戻すか（横）。大きいほど頬まで元に戻る"}),
                "mouth_height": ("FLOAT", {"default": 1.4, "min": 0.5, "max": 4.0, "step": 0.1,
                                           "tooltip": "口の幅の何倍を戻すか（縦）。1.4 で口〜顎先あたり。鼻まで戻したくなければ小さく"}),
                "down_shift": ("FLOAT", {"default": 0.15, "min": -0.5, "max": 1.0, "step": 0.05,
                                         "tooltip": "楕円の中心を口から顎側へずらす量（口幅比）。顎の切り替わりが目立つなら大きく"}),
                "feather": ("INT", {"default": 15, "min": 0, "max": 64, "step": 1,
                                    "tooltip": "境界のぼかし幅（px）。継ぎ目が見えるなら大きく"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "1.0 = 口の中は完全に元フレーム / 小さくすると入れ替え後の顔を少し混ぜる"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mouth_mask")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    @staticmethod
    def _mask_for(bgr, mouth_width, mouth_height, down_shift, feather):
        import cv2
        faces = _face_app().get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        k = f.kps  # [left eye, right eye, nose, left mouth, right mouth]
        le, re, lm, rm = k[0], k[1], k[3], k[4]
        mw = float(np.linalg.norm(rm - lm))
        if mw < 2:
            return None
        eye_c = (le + re) / 2
        mouth_c = (lm + rm) / 2
        down = mouth_c - eye_c
        down = down / (np.linalg.norm(down) + 1e-6)
        centre = mouth_c + down * (down_shift * mw)
        angle = math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0])))
        h, w = bgr.shape[:2]
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (int(round(centre[0])), int(round(centre[1]))),
                    (max(1, int(round(mw * mouth_width / 2))), max(1, int(round(mw * mouth_height / 2)))),
                    angle, 0, 360, 255, -1)
        if feather > 0:
            kk = feather * 2 + 1
            m = cv2.GaussianBlur(m, (kk, kk), feather / 2.0)
        return m.astype(np.float32) / 255.0

    def run(self, swapped, original, mouth_width, mouth_height, down_shift, feather, strength):
        n = min(swapped.shape[0], original.shape[0])
        if original.shape[1:3] != swapped.shape[1:3]:
            raise ValueError(f"swapped {tuple(swapped.shape[1:3])} and original {tuple(original.shape[1:3])} differ in size")
        out = swapped[:n].clone()
        masks = torch.zeros((n, swapped.shape[1], swapped.shape[2]), dtype=torch.float32)
        last = None
        missed = 0
        for i in range(n):
            bgr = (swapped[i].cpu().numpy()[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
            m = self._mask_for(np.ascontiguousarray(bgr), mouth_width, mouth_height, down_shift, feather)
            if m is None:
                missed += 1
                m = last
            else:
                last = m
            if m is None:
                continue
            mt = torch.from_numpy(m).to(out.device) * float(strength)
            masks[i] = mt.cpu()
            mt = mt.unsqueeze(-1)
            out[i] = swapped[i] * (1 - mt) + original[i].to(out.device) * mt
        logging.info(f"[LTX Chain] keep-mouth composite: {n} frames, {missed} without a detected face (previous mask reused)")
        return (out.cpu(), masks)


NODE_CLASS_MAPPINGS = {
    "LTXChainState": LTXChainState,
    "LTXChainStep": LTXChainStep,
    "LTXChainPromptSwitch": LTXChainPromptSwitch,
    "LTXChainScenePrompt": LTXChainScenePrompt,
    "LTXChainSceneCuts": LTXChainSceneCuts,
    "LTXChainAutoScenes": LTXChainAutoScenes,
    "LTXChainPromptCache": LTXChainPromptCache,
    "LTXChainFaceOutputName": LTXChainFaceOutputName,
    "LTXChainFaceKeepMouth": LTXChainFaceKeepMouth,
    "LTXChainLastFrames": LTXChainLastFrames,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXChainState": "LTX Chain: State (start of chain)",
    "LTXChainStep": "LTX Chain: Step (save + next)",
    "LTXChainPromptSwitch": "LTX Chain: Prompt Switch (intro / singing)",
    "LTXChainScenePrompt": "LTX Chain: Scene Prompt",
    "LTXChainSceneCuts": "LTX Chain: Scene Cuts (waveform)",
    "LTXChainAutoScenes": "LTX Chain: Auto Scenes (image -> prompt)",
    "LTXChainPromptCache": "LTX Chain: Prompt Cache (enhance once)",
    "LTXChainFaceOutputName": "LTX Chain: Face Output Name (ReActor)",
    "LTXChainFaceKeepMouth": "LTX Chain: Face Keep Mouth (swap + lip sync)",
    "LTXChainLastFrames": "LTX Chain: Last Frames (hand-off for ReActor)",
}
