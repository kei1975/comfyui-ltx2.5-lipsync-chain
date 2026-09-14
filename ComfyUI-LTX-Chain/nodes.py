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
    t = target.float()
    mean_t = t.mean(dim=(0, 1, 2))
    std_t = t.std(dim=(0, 1, 2)) + 1e-6
    mean_i = x.mean(dim=(1, 2))                      # [T, 3]
    std_i = x.std(dim=(1, 2)) + 1e-6                 # [T, 3]
    if window > 1 and x.shape[0] > 1:
        w = min(window, x.shape[0])
        w += (w + 1) % 2                             # odd
        pad = w // 2
        kernel = torch.ones(3, 1, w, dtype=x.dtype) / w
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
                                             "tooltip": "書き出しサイズ。LTX-2.5 が安全に出せる 64 の倍数のサイズから選ぶ。auto = 画像の縦横比のまま generation_width×height の面積で決める。数値を選ぶとその WxH で書き出し（画像はその比率に中央クロップ）。大きいほど VRAM を使うので 12GB では ~600k px 目安"}),
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
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "INT", CHAIN_TYPE, "STRING", "INT", "INT", "BOOLEAN", "BOOLEAN", "INT", "BOOLEAN",
                    "INT", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "start_sec", "num_seconds", "chunk_index", "total_chunks", "chain", "info", "width", "height",
                    "use_msr_stage1", "use_msr_stage2", "scene_index", "bypass_image", "seed", "end_image", "pin_end")
    OUTPUT_TOOLTIPS = (None, None, None, None, None, None, None, None, None, None, None,
                       "このクリップのシーン番号（0 始まり）。Scene Prompt ノードが使う",
                       "True = シーン切り替え直後のクリップ。Stage 1 の LTXVImgToVideoInplace の bypass につなぐ（受け渡しフレームを使わず参照だけから生成）",
                       "このクリップ用のシード（RandomNoise の noise_seed につなぐ）",
                       "作り直しのとき、次のクリップとの継ぎ目に合わせるための終端フレーム（LTXVAddGuide の image につなぐ）",
                       "True = 終端フレームを固定する（作り直しで次のクリップがあるとき）。AddGuide の切り替えスイッチにつなぐ")
    FUNCTION = "run"
    CATEGORY = "LTX Chain"

    def run(self, image, audio, audio_start_sec, chunk_seconds, length_mode, length_seconds,
            fps, chain_iter, chain_state, resolution=RES_PRESETS[0], generation_width=609, generation_height=1056,
            msr_clips="stage2_all", clips_per_scene=0, scene_cuts="", overlap_frames=8, handoff_color_match=1.0,
            scene_crossfade=True, seed=42, redo_session="", redo_clips=""):
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
                                        "overlap": overlap, "target": target, "cuts": cuts, "clips_per_scene": int(clips_per_scene)},
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
                use_msr_stage1, use_msr_stage2, scene_index, scene_start, clip_seed, end_image, pin_end)


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
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "LTX Chain"

    def run(self, images, chain, final_name, final_crf, auto_continue, chunk_audio=None,
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
        hand = (images[-k:] * 255).clamp(0, 255).byte().cpu().numpy()
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

    CAMERAS = [
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

    def _sing(self, ns, clause):
        common = ("Single continuous take, photorealistic, crisp focus on the face"
                  f"{'s' if ns > 1 else ''}, fine skin and fabric texture, steady exposure and white balance. "
                  "Audio: the vocal performance over the song with quiet room tone.")
        if ns == 1:
            return ("The singer performs the song into the microphone with precise lip sync to the vocals: the mouth "
                    "shapes match every word, breaths between phrases, the mouth closed during instrumental passages. "
                    f"{clause}. " + common)
        who = "The two singers" if ns == 2 else f"The {self._num_word(ns)} singers"
        together = ("trade lines and harmonise" if ns == 2 else "sing together as an ensemble")
        return (f"{who} perform the song together and {together}: each one lip-syncs their own part with precise timing "
                "to the vocals, mouths matching every word, breaths between phrases, mouths closed during instrumental "
                f"passages. {clause}. " + common)

    def _intro(self, ns):
        if ns == 1:
            return self.ACTION_INTRO
        who = "The two singers" if ns == 2 else f"The {self._num_word(ns)} singers"
        return (f"{who} are not singing yet: lips closed, listening to the music, swaying gently to the rhythm, heads "
                "nodding softly on the beat, glancing at each other and at the microphone while waiting for the cue. "
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
                                     "tooltip": "全シーン共通で足したい雰囲気（照明・質感など）。※全シーンに付くので、カメラアングルはここに書かない"}),
                "extra_cameras": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "任意。自分で足したいカメラアングルを1行に1つ（例: profile shot from the side, 85mm, the microphone between her and the camera）。ここに書いた分がシーンのローテーションに追加され、時々そのアングルになります。空なら既定の雛形だけ"}),
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
    def _cache_path(cls, image, num_scenes, style, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras=""):
        d = os.path.join(_chains_root(), "_autoprompt")
        os.makedirs(d, exist_ok=True)
        key = f'{style}|n{num_singers}|e{ethnicity}|p{performance}|a{age}|c{extra_cameras}'
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

    def check_lazy_status(self, image, caption, num_scenes, style, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras="", chain=None):
        if os.path.isfile(self._cache_path(image, num_scenes, style, num_singers, ethnicity, performance, age, extra_cameras)):
            return []
        return ["caption"]

    def _build(self, image, num_scenes, style, caption, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras=""):
        desc = self._clean_caption(caption)
        ns = max(1, int(num_singers))
        eth = ethnicity.strip()
        agep = self._age_phrase(age)
        # keep the face from drifting older over a long video
        age_lock = (" The singer looks exactly the same age in every frame and never ages." if ns == 1
                    else " They look exactly the same age in every frame and never age.") if agep else ""
        clause = self.PERFORMANCE.get(performance, self.PERFORMANCE["restrained"])
        art = lambda t: ("an" if t[:1].lower() in "aeiou" and not t[:1].isdigit() else "a")
        # only lock "glasses" when the reference actually has eyewear, otherwise the word itself
        # makes the model add glasses to a person who has none
        has_eyewear = any(w in desc.lower() for w in ("glass", "eyewear", "sunglass", "spectacle", "goggle"))
        feat = "face, hair, glasses, clothing" if has_eyewear else "face, hair, clothing"
        if ns == 1:
            who = " ".join(x for x in (agep, eth) if x)
            tag = f" ({art(who)} {who} person)" if who else ""
            character = (f"Image 1 is the singer{tag}: {desc}. The singer's {feat} and the "
                         f"setting stay exactly the same as in the reference image in every shot.{age_lock}")
        else:
            word = self._num_word(ns)
            pre = " ".join(x for x in (agep, eth) if x)
            pre = (pre + " ") if pre else ""
            group = f"{art(pre)} {pre}duet" if ns == 2 else f"a group of {word} {pre}singers"
            allof = "Both of them" if ns == 2 else f"All {word} of them"
            character = (f"Image 1 shows the {word} singers ({group}): {desc}. {allof} appear together "
                         f"in every shot; their faces, hair, clothing and the setting stay exactly the same as in "
                         f"the reference image.{age_lock}")
        act_sing = self._sing(ns, clause)
        act_intro = self._intro(ns)
        tail = ("\n\n" + style.strip()) if style.strip() else ""
        # camera rotation = the first `num_scenes` built-in angles + any custom angles the user added
        extras = [ln.strip() for ln in (extra_cameras or "").replace("\r", "").split("\n") if ln.strip()]
        cams = list(self.CAMERAS[:max(1, int(num_scenes))]) + extras
        if ns > 1:  # phrase every angle for more than one person
            cams = [c.replace("the singer's", "the singers'").replace("the singer ", "the singers ")
                     .replace("at the singer", "at the singers").replace("the singer,", "the singers,")
                    for c in cams]
        scenes = [character + "\n\n" + c + tail for c in cams]
        return {"character": character, "scenes": "\n---\n".join(scenes),
                "action_singing": act_sing, "action_intro": act_intro}

    def run(self, image, caption, num_scenes, style, num_singers=1, ethnicity="", performance="restrained", age="", extra_cameras="", chain=None):
        path = self._cache_path(image, num_scenes, style, num_singers, ethnicity, performance, age, extra_cameras)
        clip = (chain.get("index", 0) + 1) if chain else 1
        if os.path.isfile(path):
            data = json.load(open(path, encoding="utf-8"))
            logging.info(f"[LTX Chain] clip {clip}: using cached auto-scenes {os.path.basename(path)}")
        else:
            data = self._build(image, num_scenes, style, caption or "", num_singers, ethnicity, performance, age, extra_cameras)
            json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            logging.info(f"[LTX Chain] clip {clip}: analysed image -> auto-scenes cached ({os.path.basename(path)})")
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


NODE_CLASS_MAPPINGS = {
    "LTXChainState": LTXChainState,
    "LTXChainStep": LTXChainStep,
    "LTXChainPromptSwitch": LTXChainPromptSwitch,
    "LTXChainScenePrompt": LTXChainScenePrompt,
    "LTXChainSceneCuts": LTXChainSceneCuts,
    "LTXChainAutoScenes": LTXChainAutoScenes,
    "LTXChainPromptCache": LTXChainPromptCache,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXChainState": "LTX Chain: State (start of chain)",
    "LTXChainStep": "LTX Chain: Step (save + next)",
    "LTXChainPromptSwitch": "LTX Chain: Prompt Switch (intro / singing)",
    "LTXChainScenePrompt": "LTX Chain: Scene Prompt",
    "LTXChainSceneCuts": "LTX Chain: Scene Cuts (waveform)",
    "LTXChainAutoScenes": "LTX Chain: Auto Scenes (image -> prompt)",
    "LTXChainPromptCache": "LTX Chain: Prompt Cache (enhance once)",
}
