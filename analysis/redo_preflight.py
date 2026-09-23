"""Does this workflow match the session it is about to redo?

`plan.json` records what a session was BUILT with, but LTXChainState never reads that block back
(it loads only `["plan"]`), so every shaping widget has to be set back by hand before a redo.
A mismatch does not warn: a size change crashes in Step when the dissolve meets the old
`seam_NNN.npy` at the old size, and the rest misaligns the clip grid silently.

    python -s redo_preflight.py <workflow.json> [chains_dir]

chains_dir defaults to $LTX_CHAINS_DIR, else ComfyUI/output/LTX2.5Chains next to this checkout's
usual install. Exit code 0 = safe to run.
"""
import json
import os
import sys
from pathlib import Path

DEFAULT_CHAINS = Path(os.environ.get(
    "LTX_CHAINS_DIR",
    "E:/download/ComfyUI-Easy-Install-Windows/ComfyUI-Easy-Install/ComfyUI/output/LTX2.5Chains"))


def main(workflow_path, chains_dir=DEFAULT_CHAINS):
    wf = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    state = next(n for n in wf["nodes"] if n["type"] == "LTXChainState")
    w = state["widgets_values_named"]

    session = str(w.get("redo_session", "")).strip()
    clips = str(w.get("redo_clips", "")).strip()
    print("workflow :", Path(workflow_path).name)
    print("redo     :", f"session={session!r} clips={clips!r}")
    if not clips:
        print("\nnot a redo run (redo_clips empty) - the session settings do not have to match.")
        return 0

    plan_path = Path(chains_dir) / session / "plan.json"
    if not plan_path.is_file():
        print(f"\nplan.json not found: {plan_path}")
        return 1
    s = json.loads(plan_path.read_text(encoding="utf-8"))["settings"]

    res_w, res_h = s["generation"]
    checks = [
        ("resolution (WxH)", f"{res_w}x{res_h}", str(w.get("resolution", "")).split(" ")[0]),
        ("chunk_seconds", s["chunk_seconds"], w.get("chunk_seconds")),
        ("fps", s["fps"], w.get("fps")),
        ("overlap_frames", s["overlap"], w.get("overlap_frames")),
        ("msr_clips", s["msr_clips"], w.get("msr_clips")),
        ("audio_start_sec", s["audio_start_sec"], float(w.get("audio_start_sec", 0))),
        ("clips_per_scene", s["clips_per_scene"], w.get("clips_per_scene")),
    ]

    print(f"\n{'setting':<20} {'session':<16} {'workflow':<16}")
    bad = []
    for name, want, got in checks:
        ok = str(want) == str(got)
        if not ok:
            bad.append(name)
        print(f"{name:<20} {str(want):<16} {str(got):<16} {'OK' if ok else '<-- MISMATCH'}")

    # The end pin must land on the frame the next clip resumes from, i.e. -(overlap + 1).
    guide = next((n for n in wf["nodes"] if n["type"] == "LTXVAddGuide"), None)
    want_idx = -(int(s["overlap"]) + 1)
    if guide is None:
        print("\nend pin  : no LTXVAddGuide in this workflow")
    else:
        slot = next(i for i, p in enumerate(guide["inputs"]) if p["name"] == "frame_idx")
        if guide["inputs"][slot].get("link") is not None:
            print("\nend pin  : linked to State.end_pin_frame (follows overlap automatically)")
        else:
            have = guide["widgets_values"][0]
            print(f"\nend pin  : widget {have}, needs {want_idx}"
                  " - wire State.end_pin_frame to it instead")
            if have != want_idx:
                bad.append("LTXVAddGuide.frame_idx")

    # The stage-2 in-place node re-pins the hand-off after the upsampler drops the noise mask.
    for n in wf["nodes"]:
        if n["type"] == "LTXVImgToVideoInplace" and n.get("mode", 0) != 0:
            bad.append(f"#{n['id']} LTXVImgToVideoInplace is bypassed/muted")
            print(f"#{n['id']} LTXVImgToVideoInplace is bypassed at the node level")

    print("\nnote     : continuity_lead_seconds is ignored on a redo; redo_lead_seconds =",
          w.get("redo_lead_seconds"), "is the one that applies")
    print("\nRESULT:", "SAFE TO RUN" if not bad else "DO NOT RUN - " + ", ".join(map(str, bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(*sys.argv[1:]))
