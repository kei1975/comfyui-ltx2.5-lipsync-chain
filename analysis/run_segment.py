"""Submit one member's segment straight to ComfyUI, with no browser involved.

    python -s run_segment.py <workflow.json> <member 1-4> <start_sec> <length_sec> [--go]

comfy-cli cannot convert this graph: its cached object_info still describes LTXChainState with 7
outputs (the live server reports 18) and does not know LTXChainAutoScenes at all, so
`comfy run`/validate mangle it. The way round that is to never convert anything: ComfyUI keeps the
already-resolved API prompt of every run in /history, so this takes that graph as-is and only
transplants widget values into it from the workflow file, which `set_member.py` is the single
place that edits. Inputs wired to another node are left alone - only literal widget values move.

Without --go it prints what it would change and submits nothing.
"""
import json
import sys
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:8188"
# the snapshot to build on: any past run of THIS graph; its widget values are overwritten below
SNAPSHOT_HINT = "576x1024"

OUTFITS = {
    "01": ("a soft cream white crew-neck sweatshirt with arched navy varsity lettering reading GALO "
           "on the left chest, worn loose and untucked, plain black trousers, white trainers"),
    "02": ("a navy crew-neck sweatshirt with arched white varsity lettering reading GALO on the left "
           "chest, worn under an open black zip-up blouson with a stand collar, black trousers, "
           "dark trainers"),
    "03": ("a black crew-neck sweatshirt with arched white varsity lettering reading GALO on the left "
           "chest, worn loose and untucked, black trousers, dark trainers"),
    "04": ("a black crew-neck sweatshirt with arched white varsity lettering reading GALO on the left "
           "chest, worn loose and untucked, black trousers, white trainers, heavy black-framed "
           "rectangular glasses"),
}


def get(path):
    with urllib.request.urlopen(f"{HOST}{path}") as r:
        return json.load(r)


def newest_snapshot():
    """The most recent history entry whose graph is this workflow at the right size."""
    hist = get("/history?max_items=6")
    best = None
    for pid, entry in hist.items():
        prompt = (entry.get("prompt") or [None, None, {}])[2]
        state = next((v for v in prompt.values() if v.get("class_type") == "LTXChainState"), None)
        if state and SNAPSHOT_HINT in str(state["inputs"].get("resolution", "")):
            best = (pid, prompt)      # dict order is submission order: keep the last match
    if not best:
        raise SystemExit("no usable prompt in /history - run the workflow once from the browser")
    return best


def transplant(prompt, workflow):
    """Copy every literal widget value from the workflow file into the API prompt."""
    wf = json.loads(Path(workflow).read_text(encoding="utf-8"))
    changed = []
    for node in wf["nodes"]:
        nid, named = str(node["id"]), node.get("widgets_values_named")
        if not named or nid not in prompt:
            continue
        for key, value in named.items():
            inputs = prompt[nid]["inputs"]
            if key not in inputs or isinstance(inputs[key], list):
                continue              # absent, or wired to another node: leave it
            if inputs[key] != value:
                changed.append((nid, prompt[nid]["class_type"], key, inputs[key], value))
                inputs[key] = value
    return changed


def main(workflow, member, start, length, go=False):
    member = f"{int(member):02d}"
    pid, prompt = newest_snapshot()
    prompt = json.loads(json.dumps(prompt))       # the history copy stays untouched
    changed = transplant(prompt, workflow)

    state = next(k for k, v in prompt.items() if v["class_type"] == "LTXChainState")
    scenes = next(k for k, v in prompt.items() if v["class_type"] == "LTXChainAutoScenes")
    prompt[state]["inputs"].update(audio_start_sec=float(start), length_mode="seconds",
                                   length_seconds=int(length), chain_iter=0, chain_state="",
                                   redo_session="", redo_clips="")
    prompt[scenes]["inputs"]["outfit"] = OUTFITS[member]

    print(f"snapshot {pid[:8]}  ->  member {member}, {start}-{int(start) + int(length)}s")
    for nid, cls, key, old, new in changed:
        f = lambda v: (v[:40] + "...") if isinstance(v, str) and len(v) > 40 else v
        print(f"  #{nid:<4} {cls[:22]:<22} {key:<18} {f(old)!r} -> {f(new)!r}")
    print(f"  #{scenes:<4} {'LTXChainAutoScenes':<22} {'outfit':<18} -> {OUTFITS[member][:46]}...")
    for k in ("audio_start_sec", "length_mode", "length_seconds"):
        print(f"  #{state:<4} {'LTXChainState':<22} {k:<18} = {prompt[state]['inputs'][k]!r}")

    if not go:
        print("\ndry run - pass --go to submit")
        return 0

    body = json.dumps({"prompt": prompt, "client_id": "claude-galo"}).encode()
    req = urllib.request.Request(f"{HOST}/prompt", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        out = json.load(r)
    print("\nsubmitted:", out.get("prompt_id"), "| queue position", out.get("number"))
    if out.get("node_errors"):
        print("node_errors:", json.dumps(out["node_errors"])[:500])
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--go"]
    if len(args) != 4:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(*args, go="--go" in sys.argv))
