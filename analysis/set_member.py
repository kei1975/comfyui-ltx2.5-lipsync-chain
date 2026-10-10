"""Configure the GALO workflow for one member, and optionally for one segment of the track.

    python -s set_member.py <workflow.json> <member 1-4> [start_sec] [length_sec]

With start/length it also sets audio_start_sec, length_mode="seconds" and length_seconds, which is
how the four members are built as four short sessions instead of one long chain: a session only
ever restarts from the reference image at clip 0 (nothing else in the forward path resets it), so
splitting the track is itself the drift control. Omit them to leave the timing alone; pass
length_sec as 0 for length_mode="all".

The four MSR picture slots are four VIEWS OF ONE PERSON, not four people: pic1..pic4 are encoded as
references for the single singer the prompt describes, so mixing members there asks for one face
that is simultaneously all of them. Every slot therefore moves together.

ComfyUI's LoadImage lists only the top level of `input/`, so the files are flat:
Galo_mNN_VV.png, NN = member, VV = view (01 face, 02 upper body, 03 profile, 04 full body).
"""
import json
import sys
from pathlib import Path

# node id -> view file, and what that slot actually feeds
SLOTS = {
    # 02 (standing) and not 01 (the close-up): the END framing is set by the motion, not by the
    # start image - walking lands on a full-body shot whatever it starts from - so starting tight
    # only buys a 374px -> 132px zoom-out at the head of every segment. Measured on 20261010_v01.
    23: ("02", "start image, and MSR pic1"),
    183: ("01", "MSR pic2"),
    196: ("03", "MSR pic3"),
    197: ("04", "MSR pic4"),
}
# #182 feeds ReActor alone, and ReActor is a face SWAP: it wants a real photograph of the person,
# not the stylised portrait the other slots carry. A photo among the MSR references would drag the
# whole video towards photorealism, so it is deliberately kept out of pic1..pic4. Falls back to the
# drawn close-up when no photo has been supplied for that member.
REACTOR = 182
STATE = 178
INPUT_DIR = Path("E:/download/ComfyUI-Easy-Install-Windows/ComfyUI-Easy-Install/ComfyUI/input")


def set_widget(node, name, value):
    named = node["widgets_values_named"]
    assert name in named, f"no widget {name!r}"
    node["widgets_values"][list(named).index(name)] = value
    named[name] = value


def main(workflow, member, start=None, length=None):
    member = f"{int(member):02d}"
    p = Path(workflow)
    wf = json.loads(p.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}

    face = f"Galo_m{member}_face.png"
    if not (INPUT_DIR / face).is_file():
        face = f"Galo_m{member}_01.png"
        print(f"  (no real photo for member {member}; ReActor falls back to the drawn close-up)")
    slots = dict(SLOTS)

    print(f"member {member}")
    for nid, (view, what) in list(slots.items()) + [(REACTOR, (None, "ReActor source (a photo)"))]:
        name = face if nid == REACTOR else f"Galo_m{member}_{view}.png"
        node = nodes[nid]
        mark = "" if node["widgets_values"][0] == name else "  <- changed"
        node["widgets_values"][0] = name
        node["widgets_values_named"]["image"] = name
        print(f"  #{nid:<4} {what:<28} {name}{mark}")

    if start is not None:
        st = nodes[STATE]
        set_widget(st, "audio_start_sec", float(start))
        if length is not None:
            length = float(length)
            set_widget(st, "length_mode", "all" if length <= 0 else "seconds")
            if length > 0:
                set_widget(st, "length_seconds", int(length))
        n = st["widgets_values_named"]
        span = ("to the end of the track" if n["length_mode"] == "all"
                else f"{n['audio_start_sec']:.0f}-{n['audio_start_sec'] + n['length_seconds']:.0f}s")
        print(f"  #{STATE}  segment                      {span}")

    p.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {p.name}")
    return 0


if __name__ == "__main__":
    if not 3 <= len(sys.argv) <= 5:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(*sys.argv[1:]))
