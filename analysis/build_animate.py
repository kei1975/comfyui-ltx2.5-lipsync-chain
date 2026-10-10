"""Derive a plain REF+ReActor animation workflow from the lip-sync chain workflow.

Built by DELETION rather than from scratch: every surviving node keeps the exact configuration
that is already known to run here, which a hand-written graph would have to rediscover. What goes
is the chain (State/Step/AutoScenes/ScenePrompt/...), the song (LoadAudio/MelBandRoFormer/Trim),
the Florence-2 captioner and the end-pin guide; what arrives in their place is a handful of
primitives for the values the chain used to supply.

The audio half of the graph STAYS. LTX-2.5 is an audio-video model - the official i2v template
keeps LTXVEmptyLatentAudio/ConcatAVLatent/SeparateAVLatent too - so the video cannot be generated
without it. The difference is that nothing is encoded INTO it any more: the empty audio latent goes
in, and whatever the model invents for the scene comes back out of LTXVAudioVAEDecode, which this
workflow wires to the video instead of discarding.

ReActor also moves: in the chain it only ever touched the 9 hand-off frames, because those were
what the next clip grew from. Here there is no next clip, so it runs over the whole decoded take.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

PROMPT = ("The man walks slowly toward the camera along a sunlit Harajuku back street, looking "
          "around and breaking into a grin, his arms swinging naturally, the camera gliding "
          "backward at his pace so he stays the same size in frame.")
NEGATIVE = ("blurry, low quality, still frame, frozen, watermark, overlay, titles, subtitles, "
            "distorted face, distorted anatomy, wrong body proportions, oversized head, "
            "elongated neck, extra fingers, deformed hands")


DROP = {
    24, 178, 179, 189, 198, 205, 208, 209,   # the chain
    135, 136, 137, 144, 146, 148,            # the song and its vocal separation
    203, 204, 211,                           # Florence-2 captioning
    199, 200, 201, 202,                      # the end-pin guide and its gates
    36,                                      # prompt enhancer (already bypassed)
    155, 156, 157,                           # "use the song as the audio latent?" switch
    129, 130, 131, 132, 145,                 # and the mask branch that fed that switch
    133, 134, 138, 151, 152, 153, 154,       # frame count derived from the song length
}

NEW = [
    (900, "easy int", [10], "秒数 (num_seconds)"),
    (901, "easy int", [576], "幅 (width)"),
    (902, "easy int", [1024], "高さ (height)"),
    (903, "PrimitiveInt", [42, "randomize"], "シード"),
    (904, "PrimitiveStringMultiline", [""], "プロンプト（ここに動きを書く）"),
    (905, "easy boolean", [True], "MSR を使う"),
    (906, "easy boolean", [True], "ReActor を使う"),
]

# (dst node, dst input name, src node, src output index)
WIRE = [
    (19, "INT", 900, 0), (149, "INT", 901, 0), (150, "INT", 902, 0),
    (66, "noise_seed", 903, 0), (73, "noise_seed", 903, 0),
    (53, "text", 904, 0),
    (190, "boolean", 905, 0), (191, "boolean", 905, 0), (192, "boolean", 905, 0),
    (193, "boolean", 905, 0), (194, "boolean", 905, 0), (195, "boolean", 905, 0),
    (206, "enabled", 906, 0),
    # the end-pin gates are gone: feed their "no pin" side straight through
    (184, "positive", 35, 0), (184, "negative", 55, 0), (184, "latent", 44, 0),
    (190, "on_false", 35, 0), (191, "on_false", 55, 0), (192, "on_false", 44, 0),
    # the song is gone: the model's own audio latent goes straight into the sampler
    (84, "audio_latent", 83, 0),
    # frame count is now seconds x fps + 1, not the length of a song
    (81, "*", 20, 0),
    # the execution gate used to wait on the chain's image; wait on the loader instead
    (35, "value", 23, 0),
    # ReActor over the whole take, and the generated audio onto the video
    (206, "input_image", 74, 0), (100, "images", 206, 0), (100, "audio", 72, 0),
]

TITLES = {
    23: "① メイン画像 (pic1 / 開始フレーム)",
    206: "ReActor（全フレーム）",
    904: "② プロンプト",
    102: "③ 保存",
}


def main(src, dst):
    wf = json.loads(Path(src).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    proto = {t: next(n for n in wf["nodes"] if n["type"] == t) for _, t, _, _ in NEW}

    for nid, typ, widgets, title in NEW:
        n = deepcopy(proto[typ])
        n.update(id=nid, title=title, mode=0, widgets_values=widgets,
                 pos=[-1400, -600 + 140 * (nid - 900)], size=[280, 90])
        for o in n.get("outputs", []):
            o["links"] = []
        wf["nodes"].append(n)
        nodes[nid] = n

    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in DROP]
    nodes = {n["id"]: n for n in wf["nodes"]}

    # drop links that touch a deleted node, and clear the inputs that held them
    gone = {l[0] for l in wf["links"] if l[1] in DROP or l[3] in DROP}
    wf["links"] = [l for l in wf["links"] if l[0] not in gone]
    for n in wf["nodes"]:
        for i in n.get("inputs", []):
            if i.get("link") in gone:
                i["link"] = None
        for o in n.get("outputs", []):
            o["links"] = [x for x in (o.get("links") or []) if x not in gone]

    lid = wf["last_link_id"]
    for dst_id, name, src_id, slot in WIRE:
        node = nodes[dst_id]
        idx = next(i for i, p in enumerate(node["inputs"]) if p["name"] == name)
        old = node["inputs"][idx].get("link")
        if old is not None:                      # detach whatever was there
            wf["links"] = [l for l in wf["links"] if l[0] != old]
            for n in wf["nodes"]:
                for o in n.get("outputs", []):
                    o["links"] = [x for x in (o.get("links") or []) if x != old]
        lid += 1
        typ = nodes[src_id]["outputs"][slot].get("type", "*")
        wf["links"].append([lid, src_id, slot, dst_id, idx, typ])
        node["inputs"][idx]["link"] = lid
        out = nodes[src_id]["outputs"][slot]
        out["links"] = (out.get("links") or []) + [lid]
    wf["last_link_id"] = lid

    # sensible defaults so the file runs as delivered
    nodes[904]["widgets_values"] = [PROMPT]
    nodes[54]["widgets_values"] = [NEGATIVE]
    nodes[102]["widgets_values"][0] = "video/LTX-REF-Animate"
    for nid, view in ((23, "02"), (182, "face"), (183, "01"), (196, "03"), (197, "04")):
        nodes[nid]["widgets_values"][0] = f"Galo_m01_{view}.png"
    nodes[210]["mode"] = 4          # REF 5 (background) off: a stale image here taints every clip

    # the chain used to do the saving, so these were bypassed; here they ARE the output
    for nid in (100, 102):
        nodes[nid]["mode"] = 0

    for nid, t in TITLES.items():
        if nid in nodes:
            nodes[nid]["title"] = t

    Path(dst).write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")

    # report anything left dangling
    ids = {l[0] for l in wf["links"]}
    bad = []
    for n in wf["nodes"]:
        for i in n.get("inputs", []):
            if i.get("link") is not None and i["link"] not in ids:
                bad.append(f"#{n['id']} {n['type']}.{i['name']} -> missing link {i['link']}")
    setters = {str(n["widgets_values"][0]) for n in wf["nodes"] if n["type"] == "SetNode"}
    orphan = sorted({str(n["widgets_values"][0]) for n in wf["nodes"]
                     if n["type"] == "GetNode" and str(n["widgets_values"][0]) not in setters})
    print(f"nodes {len(wf['nodes'])} (was {len(wf['nodes']) + len(DROP) - len(NEW)}), "
          f"links {len(wf['links'])}")
    print("dangling links:", bad or "none")
    print("GetNode with no SetNode:", orphan or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
