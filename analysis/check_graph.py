"""Pre-flight the built workflow against the LIVE ComfyUI, not comfy-cli's cache.

comfy-cli still believes LTXChainState has 7 outputs and that LTXChainAutoScenes does not exist,
so `comfy validate` cannot judge this graph. The server's own /object_info can.
"""
import json
import sys
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:8188"
# types carried by a wire, never by a widget
LINKED = {"IMAGE", "LATENT", "CONDITIONING", "MODEL", "VAE", "CLIP", "AUDIO", "MASK",
          "SIGMAS", "SAMPLER", "GUIDER", "NOISE", "UPSCALE_MODEL", "VIDEO", "LTX_CHAIN",
          "MSR_PARAMETERS", "FACE_MODEL", "FLOAT", "INT", "BOOLEAN", "STRING"}

wf = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
nodes = [n for n in wf["nodes"] if n.get("mode", 0) != 4 and not n["type"].endswith("Note")]
types = sorted({n["type"] for n in nodes})

info = {}
for t in types:
    try:
        with urllib.request.urlopen(f"{HOST}/object_info/{urllib.parse.quote(t)}", timeout=20) as r:
            info.update(json.load(r))
    except Exception as e:
        print(f"  ! {t}: {e}")

missing_class = [t for t in types if t not in info and t not in ("GetNode", "SetNode")]
print("unknown class_type:", missing_class or "none")

problems = []
for n in nodes:
    spec = info.get(n["type"])
    if not spec:
        continue
    required = spec["input"].get("required", {})
    fed = {i["name"] for i in n.get("inputs", []) if i.get("link") is not None}
    named = n.get("widgets_values_named") or {}
    for name, d in required.items():
        t = d[0] if isinstance(d, list) else d
        if name in fed or name in named:
            continue
        slot = next((i for i in n.get("inputs", []) if i["name"] == name), None)
        if isinstance(t, list):          # a combo: the widget carries it
            continue
        if t in LINKED and slot is not None and slot.get("link") is None:
            problems.append(f"#{n['id']:<5} {n['type']:<36} .{name} ({t}) is unconnected")

print(f"\nchecked {len(nodes)} active nodes across {len(types)} classes")
print("unfed required inputs:")
for p in problems:
    print("  " + p)
if not problems:
    print("  none")
