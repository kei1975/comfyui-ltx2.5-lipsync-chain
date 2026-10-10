"""Walk back from the save node and report every input slot that nothing feeds.

The required/optional check missed a broken branch once already: MathExpression declares its
operands optional, so an operand left dangling by a deletion looked fine. Reachability does not
care what the schema calls optional - if the video depends on a slot and the slot is empty, that
is a failure, and if a node is NOT reachable it cannot break the run at all.
"""
import json
import sys
from collections import deque
from pathlib import Path

wf = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
N = {n["id"]: n for n in wf["nodes"]}
L = {l[0]: l for l in wf["links"]}
setter = {str(n["widgets_values"][0]): n["id"] for n in wf["nodes"] if n["type"] == "SetNode"}

start = [n["id"] for n in wf["nodes"] if n["type"] in ("SaveVideo", "CreateVideo")]
seen, q, holes = set(start), deque(start), []
while q:
    nid = q.popleft()
    node = N[nid]
    if node.get("mode") == 4:                      # bypassed: it passes through, nothing to feed
        continue
    nxt = []
    for i in node.get("inputs", []):
        l = L.get(i.get("link"))
        if l:
            nxt.append(l[1])
        elif not (node.get("widgets_values_named") or {}).get(i["name"]):
            holes.append(f"#{nid:<5} {node['type']:<30} .{i['name']}")
    if node["type"] == "GetNode":
        nxt.append(setter.get(str(node["widgets_values"][0])))
    for p in nxt:
        if p is not None and p not in seen:
            seen.add(p)
            q.append(p)

print(f"reachable from the video output: {len(seen)} of {len(N)} nodes")
print("\nunfed input slots on the path to the output:")
for h in holes:
    print("  " + h)
print("  none" if not holes else "")
