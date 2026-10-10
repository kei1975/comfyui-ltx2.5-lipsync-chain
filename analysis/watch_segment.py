"""Emit one line per finished clip, and one when the segment ends - however it ends.

Silence has to mean "still working", so an empty queue with no final.mp4 is reported as a stall
rather than waited on forever: that is what a crashed or rejected run looks like from here.
"""
import glob
import json
import os
import sys
import time
import urllib.request

D = sys.argv[1]
WANT = int(sys.argv[2])
idle = 0
seen = -1

while True:
    clips = len(glob.glob(os.path.join(D, "clip_*.mp4")))
    if clips != seen:
        if seen >= 0:
            print(f"clip {clips}/{WANT} done", flush=True)
        seen = clips
        idle = 0

    if os.path.isfile(os.path.join(D, "final.mp4")):
        print(f"SEGMENT COMPLETE - {clips} clips, final.mp4 written", flush=True)
        break

    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=10) as r:
            q = json.load(r)
        busy = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
    except Exception:
        busy = 1          # a failed probe is not evidence of an idle queue

    idle = idle + 1 if busy == 0 else 0
    if idle >= 4:
        print(f"STALLED - queue empty after clip {clips}/{WANT}, no final.mp4", flush=True)
        sys.exit(1)
    time.sleep(20)
