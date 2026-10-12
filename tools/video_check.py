"""Video input served end to end: short clips the model must describe (what moves, its color, which way, where), a
30-second 720p clip with two scenes in order, a video beside an image, drafted against undrafted, a video and a text
request at once each equal to their replies alone, and a text reply unchanged by the video requests before it.

    python3 -I tools/video_check.py --url http://127.0.0.1:8888 [--json OUT]

Stdlib only: the clips are in tools/clips (drawn with PyAV: a yellow square crossing a gradient left to right in H.264
and VP9, a red square crossing the bottom right to left, a white box crossing the top for 15 s then a still blue box).
Exit 0 when every check passes.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_check import Server, part, shapes  # noqa: E402

CLIPS = Path(__file__).resolve().parent / "clips"


def video(name: str) -> dict:
    mime = {".mp4": "video/mp4", ".webm": "video/webm"}[Path(name).suffix]
    data = base64.b64encode((CLIPS / name).read_bytes()).decode()
    return {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}}


def ask(clip: str, text: str, extra=()) -> list:
    return [{"role": "user", "content": [video(clip), *extra, {"type": "text", "text": text}]}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--json")
    a = ap.parse_args()
    s = Server(a.url)
    what = "What moves in this video, what color is it, and which way does it go? One sentence."
    where = "What moves in this video, what color is it, where in the frame is it vertically, and which way does it go? One sentence."
    checks = [
        ("yellow right, H.264", ask("yellow-right.mp4", what), [["yellow"], ["square"], ["right"]]),
        ("yellow right, VP9", ask("yellow-right.webm", what), [["yellow"], ["square"], ["right"]]),
        ("red left, bottom", ask("red-left.mp4", where), [["red"], ["left"], ["bottom", "lower"]]),
        ("30 s 720p, in order", ask("white-then-blue-720p.mp4", "Describe what happens in this video in order. Two sentences."),
         [["white"], ["blue"], ["then", "after", "later"]]),
        ("video and image", ask("red-left.mp4", "One sentence each: what moves in the video and its color, then the shapes in the image.",
                                [part(shapes())]), [["red"], ["circle"], ["square"]]),
    ]
    text_q = [{"role": "user", "content": "Write two sentences about lighthouses."}]
    before = s.chat(text_q)
    rows, failed = [], 0
    for name, messages, want in checks:
        d = s.chat(messages, max_tokens=120)
        p = s.chat(messages, max_tokens=120, draft=False)
        low = d["text"].lower()
        missing = [alts[0] for alts in want if not any(w in low for w in alts)]
        ok = not missing and d["sha"] == p["sha"]
        failed += not ok
        rows.append({"check": name, "ok": ok, "prompt_tokens": d["prompt"], "drafted_equals_plain": d["sha"] == p["sha"],
                     "missing": missing, "ttft": d["ttft"], "seconds": d["seconds"], "reply": d["text"]})
        print(f"{'ok  ' if ok else 'FAIL'} {name:22} prompt {d['prompt']:>6}  drafted==plain {d['sha'] == p['sha']}  "
              f"first token {d['ttft'] or 0:5.2f} s  {d['text'][:100]!r}", flush=True)
    long_v = checks[3][1]
    alone = s.chat(long_v, max_tokens=120)
    got = {}
    jobs = [threading.Thread(target=lambda k=k, m=m: got.__setitem__(k, s.chat(m, max_tokens=120))) for k, m in (("v", long_v), ("t", text_q))]
    [j.start() for j in jobs]
    [j.join() for j in jobs]
    together = got["v"]["sha"] == alone["sha"] and got["t"]["sha"] == before["sha"]
    failed += not together
    print(f"{'ok  ' if together else 'FAIL'} a video and a text request at once each equal their replies alone", flush=True)
    after = s.chat(text_q)
    same = before["sha"] == after["sha"]
    failed += not same
    print(f"{'ok  ' if same else 'FAIL'} text unchanged by the video requests before it", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"url": a.url, "checks": rows, "concurrent_equal": together, "text_unchanged": same}, f, indent=1)
    print("PASS" if not failed else f"FAIL: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
