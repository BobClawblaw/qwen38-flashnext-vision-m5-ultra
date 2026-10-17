"""Image input served end to end: drawn images the model must read, drafted against undrafted, long text around an
image, two images in one request, a follow-up turn, WebP, and a text reply unchanged by the image requests before it.

    python3 -I tools/vision_check.py --url http://127.0.0.1:8888 [--json OUT]

Stdlib only: the images are drawn here and written as PNG (zlib); the two WebP files in tools/images are Pillow 13.0.0's
encodings of two of them (shapes lossless, dots at quality 80), so nothing third-party runs in the check. Each
check asks for a short greedy reply and looks for the words the image forces (colors, shapes, positions, counts); every
drafted reply must equal its `"draft": false` twin byte for byte. Exit 0 when every check passes.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

RED, BLUE, GREEN, YELLOW, BLACK, BG = (220, 40, 40), (40, 70, 210), (40, 160, 70), (235, 200, 40), (20, 20, 20), (245, 240, 230)


def png(w: int, h: int, px) -> bytes:
    """An RGB image as PNG bytes; px(x, y) gives each pixel."""
    raw = b"".join(b"\x00" + bytes(c for x in range(w) for c in px(x, y)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def shapes(w=480, h=320):
    """A red circle on the left, a blue square on the right."""
    def px(x, y):
        if (x - w * 0.25) ** 2 + (y - h * 0.5) ** 2 < (h * 0.28) ** 2:
            return RED
        if w * 0.58 < x < w * 0.9 and h * 0.22 < y < h * 0.78:
            return BLUE
        return BG
    return png(w, h, px)


def stack(w=300, h=420):
    """A green triangle above a yellow circle."""
    def px(x, y):
        if h * 0.08 < y < h * 0.45 and abs(x - w / 2) < (y - h * 0.08) * 0.6:
            return GREEN
        if (x - w / 2) ** 2 + (y - h * 0.72) ** 2 < (w * 0.25) ** 2:
            return YELLOW
        return BG
    return png(w, h, px)


def dots(w=400, h=200, n=5):
    """`n` black dots in a row."""
    def px(x, y):
        for k in range(n):
            if (x - w * (k + 1) / (n + 1)) ** 2 + (y - h / 2) ** 2 < 18 ** 2:
                return BLACK
        return BG
    return png(w, h, px)


def part(data: bytes, media: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{media};base64," + base64.b64encode(data).decode()}}


def webp(name: str) -> dict:
    with open(f"{__file__.rsplit('/', 1)[0]}/images/{name}", "rb") as f:
        return part(f.read(), "image/webp")


class Server:
    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/v1/chat/completions"

    def chat(self, messages, max_tokens=80, draft=True):
        body = {"model": "default", "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if not draft:
            body["draft"] = False
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
        rt = out.get("runtime") or out.get("tensorfold") or {}
        return {"text": out["choices"][0]["message"]["content"], "prompt": out["usage"]["prompt_tokens"],
                "sha": rt.get("token_sha"), "tps": rt.get("tokens_per_second"), "ttft": rt.get("time_to_first_token"),
                "seconds": round(time.monotonic() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--json")
    a = ap.parse_args()
    s = Server(a.url)
    rng = random.Random(7)
    words = "the harbour pilot logged tides, cargo weights and the weather for every ship that entered the channel".split()
    filler = " ".join(rng.choice(words) for _ in range(9000))
    fact = "Note for later: the harbour master's cat is named Biscuit."
    q1 = [{"role": "user", "content": [part(stack()), {"type": "text", "text": "What shape is on top and what color is it? One sentence."}]}]
    checks = [
        ("shapes and sides", [{"role": "user", "content": [part(shapes()), {"type": "text", "text": "Which colors and shapes are in this image, and which is on the left? One sentence."}]}], [["red"], ["circle"], ["blue"], ["square"], ["left"]]),
        ("top and bottom", q1, [["triangle"], ["green"]]),
        ("counting", [{"role": "user", "content": [part(dots()), {"type": "text", "text": "How many black dots are there? Answer with a number."}]}], [["5", "five"]]),
        ("image then 9k words", [{"role": "user", "content": [part(shapes()), {"type": "text", "text": fact + "\n\n" + filler + "\n\nAnswer in one line each: 1) the shapes in the image at the start, 2) the cat's name."}]}], [["circle"], ["square"], ["biscuit"]]),
        ("9k words then image", [{"role": "user", "content": [{"type": "text", "text": fact + "\n\n" + filler}, part(stack()), {"type": "text", "text": "Answer in one line each: 1) the shape on top in the image just above, 2) the cat's name."}]}], [["triangle"], ["biscuit"]]),
        ("two images", [{"role": "user", "content": [part(shapes()), part(dots()), {"type": "text", "text": "One sentence each: the first image's shapes, then how many dots the second has."}]}], [["circle"], ["square"], ["5", "five"]]),
    ]
    ask_shapes = {"type": "text", "text": "Which colors and shapes are in this image, and which is on the left? One sentence."}
    checks.append(("lossless WebP", [{"role": "user", "content": [webp("shapes-lossless.webp"), ask_shapes]}], [["red"], ["circle"], ["blue"], ["square"], ["left"]]))
    checks.append(("lossy WebP counting", [{"role": "user", "content": [webp("dots-q80.webp"), {"type": "text", "text": "How many black dots are there? Answer with a number."}]}], [["5", "five"]]))
    first = s.chat(q1)
    checks.append(("follow-up turn", q1 + [{"role": "assistant", "content": first["text"]}, {"role": "user", "content": "And what is below it? Shape and color."}], [["circle"], ["yellow"]]))
    text_q = [{"role": "user", "content": "Write two sentences about lighthouses."}]
    before = s.chat(text_q)
    rows, failed = [], 0
    for name, messages, want in checks:
        d = s.chat(messages)
        p = s.chat(messages, draft=False)
        low = d["text"].lower()
        missing = [alts[0] for alts in want if not any(w in low for w in alts)]
        ok = not missing and d["sha"] == p["sha"]
        failed += not ok
        rows.append({"check": name, "ok": ok, "prompt_tokens": d["prompt"], "drafted_equals_plain": d["sha"] == p["sha"],
                     "missing": missing, "tps_drafted": d["tps"], "tps_plain": p["tps"], "ttft": d["ttft"], "reply": d["text"]})
        print(f"{'ok  ' if ok else 'FAIL'} {name:22} prompt {d['prompt']:>6}  drafted==plain {d['sha'] == p['sha']}  "
              f"{d['tps'] or 0:5.0f} tok/s  first token {d['ttft'] or 0:5.2f} s  {d['text'][:90]!r}", flush=True)
    by = {r["check"]: r for r in rows}
    twin = by["lossless WebP"]["reply"] == by["shapes and sides"]["reply"]  # the same pixels: the PNG's reply
    failed += not twin
    print(f"{'ok  ' if twin else 'FAIL'} the lossless WebP answers as its PNG", flush=True)
    after = s.chat(text_q)
    same = before["sha"] == after["sha"]
    failed += not same
    print(f"{'ok  ' if same else 'FAIL'} text unchanged by the image requests before it", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"url": a.url, "checks": rows, "webp_equals_png": twin, "text_unchanged": same}, f, indent=1)
    print("PASS" if not failed else f"FAIL: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
