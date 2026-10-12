"""Two streams decode long replies while a long prompt fills: their token gaps, equality to their solo replies, the needle.

    python3 -I tools/stall_check.py TOKENS   (TOKENS: the long prompt's size, as tools/prompt_pass.py sizes it)
"""
import json, sys, threading, time, urllib.request
sys.path.insert(0, __import__("os").path.dirname(__file__))
from prompt_pass import one
URL = "http://127.0.0.1:8888/v1/chat/completions"
def stream(text, n, out):
    body = {"model": "x", "stream": True, "max_tokens": n, "temperature": 0, "messages": [{"role": "user", "content": text}], "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    stamps, chunks = [], []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            ev = json.loads(line[5:])
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece: stamps.append(time.monotonic()); chunks.append(piece)
    out["text"] = "".join(chunks); out["stamps"] = stamps
prompts = ["Write a long, detailed essay about the history of lighthouses, at least 1500 words.", "Write a long, detailed guide to growing tomatoes at home, at least 1500 words."]
solo = []
for p in prompts:
    o = {}; stream(p, 1500, o); solo.append(o["text"])
print("solo replies recorded", flush=True)
outs = [{}, {}]
ts = [threading.Thread(target=stream, args=(p, 1500, o)) for p, o in zip(prompts, outs)]
for t in ts: t.start()
time.sleep(3)
t0 = time.monotonic()
long = one("http://127.0.0.1:8888", "x", int(sys.argv[1]), seed=1000)
t1 = time.monotonic()
for t in ts: t.join()
print(f"long prompt: {long['prompt_tokens']} tokens, first token {long['ttft_s']} s, {long['prefill_tok_s']} tok/s, needle found {long['found']}")
for k, o in enumerate(outs):
    st = o["stamps"]; during = [b - a for a, b in zip(st, st[1:]) if t0 <= b <= t1]
    print(f"stream {k}: equal to solo {o['text'] == solo[k]}; {sum(1 for x in st if t0 <= x <= t1)} chunks arrived during the fill, largest gap {max(during) if during else 0:.2f} s, mean {sum(during)/len(during) if during else 0:.3f} s")
