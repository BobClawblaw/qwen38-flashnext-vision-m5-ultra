"""8 streams: greedy and sampled replies sent together each equal their solo runs; a mixed round; drafted == plain."""
import json, sys, threading, urllib.request
U = "http://127.0.0.1:8888/v1/chat/completions"
P = ["Write a short poem about the sea.", "Explain what a hash table is in three sentences.",
     "List five fruits and one fact about each.", "Write a Python function that reverses a linked list.",
     "Describe how a bicycle gear works.", "Give three tips for writing clear emails.",
     "Summarize the water cycle in four sentences.", "Write a haiku about autumn, then explain it."]
def chat(c, **kw):
    b = {"model": "x", "messages": [{"role": "user", "content": c}], "max_tokens": 160, **kw}
    with urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        o = json.load(r)
    rt = o.get("runtime") or o.get("tensorfold") or {}
    return rt.get("token_sha"), rt.get("tokens_per_second")
def together(kws):
    got = {}
    th = [threading.Thread(target=lambda i=i: got.__setitem__(i, chat(P[i], **kws[i])[0])) for i in range(8)]
    [t.start() for t in th]; [t.join() for t in th]
    return [got[i] for i in range(8)]
fails = 0
def check(name, ok, d=""):
    global fails; fails += not ok; print(f"{'ok  ' if ok else 'FAIL'} {name} {d}", flush=True)
samp = [{"temperature": 0.8, "seed": 5 + i} for i in range(8)]
g_alone = [chat(p)[0] for p in P]
s_alone = [chat(p, **samp[i])[0] for i, p in enumerate(P)]
for r in range(2):
    check(f"8 greedy together equal alone (rep {r+1})", together([{}] * 8) == g_alone)
    check(f"8 sampled together equal alone (rep {r+1})", together(samp) == s_alone)
mix = [samp[i] if i % 2 else {} for i in range(8)]
check("4 greedy + 4 sampled mixed equal alone", together(mix) == [s_alone[i] if i % 2 else g_alone[i] for i in range(8)])
plain = [chat(p, draft=False)[0] for p in P[:4]] + [chat(p, draft=False, **samp[i + 4])[0] for i, p in enumerate(P[4:])]
check("drafted equals draft:false (4 greedy, 4 sampled)", plain == g_alone[:4] + s_alone[4:])
print("PASS" if not fails else f"FAIL: {fails}")
