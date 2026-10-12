"""Sampled replies with top_k 0 (and wide top-k): their token shas and decode speed, alone."""
import json, sys, urllib.request
U = "http://127.0.0.1:8888/v1/chat/completions"
CASES = [("Describe the history of the bicycle.", {"temperature": 1.0, "seed": 2, "top_k": 0, "top_p": 1.0}),
         ("Write a poem about the sea.", {"temperature": 0.7, "seed": 5, "top_k": 0, "top_p": 0.95}),
         ("Explain how compilers work.", {"temperature": 1.0, "seed": 8, "top_k": 0, "top_p": 1.0, "min_p": 0.05}),
         ("List ten animals and a fact about each.", {"temperature": 0.8, "seed": 3, "top_k": 500, "top_p": 1.0}),
         ("Describe the history of the bicycle.", {"temperature": 0.7, "seed": 2})]
out = {}
for p, kw in CASES:
    b = {"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": 300, **kw}
    o = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900))
    rt = o.get("runtime") or o.get("tensorfold") or {}
    out[p + json.dumps(kw)] = rt.get("token_sha")
    print(f"{rt.get('tokens_per_second'):6.1f} tok/s  {rt.get('token_sha')}  {kw}", flush=True)
if len(sys.argv) > 2:
    base = json.load(open(sys.argv[2]))
    same = sum(out[k] == base[k] for k in out)
    print(f"{same}/{len(out)} equal to {sys.argv[2]}")
json.dump(out, open(sys.argv[1], "w"))
