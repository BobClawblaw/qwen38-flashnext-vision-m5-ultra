"""8 prompts of ~120k tokens sent at once, each with its own needle: every answer names its needle."""
import json, threading, time, urllib.request
U = "http://127.0.0.1:8888/v1/chat/completions"
filler = "The archive room held shelves of ledgers, each page listing shipments, weights and dates in faded ink. " * 5600
res = {}
def one(i):
    code = f"ZX-{7000 + i * 37}"
    text = filler[: len(filler) // 2] + f" The vault code for team {i} is {code}. " + filler[len(filler) // 2:]
    b = {"model": "x", "messages": [{"role": "user", "content": text + f"\nWhat is the vault code for team {i}? Reply with the code only."}], "max_tokens": 16}
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        o = json.load(r)
    res[i] = (code, o["choices"][0]["message"]["content"].strip(), o["usage"]["prompt_tokens"], time.time() - t)
th = [threading.Thread(target=one, args=(i,)) for i in range(8)]
[t.start() for t in th]; [t.join() for t in th]
ok = 0
for i in range(8):
    code, ans, n, s = res[i]
    ok += code in ans
    print(f"{'ok  ' if code in ans else 'FAIL'} team {i}: {n} prompt tokens, answered in {s:.0f} s: {ans!r}")
print("PASS" if ok == 8 else f"FAIL: {8 - ok}")
