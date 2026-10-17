"""1M server: drafted == plain past the trained window, and a short request beside a long read (the 262k session)."""
import json, sys, threading, time, urllib.request
sys.path.insert(0, sys.argv[1] + "/tools")
from prompt_pass import prompt_of
URL, M = "http://127.0.0.1:8888", "Qwen3.8-Flash-Next-MLX-6bit"


def chat(content, n, **kw):
    body = {"model": M, "max_tokens": n, "temperature": 0, "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False}, **kw}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                                                 {"Content-Type": "application/json"}), timeout=3600).read())
    return r["choices"][0]["message"]["content"], r["usage"], round(time.time() - t, 1)


text, needle = prompt_of(290000, 42)
q = text + "\n\nWhat is the access code for this run? Then write three sentences about lighthouses."
d = chat(q, 96)
p = chat(q, 96, draft=False)
print("long prompt", d[1]["prompt_tokens"], "tokens; drafted == plain:", d[0] == p[0], "| found:", needle.split()[-1].rstrip(".") in d[0])
print(" drafted", d[2], "s; plain", p[2], "s;", repr(d[0][:90]))
short = "Write a haiku about the sea."
alone = chat(short, 48)
got = {}
th = threading.Thread(target=lambda: got.__setitem__("long", chat(prompt_of(600000, 43)[0] + "\n\nSummarise in one line.", 16)))
th.start()
time.sleep(20)
beside = chat(short, 48)
th.join()
print("short beside a 600k read: equal to alone:", beside[0] == alone[0], "| its time", beside[2], "s; long prompt", got["long"][1]["prompt_tokens"], "in", got["long"][2], "s")
