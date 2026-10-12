"""Prompt passes at several lengths: time to the first token and the read rate, one request each, nothing cached.

    python3 tools/prompt_pass.py --url http://127.0.0.1:8888 --model NAME --sizes 4000 16000 64000 128000 250000 --json OUT

Each prompt is distinct text (a numbered run of sentences with a random seed in front) so the prompt cache resumes
none of it; a needle sentence sits near the start and the question at the end asks for it, so the reply shows the
whole prompt was read. The streamed reply's first content token stamps the time; prompt tokens come from the final
usage. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

WORDS = ("the engine reads", "every prompt chunk", "on the tensor units", "while the drafts", "wait for a round",
         "and the kept state", "resumes the next turn", "without reading it again", "the window holds", "what the Mac leaves")


def prompt_of(size: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    needle = f"The access code for run {seed} is {rng.randint(100000, 999999)}."
    lines = [f"Session {seed}.", needle]
    n = 2
    while n * 28 < size:  # about 28 tokens a note line on this tokenizer (measured: 13,028 tokens for 445 lines)
        lines.append(f"Note {n}: " + " ".join(rng.choice(WORDS) for _ in range(6)) + ".")
        n += 1
    text = "\n".join(lines)
    return text, needle


def one(url: str, model: str, size: int, seed: int) -> dict:
    text, needle = prompt_of(size, seed)
    body = {"model": model, "stream": True, "max_tokens": 24, "temperature": 0, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": text + "\n\nWhat is the access code for this run? Answer with the sentence that states it."}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    reply = ""
    usage = {}
    try:
        r = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        return {"asked": size, "prompt_tokens": 0, "ttft_s": 0, "prefill_tok_s": None, "found": False, "cached": 0,
                "reply": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"}
    with r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and first is None:
                    first = time.monotonic() - t0
                reply += piece
    tokens = usage.get("prompt_tokens", 0)
    code = needle.split()[-1].rstrip(".")
    return {"asked": size, "prompt_tokens": tokens, "ttft_s": round(first or 0, 2), "prefill_tok_s": round(tokens / first) if first else None,
            "found": code in reply, "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "reply": reply.strip()[:120]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4000, 16000, 64000, 128000, 250000])
    ap.add_argument("--json")
    a = ap.parse_args()
    out = []
    for i, size in enumerate(a.sizes):
        row = one(a.url, a.model, size, seed=1000 + i)
        out.append(row)
        print(f"{row['prompt_tokens']:>8} tokens  ttft {row['ttft_s']:7.2f} s  {row['prefill_tok_s'] or 0:>6} tok/s  "
              f"found={row['found']} cached={row['cached']}  {row['reply'][:60]!r}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"url": a.url, "model": a.model, "rows": out}, f, indent=1)
    return 0 if all(r["found"] for r in out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
