# Operations note: the LLM server on port 8888 of this Mac

For the agent that manages this Mac (Hermes) and for anyone else. Written 2026-10-11; video added the same day.

## What runs on :8888

- Qwen3.8-Flash-Next, 6-bit, with image and video input, two replies at once, each with a 262,144-token window
  (the trained window, no YaRN), and an 8 GiB prompt cache, served by TensorFold's native Metal engine from this recipe
  (`~/github/qwen38-flashnext-vision-m5-ultra`). These are the recipe's defaults since 2026-10-12; this Mac's
  `local.toml` only sets the host and port.
- The prompt cache is what makes an agent loop fast: each call resumes where the last one ended and reads only the
  new turns (a 42k-token conversation: first call ~11 s, later calls ~0.2 s plus ~0.3 s per 1,000 new tokens).
  Without it every call rereads the whole conversation.
- It supersedes the MTPLX app on :8888. MTPLX stays installed but must not be started: both want the port, and the
  two models cannot share the 256 GB at once. This server keeps about 189 GiB wired with its cache full, leaving
  ~41 GiB: an idle ComfyUI (8 GB) fits beside it, a second model does not. Past its budget `mac up` refuses.
- Two requests run at once; a third waits. Each holds up to 262,144 tokens; a longer prompt gets a 400.
- A 1M window is a setting (`context = 1048576`, static YaRN x4 on every prompt, 203 GiB wired), not the default.
- Endpoint: `http://<this Mac's LAN address>:8888/v1` or `http://127.0.0.1:8888/v1`. OpenAI-compatible chat, completions
  and Responses, plus Anthropic Messages. No API key: the LAN is trusted. Any model name is accepted; `/v1/models`
  lists `Qwen3.8-Flash-Next-MLX-6bit`.
- Sampling works: a request's `temperature` (with `top_p`, `top_k`, `min_p`, `seed`) is honored; a request without
  one is greedy. Without a `seed` the same request gives the same reply (the seed comes from the prompt). Four sampled
  streams run about 10% slower than greedy ones.
- Images go in `image_url` parts as data URLs or public https URLs (PNG, JPEG or WebP), up to 4 a request and 4,096
  image tokens in all. The server fetches URLs itself and refuses plain http and private addresses.
- Videos go in `video_url` parts as data URLs or public https URLs (MP4, MOV, WebM or MKV), up to 2 a request, 16 MiB each, sharing 16,384
  visual tokens. Frames are sampled at 2 a second (at most 256), so a 30 s 720p clip is about 8,000 tokens and 5 s to
  the first token; the model gets each pair of frames with its time.
- A very long prompt reads at a few thousand tokens a second (a 255k-token prompt takes about 70 s). Replies running
  beside it keep streaming: the long read and the replies share the GPU half and half until it is done.

## Managing it

It runs as the launchd agent `com.tensorfold.qwen38-vision`, which restarts it if it dies and starts it at login.

```bash
launchctl list | grep tensorfold                              # running: a pid in the first column
python3 -m mac restart      # (from the recipe folder) waits for requests in flight, then restarts (~40 s to ready)
launchctl kickstart -k gui/$(id -u)/com.tensorfold.qwen38-vision   # immediate restart: cuts off replies and tool calls
launchctl bootout gui/$(id -u)/com.tensorfold.qwen38-vision   # stop and unload (stays stopped until bootstrapped)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tensorfold.qwen38-vision.plist   # load and start again
```

From the recipe folder (`cd ~/github/qwen38-flashnext-vision-m5-ultra`):

```bash
python3 -m mac status          # pid, /health, /v1/models, /stats
python3 -m mac logs --tail 50  # the server log (~/.cache/tensorfold-qwen38-vision/server.log)
python3 -I tools/vision_check.py --url http://127.0.0.1:8888   # the image gate: PASS expected, about 2 minutes
python3 -I tools/video_check.py --url http://127.0.0.1:8888    # the video gate: PASS expected, about a minute
```

Do not run `python3 -m mac up` or `down` while the launchd agent is loaded: launchd owns the process and restarts it.
Boot the agent out first, then use `up`/`down` by hand if needed.

- Startup log: `~/.cache/tensorfold-qwen38-vision/launchd.log`. A start needs about 175 GiB free: close the other big
  process (MTPLX, another model) and `python3 -m mac restart`. `settings: ... holds about N GiB on the GPU` means `local.toml` asks for
  more than the GPU budget: lower `parallel`, `context` or `prompt_cache_gib`.
- Healthy start: `image tower loaded (1.73 GiB ...)` then `serving Qwen3.8-Flash-Next-MLX-6bit at http://0.0.0.0:8888`.

## Changing it

- Settings live in `recipe.toml` (defaults) and `local.toml` (this Mac: host 0.0.0.0, port 8888). After a change,
  `python3 -m mac restart`: it waits until no request is in flight, so no reply or tool call is cut off (a bare
  kickstart did that several times on 2026-10-12; `--timeout` and `--force` bound the wait).
- The engine is pinned by commit (`tensorfold_sha`, BobClawblaw/TensorFold branch `flashnext-shared-vision`, upstream PR #986) and built into
  `~/.cache/tensorfold-qwen38-vision/engine/<sha>`. To upgrade: set the new sha, `python3 -m mac build`, `python3 -m mac restart`,
  then run the image gate and `python3 -m mac ruler -- --runs 3` before trusting it.
- Text only (no tower, 1.7 GiB less): set `vision = false` in `local.toml` and `python3 -m mac restart`.
- Smaller footprint: `prompt_cache_gib = 0` in `local.toml` and `python3 -m mac restart` (~161 GiB, but agent loops reread
  everything). More at once: `parallel = 2` with `context = 131072` fits the budget; four full 262k streams do not.

## If something is wrong

- `curl -s localhost:8888/health` hangs or refuses: check `launchctl list | grep tensorfold` and the launchd log.
- Out of memory or the Mac stalls: something else large is loaded beside it. Never run another model of this size,
  or a heavy CPU job, while it serves.
- Video requests answer "libtfvideo" errors: the library is missing beside the binary; run `python3 -m mac build`
  and `python3 -m mac restart`.
- Replies to images or videos look wrong: run both gates. If they fail, roll `tensorfold_sha` back to `ddc17bb6`, the
  last build before sampling (`python3 -m mac build` rebuilds it; requests with a temperature then get a 400).
