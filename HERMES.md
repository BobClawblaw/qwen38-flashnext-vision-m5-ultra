# Operations note: the LLM server on port 8888 of this Mac

For the agent that manages this Mac (Hermes) and for anyone else. Written 2026-10-11; video added the same day.

## What runs on :8888

- Qwen3.8-Flash-Next, 6-bit, with image and video input, four streams at once and a 1,048,576-token window, served by
  TensorFold's native Metal engine from this recipe (`~/github/qwen38-flashnext-vision-m5-ultra`).
- It supersedes the MTPLX app on :8888. MTPLX stays installed but must not be started: both want the port, and the
  two models cannot share the 256 GB at once. This server holds about 222 GiB once loaded (its load limit is 240 GB,
  `load_limit_gb`), so nothing else large fits beside it: no ComfyUI, no second model, no heavy CPU jobs.
- Four requests run at once; a fifth waits. Only one stream (slot 0) holds the full 1M window; the other three hold
  262,144 tokens. A request longer than 262,144 tokens waits for slot 0 if it is busy. Short requests go to the small
  slots first, so slot 0 stays free for long ones.
- Past 262,144 tokens the engine rotates with static YaRN x4, and it does so for every prompt: short prompts score a
  little lower than on the text-only recipe (tool calls 56/60 against 59/60). For the best short-prompt quality,
  set `context = 262144` in `local.toml` and kickstart (four streams, no YaRN).
- Endpoint: `http://192.168.5.178:8888/v1` (LAN) or `http://127.0.0.1:8888/v1`. OpenAI-compatible chat, completions
  and Responses, plus Anthropic Messages. No API key: the LAN is trusted. Any model name is accepted; `/v1/models`
  lists `Qwen3.8-Flash-Next-MLX-6bit`.
- Greedy only: leave `temperature` out or send 0. A request with another temperature gets a 400 saying it is not
  supported; Hermes retries without it. Hermes's own `auxiliary.vision.temperature` is set to 0 in
  `~/.hermes/config.yaml` so its image tool never sends one.
- Images go in `image_url` parts as data URLs (base64 PNG or JPEG), up to 4 a request and 4,096 image tokens in all.
- Videos go in `video_url` parts as data URLs (MP4, MOV, WebM or MKV), up to 2 a request, 16 MiB each, sharing 16,384
  visual tokens. Frames are sampled at 2 a second (at most 256), so a 30 s 720p clip is about 8,000 tokens and 5 s to
  the first token; the model gets each pair of frames with its time.
- A very long prompt reads at a few thousand tokens a second (a 1M-token prompt takes minutes). Replies running
  beside it keep streaming: the long read and the replies share the GPU half and half until it is done.

## Managing it

It runs as the launchd agent `com.tensorfold.qwen38-vision`, which restarts it if it dies and starts it at login.

```bash
launchctl list | grep tensorfold                              # running: a pid in the first column
launchctl kickstart -k gui/$(id -u)/com.tensorfold.qwen38-vision   # restart (about 40 s to ready)
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

- Startup log: `~/.cache/tensorfold-qwen38-vision/launchd.log`. A start needs about 230 GiB free: close the other big
  process (ComfyUI, MTPLX, another model) and kickstart. `OverMemoryLimit` in the log means the four slots did not fit
  under `load_limit_gb`: free memory, or lower `parallel` or `context` in `local.toml`.
- Healthy start: `image tower loaded (1.73 GiB ...)` then `serving Qwen3.8-Flash-Next-MLX-6bit at http://0.0.0.0:8888`.

## Changing it

- Settings live in `recipe.toml` (defaults) and `local.toml` (this Mac: host 0.0.0.0, port 8888). After a change,
  `launchctl kickstart -k gui/$(id -u)/com.tensorfold.qwen38-vision`.
- The engine is pinned by commit (`tensorfold_sha`, BobClawblaw/TensorFold branch `flashnext-video`) and built into
  `~/.cache/tensorfold-qwen38-vision/engine/<sha>`. To upgrade: set the new sha, `python3 -m mac build`, kickstart,
  then run the image gate and `python3 -m mac ruler -- --runs 3` before trusting it.
- Text only (no tower, 1.7 GiB less): set `vision = false` in `local.toml` and kickstart.
- Smaller footprints, each a `local.toml` line and a kickstart: `context = 262144` (four streams, no YaRN, ~205 GiB),
  `parallel = 1` (one reply at a time, the single-reply driver, ~200 GiB at 1M).

## If something is wrong

- `curl -s localhost:8888/health` hangs or refuses: check `launchctl list | grep tensorfold` and the launchd log.
- Out of memory or the Mac stalls: something else large is loaded beside it. Never run another model of this size,
  or a heavy CPU job, while it serves.
- Video requests answer "libtfvideo" errors: the library is missing beside the binary; run `python3 -m mac build`
  and kickstart.
- Replies to images or videos look wrong: run both gates. If they fail, roll `tensorfold_sha` back to `ddc17bb6`, the
  commit this note was last checked against (`fdbd0228` the same with the old refusal wording) (`4cef40eb` is the same without video) (`b103ecc2` is the earlier single-stream vision build: set `parallel = 1` and
  `context = 262144` with it).
