# Operations note: the LLM server on port 8888 of this Mac

For the agent that manages this Mac (Hermes) and for anyone else. Written 2026-10-11.

## What runs on :8888

- Qwen3.8-Flash-Next, 6-bit, with image input, served by TensorFold's native Metal engine from this recipe
  (`~/github/qwen38-flashnext-vision-m5-ultra`).
- It supersedes the MTPLX app on :8888. MTPLX stays installed but must not be started: both want the port, and the
  two models cannot share the 256 GB at once. This model alone holds about 175 GiB once loaded.
- Endpoint: `http://192.168.5.178:8888/v1` (LAN) or `http://127.0.0.1:8888/v1`. OpenAI-compatible chat, completions
  and Responses, plus Anthropic Messages. No API key: the LAN is trusted. Any model name is accepted; `/v1/models`
  lists `Qwen3.8-Flash-Next-MLX-6bit`.
- Greedy only: leave `temperature` out or send 0. A request with another temperature gets a 400.
- One reply at a time: a second request waits for the first. Images go in `image_url` parts as data URLs (base64 PNG
  or JPEG), up to 4 a request and 4,096 image tokens in all.

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
```

Do not run `python3 -m mac up` or `down` while the launchd agent is loaded: launchd owns the process and restarts it.
Boot the agent out first, then use `up`/`down` by hand if needed.

- Startup log: `~/.cache/tensorfold-qwen38-vision/launchd.log`. A start refuses when free memory is under 157 GiB
  (the model's 149 GiB floor plus 8): close the other big process (ComfyUI, MTPLX, another model) and kickstart.
- Healthy start: `image tower loaded (1.73 GiB ...)` then `serving Qwen3.8-Flash-Next-MLX-6bit at http://0.0.0.0:8888`.

## Changing it

- Settings live in `recipe.toml` (defaults) and `local.toml` (this Mac: host 0.0.0.0, port 8888). After a change,
  `launchctl kickstart -k gui/$(id -u)/com.tensorfold.qwen38-vision`.
- The engine is pinned by commit (`tensorfold_sha`, BobClawblaw/TensorFold branch `flashnext-vision`) and built into
  `~/.cache/tensorfold-qwen38-vision/engine/<sha>`. To upgrade: set the new sha, `python3 -m mac build`, kickstart,
  then run the image gate and `python3 -m mac ruler -- --runs 3` before trusting it.
- Text only (no tower, 1.7 GiB less): set `vision = false` in `local.toml` and kickstart.

## If something is wrong

- `curl -s localhost:8888/health` hangs or refuses: check `launchctl list | grep tensorfold` and the launchd log.
- Out of memory or the Mac stalls: something else large is loaded beside it. Never run another model of this size,
  or a heavy CPU job, while it serves.
- Replies to images look wrong: run the image gate. If it fails, roll `tensorfold_sha` back to `b103ecc2`, the commit
  this note was written against.
