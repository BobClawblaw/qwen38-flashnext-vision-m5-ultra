# Changelog

## 2026-10-17 (main's engine)

- Engine `8a509ab5` (BobClawblaw/TensorFold `flashnext-shared-vision`, upstream PR #986 stacked on #606): TensorFold
  main with cwschroeder's shared sessions and our images and video on the Qwen-VL tower TensorFold's 27B uses, in place of
  `7808bee8` (v1.0.7 with its own tower). The recipe harness's speed plan the same night: one stream 138/206/169/211
  tok/s and 0.11-0.12 s to the first token (the previous engine 138/203/170/207, 0.11-0.12 s), two streams 195/262/221/270
  in all (196/253/221/269), drafts 155/84 tok/s (154/83). Image gate 8/8, video gate 7/7 (the previous engine 6/7),
  180.4 GiB wired.
- Image and video chats resume from the prompt cache: a follow-up with the same images resumes past them, replies equal
  to the cache off's.
- `mac build` builds libjpeg-turbo 3.1.4.1 (its release tarball, SHA-256 pinned) into the cache and puts libturbojpeg
  beside the server: the engine reads JPEG with that release only. WebP is refused.
- No 1M window: the engine has no YaRN, so `mac validate` refuses `context` past 262,144. `extra_env` drops
  `FLASHNEXT_RESIDENCY=all` (the engine holds its buffers itself).
- The quality suite was not rerun on this engine.
- README: constrained output (`response_format`, `guided_*`, grammars) is listed as not supported; the engine answers it
  with a 400, as the previous one did (the recipe harness's structured check, 0/18 on both).

## 2026-10-12 (graceful restart)

- `python3 -m mac restart` waits until /health shows no request running or waiting (two readings in a row), then
  kickstarts the launchd unit (or downs and ups) and waits for ready. Every broken tool call Hermes logged against
  8888 on 2026-10-12 came from a bare kickstart mid-stream. Live: a 1,844-token reply finished (finish=stop) before
  the restart began. The tool-call suite on the current defaults: 59/60, three runs, thinking on and off.

## 2026-10-12 (two streams)

- Defaults: `parallel = 2`, `prompt_cache_gib = 8`, `FLASHNEXT_NO_LONE=1`. Two 262k streams with every buffer held:
  0 of 90 mixed replies slow, ~189 GiB wired, ~41 GiB free, idle first token 0.15 s, two 99k-token prompts beside
  two chats. The lone driver ran 3 of 60 replies at half speed (identical rounds); the shared rounds match its speed.
  Four 262k streams (5 GiB cache) worked but at 202-209 GiB wired, 25 GiB free, 2 of 30 slow.

## 2026-10-12 (a residency set kept whole)

- Engine `7808bee8` with `FLASHNEXT_RESIDENCY=all`: every live Metal buffer in the residency set, buffers made or
  freed during a request committed by the idle keepalive between requests. macOS defaults (the wired collector on):
  60 of 60 mixed replies steady, the first token 0.09-0.10 s after 10 s to 5 min idle, a 25k-token prompt after a
  minute idle 5.4 s. `keep_warm` is a day. The collector setting and its LaunchDaemon are no longer needed.

## 2026-10-12 (keep it wired)

- `iogpu.disable_wired_collector=1` keeps the engine's memory wired with no residency set: 30 of 30 mixed replies
  steady, the first token after 10-30 s idle 0.10-0.12 s (2.1-2.4 s with the collector on), a 25k-token prompt after
  60 s idle 5.7 s (24 s). `launchd/local.iogpu-keep-wired.plist` sets it at boot; `mac validate` and `up` report it.

## 2026-10-12 (no residency set)

- `extra_env = "FLASHNEXT_NO_RESIDENCY=1"` by default. With the engine and tower held (172.7 GiB) 10-30% of replies
  decoded at 50-100 tok/s with identical rounds, including a rerun of the configuration that measured 10/10 steady at
  14:00; with weights only every reply ran ~45% slower; holding each buffer as made stalled new prompts at 5 tok/s.
  Without a set: 30 of 30 replies at full speed, the first token after idle ~2 s, a 25k-token prompt after a minute
  idle 24 s (5.9 s with the set). README: the residency table.

## 2026-10-12 (one stream, a GPU budget)

- Defaults: `parallel = 1` and `prompt_cache_gib = 6` (were 4 and 24). Four 262k streams hold about 184 GiB on the GPU
  before any cache; with the cache and an idle ComfyUI (8 GB) the GPU passed ~190 GiB, where macOS (with
  `iogpu.wired_limit_mb` unset) reclaimed the server's memory: decode 30-207 tok/s on one request, `GpuFailed` on
  cold prompts, a stuttering Mac. One stream: 173 GB on the GPU, 0.1 s first token after idle, most replies 201-208 /
  137 tok/s; some still run at 50-90 with identical rounds (cause not found yet).
- `mac validate` and `mac up` estimate the server's GPU memory and refuse settings past 74% of RAM (or
  `iogpu.wired_limit_mb` when set) less the new `gpu_headroom_gib` (12) for other apps. README: GPU memory section.

## 2026-10-12 (top_k 0 endpoint)

- Engine `c507c94f`: the wide top-k draw keeps a uniform that rounds to exactly 1.0 (+inf noise) as the reference
  does; the early stop could pass it over (found in review on cwschroeder/TensorFold#7). Same five top_k 0 shas.

## 2026-10-12 (weights pinned again)

- Engine `431b7643` with `FLASHNEXT_RESIDENCY_SESSIONS=0` (`extra_env`, replacing `FLASHNEXT_NO_RESIDENCY=1`): the
  residency set holds the weights, the engine's buffers and the tower (172.7 GiB), not the three extra 262k sessions.
  Without any set, idle macOS paged the mapped weights out (the server's resident memory fell to 28 GB): Hermes's
  first 30k-token prompt after 30 min idle ran 46 s into the GPU watchdog (`GpuFailed`) twice, and the Mac stuttered
  while the weights paged back. Now: decode 206-207 / 136-137 tok/s on every request (the full 196 GiB set gave
  30-207), the first token after 10-30 s idle 0.15 s (was ~2 s), a fresh 25k-token prompt after a minute idle 5.9 s.
  `local.toml` here also sets `prompt_cache_gib = 16`.

## 2026-10-12 (top_k 0)

- Engine `eb219d2a` (`add3da3` + one commit, also on #719 and cwschroeder/TensorFold#7): an unlimited or wide top-k
  draws by a counting sort of the bf16 keys instead of a full sort, the same tokens. Sampled with `top_k: 0`:
  23-30 -> 91-119 tok/s; default sampling unchanged. Evidence: `evidence/2026-10-12-topk0/`.

## 2026-10-12 (lone driver and priced rounds)

- Engine `add3da3` (sethforprivacy's commit on our `flashnext-video`, #719 and the #606 sampling; branch
  `flashnext-lane-parity`): a lone text request runs on the single-reply driver's GPU-side rounds and is handed to
  the shared rounds mid-reply when a second one arrives; shared rounds are priced from forwards timed at load.
  Ruler, 1 stream: 142 / 203 / 169 / 192 tok/s (prose / code / structured / list) against `160fed1c`'s
  128 / 175 / 150 / 177; 4 streams 245 / 325 / 277 / 327 in all against 209 / 300 / 250 / 319.
- Served with `FLASHNEXT_NO_RESIDENCY=1` (`extra_env`): `add3da3`'s residency set holds 196 GiB at four 262k
  sessions, and with it the GPU-side rounds ran 30-207 tok/s on the same request (rounds unchanged, the decode
  thread waiting on the GPU); 179 GiB at 65k and 173 GiB at 16k ran full speed. Without the set the first token
  after 10-30 s idle is ~2 s, as on `160fed1c`. Reported on upstream #606.
- Gates: greedy replies equal `160fed1c`'s; lone sampled replies equal four streams' (12/12); a reply handed over
  mid-reply equals its solo run (5/5, greedy and sampled); eight requests on four streams equal solo; images 7/7;
  prompt cache resumes (0.17-0.21 s later turns). Video 6/7: the 30 s clip's reply keeps the order but drops "then"
  (`add3da3` places image rows in a standard frame, so image and video replies may word differently). Evidence:
  `evidence/2026-10-12-lane-parity/`.

## 2026-10-12 (pipelined sampling at parallel = 1)

- Engine `160fed1c`: `parallel = 1` sampled rounds run two deep, the accept kernel waiting on the host's draws through
  a shared event. Sampled 0.7 decode 170.7 -> 173.9 tok/s (code) and 135.9 -> 139.8 (chat); greedy unchanged; all 12
  sampled replies equal `339b759a`'s. The four-stream path is unchanged.

## 2026-10-12 (eight streams)

- Measured `parallel = 8`: refused at 262k (past the 239 GB load limit), served at `context = 131072` with the 24 GiB
  cache. 272-339 tok/s in all against four streams' 210-317, 34-42 tok/s each; every reply equals its solo run,
  greedy and sampled. Four stay the default. Evidence: `evidence/2026-10-12-parallel8/`.

## 2026-10-12 (sampling at parallel = 1)

- Engine `339b759a`: the single-reply driver (`parallel = 1`) samples too, with the same host sampler and positions,
  so a seed gives the same reply at one stream or four (12 of 12 replies across three sampling settings). Its sampled
  rounds run one at a time (115 tok/s against 135 greedy); greedy rounds keep the GPU pipeline and are unchanged.
  Speed-up mode (two Macs) stays greedy only.

## 2026-10-12 (sampling)

- Engine `925ee05f` (branch `flashnext-sampling`): requests with a temperature sample instead of getting a 400. Each
  token is drawn on the host from the head's logits with the lane core's keyed sampler; MTP drafts stay on and a
  sampled reply is the same drafted or not, alone or beside other streams; greedy replies are byte-identical to
  `ddc17bb6`'s. One stream sampled at 0.7: 126-132 tok/s; four: ~163 tok/s in all (180 greedy). `parallel = 1` stays
  greedy only. Image gate 7/7, video gate 7/7.

## 2026-10-12 (defaults)

- The defaults are now `context = 262144` (the trained window, plain rotary) with a 24 GiB prompt cache past the 70%
  share (`prompt_cache_gib = 24`, new setting `prompt_cache_over_cap = true`), four streams kept: ~219 GiB in all.
  The 1M window stays a setting (`context = 1048576`, `prompt_cache_gib = 0`; validate refuses it with the cache on).
  Why, in the README's new "The 1M window and its problems": no memory left for the prompt cache (every agent call
  rereads its history), YaRN on every prompt, one long stream at a time, minutes-long reads.

## 2026-10-12 (this Mac's settings)

- This Mac serves `context = 262144` (no YaRN) with four streams and a 24 GiB prompt cache past the 70% share
  (`prompt_cache_gib = 24`, `extra_args = "--prompt-cache-over-cap"`, ~219 GiB in all). At 1M the slots left no room
  for the cache, so every agent call reread its whole conversation (Hermes: ~20 s a call at 59k tokens). Now a call
  resumes where the last ended: 42k-token turns at 0.2 s to the first token, 4k-token tool results at 1.3 s. Ruler,
  gates and a 255,674-token needle in [`evidence/2026-10-12-262k-cache/`](evidence/2026-10-12-262k-cache/).

## 2026-10-12 (latest)

- Engine `ddc17bb6`: a request with a temperature above 0 is refused as "temperature above 0 is not supported: ...",
  so OpenAI clients that retry without an unsupported parameter (Hermes's image tool among them) drop it and succeed.
  The old wording matched no client's retry, and Hermes's `vision_analyze` (temperature 0.1) failed on every image.

## 2026-10-12 (later)

- Video input (engine `fdbd0228`, TensorFold branch `flashnext-video`: upstream #618's video path merged onto
  `flashnext-vision-parallel`): `video_url` parts in MP4, MOV, WebM or MKV (H.264, HEVC, VP8, VP9, MPEG-4), sampled,
  sized and timed as transformers' Qwen3-VL video processor does at the checkpoint's bounds, each pair of frames one
  image to the tower, each written as `<X.Y seconds>` then its rows. `/tokenize` equals transformers' `input_ids` on 12
  clips; `tools/video_check.py` 7/7 (direction, color, position, two scenes in order over 30 s of 720p, a video beside
  an image, concurrent equal to alone); image gate 7/7. `video` and `vision_video_tokens` settings; `mac build` builds
  a static FFmpeg 9.0.2 and libtfvideo once.

## 2026-10-12

- Long prompts no longer stall the other streams (engine `4cef40eb`): the Flash Next backend implements the lane core's
  `prefill_step`, reading a prompt in steps between the other streams' rounds (1,024 rows a step beside decoding
  streams, each step waiting until their rounds have had as long; full chunks alone; a short prompt's single step at
  once). Two streams decoding beside a 305k-token fill got a chunk every 79-90 ms (they froze for the whole pass
  before), byte-equal to their solo replies; the fill read at 2,693 tok/s. `tools/stall_check.py`.

## 2026-10-11 (later)

- Four streams and a 1,048,576-token window by default, on TensorFold's `flashnext-vision-parallel` branch (`68306275`):
  upstream PR #606's shared lane path with images in each stream's prefill, the load limit offered on #606, static
  YaRN x4 past the trained window in every rotary kernel, slot 0 holding the full window. Ruler 131-184 tok/s one user,
  211-318 aggregate at four; needles found at 305k, 440k, 553k and 982k tokens; image gate 7/7; three images and a
  text request at once equal their replies alone. Under YaRN: tools 56/60, IFEval 132/124, GSM8K 241/250.
  `parallel`, `load_limit_gb` settings; `context` up to 1,048,576; `tools/prompt_pass.py`.

## 2026-10-11

- First version: Qwen3.8-Flash-Next 6-bit with image input on one M5 Ultra, on the `flashnext-vision` branch of
  TensorFold (v1.0.7 plus the tower, image preparation and multimodal positions on Metal, commit `b103ecc2`). Image
  gate (`tools/vision_check.py`) passes 7/7 with drafted replies byte-equal to undrafted; text decode and the suite's
  core checks equal the text-only recipe's. Served on :8888 to the LAN without an API key, superseding MTPLX on this
  Mac, by the launchd unit; HERMES.md is the operations note. The launchd unit's PATH now holds /usr/sbin (`lsof`).
