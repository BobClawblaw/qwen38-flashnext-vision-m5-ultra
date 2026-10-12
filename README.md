# Qwen3.8-Flash-Next with image and video input and four streams on one M5 Ultra

Serve [TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP)
on one Apple M5 Ultra (256 GB) through [TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine:
text, images and video in one OpenAI-compatible API (`image_url` and `video_url` parts), four replies at once sharing
each forward, the trained 262,144-token window with a 24 GiB prompt cache (an agent's next call reads only its new
turns), MTP drafts, a launchd unit. A 1,048,576-token window is a setting away, with costs listed
[below](#the-1m-window-and-its-problems).

Upstream TensorFold v1.0.7 serves Flash Next on Metal one reply at a time, up to 262,144 tokens, text only. This recipe
builds the [`flashnext-sampling`](https://github.com/BobClawblaw/TensorFold/tree/flashnext-sampling)
branch, v1.0.7 plus six things:

- **Image input on Metal** (`zig/src/families/flashnext/vision.zig`, `image.zig`, `vis_prompt.zig`): transformers'
  `Qwen4ExpVisionModel` in bf16 on the GPU (within bf16 noise of it on eight images), the processor's preparation
  (ImageIO, EXIF orientation, alpha over white, `smart_resize`, PIL's bicubic), and the tower's rows over the image
  tokens with transformers' multimodal positions, in a frame shifted so text after the last image sits at its slot:
  decode, drafts and shared rounds are untouched, and text replies are byte-identical to v1.0.7's.
- **Four streams** from upstream [#606](https://github.com/ashhart/TensorFold/pull/606) (cwschroeder's shared lane path),
  with the load limit offered there (`FLASHNEXT_LOAD_LIMIT_GB`) and images in each stream's prefill.
- **An optional 1M window** (not the default): static YaRN x4 past the trained 262,144 (transformers' table, unit-tested) in every rotary kernel,
  the engine's cap a runtime value (a 262k server allocates as before), slot 0 holding the full window and slots 1-3
  the trained one, long requests waiting for slot 0, and prompt chunks shortened past 262k so the indexer's scores fit.
- **Prompt passes that don't stall the others**: a stream's prompt is read in steps between the other streams' rounds
  (the lane core's `prefill_step`), 1,024 rows a step while others decode and only once their rounds have had as long
  as the step before, so a long read and the replies around it share the GPU half and half; alone it reads at full
  speed, and a short prompt's single step never waits.
  At the default `context = 262144` the kernels take the plain rotary, bit-identical to v1.0.7.
- **Video input** (`zig/src/families/flashnext/video.zig`, upstream [#618](https://github.com/ashhart/TensorFold/pull/618)'s
  libtfvideo over FFmpeg 9.0.2): frames sampled at 2 a second (4 to 256), sized over the clip's pixel budget and
  paired as transformers' Qwen3-VL video processor does, each pair an image to the tower, written as
  `<X.Y seconds>` then its rows. The server's `/tokenize` equals transformers' `input_ids` on 12 clips across codecs
  and containers ([`transformers_prompt_ids.txt`](evidence/2026-10-11-video/transformers_prompt_ids.txt)).

- **Sampling** (`zig/src/families/flashnext/sample.zig`): a request with a temperature draws each token on the host
  from the head's logits with the server's keyed sampler (top-k, top-p, min-p, a `seed` or one derived from the
  prompt). MTP drafts still help: a draft is kept when it equals the drawn token, so a sampled reply is the same
  drafted and undrafted, and alone or beside other streams. Greedy replies are unchanged.

Nothing to convert: the 6-bit checkpoint carries the tower (`vision_tower.*`, bf16, 1.7 GiB loaded with scratch).

## Measured on this Mac

Mac Studio M5 Ultra, 256 GB, checkpoint `45032a19`, greedy, thinking off, served by the launchd unit.

**The defaults** (four streams, 262,144 tokens, plain rotary, 24 GiB prompt cache; engine `ddc17bb6`, 2026-10-12,
[`evidence/2026-10-12-262k-cache/`](evidence/2026-10-12-262k-cache/)):

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 129.3 | 176.6 | 149.8 | 177.4 | 0.19-0.20 s |
| 4, each | 52.6 | 78.3 | 60.7 | 78.4 | 0.39-0.41 s |
| 4, aggregate | 209.5 | 309.6 | 245.9 | 321.5 | |

Decode is `bench/ruler.py` (256-token replies, median of 3). Sampling (temperature 0.7, engine `925ee05f`,
[`sampling_check.txt`](evidence/2026-10-12-sampling/sampling_check.txt)): one stream 126-132 tok/s drafted against
92-96 undrafted, four streams about 163 tok/s in all against 180 greedy (the drawn tokens match the greedy drafts
less often); every sampled reply equal drafted and undrafted, and alone and beside three others. With
`parallel = 1` (engine `339b759a`) the same seed gives the same reply as with four streams (12 of 12); engine
`160fed1c` runs its sampled rounds two deep as greedy ones are, 2-3% faster with the same replies. A 255,674-token prompt read at 3,635 tok/s (70 s) and
108,194 at 3,870 tok/s, a needle found in each. An agent loop over a 42k-token conversation: the first call 10-11 s,
each later call 0.2 s to the first token, and 1.3 s when a turn adds a 4,000-token tool result (the cache resumes at
the previous call's end). Image gate 7/7, video gate 7/7. About 219 GiB in use with the cache full.

**The 1M window** (`context = 1048576`, YaRN x4, no prompt cache; engine `68306275`, 2026-10-11,
[`evidence/2026-10-11-1m-parallel4/`](evidence/2026-10-11-1m-parallel4/)):

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 131.3 | 178.9 | 160.2 | 183.9 | 0.12-0.16 s |
| 4, each | 52.0 | 77.0 | 64.7 | 78.8 | 0.37-0.42 s |
| 4, aggregate | 211.0 | 311.5 | 259.8 | 317.6 | |

Long prompts, one request each, a needle near the start found in every reply
([`prompt_pass_long.json`](evidence/2026-10-11-1m-parallel4/prompt_pass_long.json),
[`prompt_pass_1m.json`](evidence/2026-10-11-1m-parallel4/prompt_pass_1m.json)):

| Prompt | First token | Rate |
|---:|---:|---:|
| 304,917 tokens | 86 s | 3,541 tok/s |
| 440,357 | 129 s | 3,406 tok/s |
| 553,208 | 172 s | 3,210 tok/s |
| 982,016 | 395 s | 2,488 tok/s |

Images ([`vision_check.txt`](evidence/2026-10-11-1m-parallel4/vision_check.txt), images drawn by the check itself,
every drafted reply byte-equal to `"draft": false`): shapes and sides, top and bottom, counting five dots, an image then
9k words with a fact ("A red circle and a blue square... Biscuit"), 9k words then an image, two images in one request,
a follow-up turn: 7/7, small images at 0.15-0.25 s to the first token, 10k-token image prompts at 2.5-3.6 s. Three image
requests and a text request at once each equal their replies alone, twice
([`concurrent_images.txt`](evidence/2026-10-11-1m-parallel4/concurrent_images.txt)).

A long prompt beside decoding streams (`tools/stall_check.py`, two streams writing 1,500-token replies while a
305k-token prompt fills, [`stall_check.txt`](evidence/2026-10-11-prefill-steps/stall_check.txt)): the streams kept
streaming, a chunk every 79-90 ms on average (the longest gap 1.6 s), their replies byte-equal to their solo runs; the
long prompt read at 2,693 tok/s (3,643 alone) and found its needle. Before this, the streams froze for the whole pass.

Quality, the Spark recipe's suite, thinking off, at 1M with static YaRN x4 on every prompt
([`suite/`](evidence/2026-10-11-1m-parallel4/suite/report.md)), beside the same checks on the trained window's plain
rotary (the text-only sibling, [qwen38-flashnext-mlx6bit-m5-ultra](https://github.com/BobClawblaw/qwen38-flashnext-mlx6bit-m5-ultra)):

| Check | 1M, YaRN x4 | 262k, plain |
|---|---:|---:|
| drafted equals undrafted | 12/12 | 12/12 |
| four streams together equal each alone | 12/12 | 12/12 |
| streaming, multi-turn, stop, thinking, copy, max_tokens | all pass | all pass |
| tool calls | 56/60 | 59/60 |
| JSON mode | 30/30 | 30/30 |
| IFEval loose / strict | 132 / 124 | 134 / 128 |
| GSM8K | 241/250 | 242/250 |

Video ([`video_check.txt`](evidence/2026-10-11-video/video_check.txt), `tools/video_check.py`, clips in
`tools/clips`, every drafted reply byte-equal to `"draft": false`): a yellow square going right in H.264 and VP9, a red
square going left along the bottom, a 30-second 720p clip's two scenes in order (8,206 prompt tokens, first token
5.0 s), a video beside an image, and a video and a text request at once each equal to their replies alone: 7/7.

## The 1M window and its problems

`context = 1048576` in `local.toml` (and `prompt_cache_gib = 0`, see the first point) serves up to 1,048,576 tokens.
It works, with a needle found at 982k, but it was the default only from 2026-10-11 to 2026-10-12 and these are why it
is not:

- **No prompt cache.** Slot 0's 1M caches take the memory the prompt cache would use: four streams at 1M hold about
  222 GiB, and the default 24 GiB cache on top would reach about 246 GiB of 256 (the engine's override does not stop
  it, so `mac validate` refuses a cache with four streams past 262,144). With no cache every request rereads its
  whole prompt, so an agent loop pays for its full history on every call: Hermes's calls took about 20 s each at 59k
  tokens, against 0.2-1.3 s with the cache.
- **YaRN on every prompt.** The engine's rotary is static YaRN x4 for every request, short ones too, and short-prompt
  quality dips: tool calls 56/60 against 59/60, IFEval 132/124 against 134/128, GSM8K 241 against 242 (the table above).
- **One long stream.** Only slot 0 holds more than 262,144 tokens; a second long request waits for it, and the other
  three streams stay at 262,144.
- **Slow long reads.** A 982k prompt takes 395 s to its first token (2,488 tok/s), and the read shares the GPU with
  every running reply while it lasts.
- **One unexplained failure.** A 553k prompt once ended in a GPU timeout while other jobs held memory; reruns passed.
- **Image blocks past 2,048 tokens** are scored by the sparse indexer at slot positions (selection only; at any
  window).

Without the cache's 24 GiB the 262k default needs about 195 GiB; with `parallel = 1` about 176 GiB.

**Eight streams** ([`evidence/2026-10-12-parallel8/`](evidence/2026-10-12-parallel8/)) fit only at `context = 131072`:
eight 262k sessions pass the 239 GB load limit and the server refuses to start. With `parallel = 8`, `context = 131072`
and the 24 GiB cache the server holds 199 GiB once loaded plus the cache, about what four 262k streams take. Ruler,
tok/s each / in all: prose 34 / 272, code 41 / 330, structured 37 / 293, list 42 / 339, first token 0.8 s (four streams
on the same server: 53 / 210 to 78 / 317; one: 128 to 175). Eight greedy and eight sampled replies sent together each
equal their solo runs (twice), as do four of each mixed and drafted against `"draft": false`. Eight 118k-token
prompts sent at once all found their needle, the last first token after 236 s (prompt passes run one stream at a
time). Image gate 7/7. Free memory never went under 30 GiB. Eight streams add about 7% to four's total speed and halve
each stream's, so four stay the default.

## Requirements

- One Apple Silicon Mac with 256 GB: the defaults hold about 219 GiB once the prompt cache fills (`load_limit_gb = 240`
  and `prompt_cache_over_cap` let them past the engine's 70% share). Smaller Macs: `prompt_cache_gib = 0` (~195 GiB),
  and `parallel = 1` with it (~176 GiB).
- macOS with Xcode and its Metal toolchain, Zig 0.17.0, Python 3.11+ (stdlib only), the `hf` CLI, and for video
  `curl` and `make` (FFmpeg's source is fetched and checked by SHA-256). Disk: 147 GiB.

## Quick start

```bash
git clone https://github.com/BobClawblaw/qwen38-flashnext-vision-m5-ultra.git && cd qwen38-flashnext-vision-m5-ultra
cp local.example.toml local.toml && $EDITOR local.toml   # host, port, paths if not the defaults
python3 -m mac validate            # the settings and the guards, touches nothing
python3 -m mac up                  # builds the engine once, finds or downloads the weights, waits for the API
python3 -I tools/vision_check.py --url http://127.0.0.1:8888   # the image gate
python3 -I tools/video_check.py --url http://127.0.0.1:8888    # the video gate
python3 -m mac down
```

An image request is any chat request with an `image_url` part holding a data URL:

```bash
IMG=$(base64 -i photo.jpg)
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages": [{"role": "user",
  "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
              {"type": "text", "text": "What is in this photo?"}]}], "max_tokens": 200}'
```

A video request is the same with a `video_url` part (MP4, MOV, WebM or MKV; up to 2 a request, 16 MiB each):

```bash
VID=$(base64 -i clip.mp4)
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages": [{"role": "user",
  "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,'"$VID"'"}},
              {"type": "text", "text": "What happens in this video?"}]}], "max_tokens": 200}'
```

Autostart: `launchd/com.tensorfold.qwen38-vision.plist` (install lines at its top). [HERMES.md](HERMES.md) is the
operations note for an agent managing the server on this Mac.

## Settings

As the text-only sibling's, plus:

| Setting | Default | Notes |
|---|---|---|
| `parallel` | 4 | replies at once (shared lane path); 1: the single-reply driver; 8 needs `context = 131072` (below) |
| `context` | 262144 | the trained window; up to 1048576 with static YaRN x4 for every prompt (see the 1M section) |
| `prompt_cache_gib` | 24 | kept prompt states; 0 off, -1 what 70% of RAM leaves (nothing with four streams) |
| `prompt_cache_over_cap` | true | `--prompt-cache-over-cap`: the cache may pass the 70% share; without it 24 is refused |
| `load_limit_gb` | 240 | `FLASHNEXT_LOAD_LIMIT_GB`: how far the slots may take the server past 70% of RAM |
| `vision` | true | image input; false serves text only, without the tower |
| `vision_image_tokens`, `vision_max_images` | 4096, 4 | a request's image tokens in all, and images |
| `video` | true | video input: `mac build` builds a static FFmpeg 9.0.2 and libtfvideo once (about a minute) |
| `vision_video_tokens` | 16384 | the visual tokens a request's videos share (a 30 s 720p clip takes about 8,000) |
| `port` | 8888 | |
| `temperature` | `"0"` | the default for requests that send none: greedy; a request's own temperature samples |

## Not supported

- No image or video URLs (data URLs only), no grammars, no audio.
- Without a `seed`, a sampled request's seed comes from its prompt (TensorFold's design), so the same request gives
  the same reply; send a `seed` for a different one.
- Past 262,144 tokens only with `context = 1048576`, one stream at a time (see the 1M section).
- Image prompts are read whole and never kept by the prompt cache; `--learn` needs `parallel = 1`.
- Past 2,048 tokens the sparse indexer scores blocks before the last image's end at their slot positions, not their
  multimodal ones (selection only; the long-context image checks pass with it).

## License

MIT ([LICENSE](LICENSE)). The model is under the Qwen Community License 1.0; TensorFold is Apache-2.0.
