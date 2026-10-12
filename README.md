# Qwen3.8-Flash-Next with image and video input, four streams and a 1M window on one M5 Ultra

Serve [TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP)
on one Apple M5 Ultra (256 GB) through [TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine:
text, images and video in one OpenAI-compatible API (`image_url` and `video_url` parts), four replies at once sharing each forward, a
1,048,576-token window, MTP drafts, a launchd unit.

Upstream TensorFold v1.0.7 serves Flash Next on Metal one reply at a time, up to 262,144 tokens, text only. This recipe
builds the [`flashnext-video`](https://github.com/BobClawblaw/TensorFold/tree/flashnext-video)
branch, v1.0.7 plus five things:

- **Image input on Metal** (`zig/src/families/flashnext/vision.zig`, `image.zig`, `vis_prompt.zig`): transformers'
  `Qwen4ExpVisionModel` in bf16 on the GPU (within bf16 noise of it on eight images), the processor's preparation
  (ImageIO, EXIF orientation, alpha over white, `smart_resize`, PIL's bicubic), and the tower's rows over the image
  tokens with transformers' multimodal positions, in a frame shifted so text after the last image sits at its slot:
  decode, drafts and shared rounds are untouched, and text replies are byte-identical to v1.0.7's.
- **Four streams** from upstream [#606](https://github.com/ashhart/TensorFold/pull/606) (cwschroeder's shared lane path),
  with the load limit offered there (`FLASHNEXT_LOAD_LIMIT_GB`) and images in each stream's prefill.
- **A 1M window**: static YaRN x4 past the trained 262,144 (transformers' table, unit-tested) in every rotary kernel,
  the engine's cap a runtime value (a 262k server allocates as before), slot 0 holding the full window and slots 1-3
  the trained one, long requests waiting for slot 0, and prompt chunks shortened past 262k so the indexer's scores fit.
- **Prompt passes that don't stall the others**: a stream's prompt is read in steps between the other streams' rounds
  (the lane core's `prefill_step`), 1,024 rows a step while others decode and only once their rounds have had as long
  as the step before, so a long read and the replies around it share the GPU half and half; alone it reads at full
  speed, and a short prompt's single step never waits.
  With `context = 262144` the kernels take the plain rotary, bit-identical to v1.0.7.
- **Video input** (`zig/src/families/flashnext/video.zig`, upstream [#618](https://github.com/ashhart/TensorFold/pull/618)'s
  libtfvideo over FFmpeg 9.0.2): frames sampled at 2 a second (4 to 256), sized over the clip's pixel budget and
  paired as transformers' Qwen3-VL video processor does, each pair an image to the tower, written as
  `<X.Y seconds>` then its rows. The server's `/tokenize` equals transformers' `input_ids` on 12 clips across codecs
  and containers ([`transformers_prompt_ids.txt`](evidence/2026-10-11-video/transformers_prompt_ids.txt)).

Nothing to convert: the 6-bit checkpoint carries the tower (`vision_tower.*`, bf16, 1.7 GiB loaded with scratch).

## Measured on this Mac

Mac Studio M5 Ultra, 256 GB, engine `68306275` (`4cef40eb` for the stall check, `fdbd0228` for video), checkpoint `45032a19`, greedy, thinking off, the defaults (four
streams, 1,048,576 tokens, YaRN x4), served by the launchd unit on 2026-10-11
([`evidence/2026-10-11-1m-parallel4/`](evidence/2026-10-11-1m-parallel4/)).

Decode (`bench/ruler.py`, 256-token replies, median of 3, [`ruler.json`](evidence/2026-10-11-1m-parallel4/ruler.json)):

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

Quality with static YaRN x4 on every prompt, the Spark recipe's suite, thinking off
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

For an agent loop on this Mac, `context = 262144` with a 24 GiB prompt cache
([`evidence/2026-10-12-262k-cache/`](evidence/2026-10-12-262k-cache/), see the changelog): four streams, no YaRN,
decode 129-177 tok/s one user and 210-322 at four, a 255,674-token needle at 3,635 tok/s, each agent call reading
only its new turns. The 1M defaults leave no memory for the prompt cache.

Static YaRN costs short prompts a little, as Qwen's card says it may. `context = 262144` in `local.toml` serves four
streams on the plain rotary instead (about 205 GiB).

## Requirements

- One Apple Silicon Mac with 256 GB: the defaults hold about 222 GiB once the caches fill (`load_limit_gb = 240` lets
  them past the engine's 70% share). Smaller Macs: `context = 262144` (~205 GiB) or `parallel = 1` with it (~176 GiB).
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
| `parallel` | 4 | replies at once (shared lane path); 1: the single-reply driver |
| `context` | 1048576 | past 262,144 static YaRN x4 for every prompt; slot 0 holds it, the other slots 262,144 |
| `load_limit_gb` | 240 | `FLASHNEXT_LOAD_LIMIT_GB`: how far the slots may take the server past 70% of RAM |
| `vision` | true | image input; false serves text only, without the tower |
| `vision_image_tokens`, `vision_max_images` | 4096, 4 | a request's image tokens in all, and images |
| `video` | true | video input: `mac build` builds a static FFmpeg 9.0.2 and libtfvideo once (about a minute) |
| `vision_video_tokens` | 16384 | the visual tokens a request's videos share (a 30 s 720p clip takes about 8,000) |
| `port` | 8888 | |
| `temperature` | `"0"` | the engine decodes Flash Next greedily only and refuses a request with a temperature |

## Not supported

- Greedy only. No image or video URLs (data URLs only), no grammars, no audio.
- One stream at a time past 262,144 tokens (slot 0); a second long request waits for it.
- Image prompts are read whole and never kept by the prompt cache; `--learn` needs `parallel = 1`.
- Past 2,048 tokens the sparse indexer scores blocks before the last image's end at their slot positions, not their
  multimodal ones (selection only; the long-context image checks pass with it).
- Once, early in testing, a 553k prompt failed with a GPU timeout while other jobs held memory; four reruns of long
  prompts with and without concurrent requests passed, and the server stayed up (that request alone errored).

## License

MIT ([LICENSE](LICENSE)). The model is under the Qwen Community License 1.0; TensorFold is Apache-2.0.
