# Qwen3.8-Flash-Next with image and video input on one M5 Ultra

Serve [TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP)
on one Apple M5 Ultra (256 GB) through [TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine:
text, images and video in one OpenAI-compatible API (`image_url` and `video_url` parts), the trained 262,144-token
window with an 8 GiB prompt cache (an agent's next call reads only its new turns), MTP drafts, a launchd unit. It serves
two replies at once by default, each up to 262,144 tokens, so that it fits macOS's GPU memory with room for other apps
([GPU memory](#gpu-memory)); more streams are a setting away. The previous engine's 1,048,576-token window is not in
this one ([below](#the-1m-window-and-its-problems)).

The engine is TensorFold's main with cwschroeder's shared sessions
([#606](https://github.com/ashhart/TensorFold/pull/606)) and our images and video on top
([#986](https://github.com/ashhart/TensorFold/pull/986), not merged yet), pinned at `8e4ba6db` on the fork's
[`flashnext-shared-vision`](https://github.com/BobClawblaw/TensorFold/tree/flashnext-shared-vision) branch:

- **Images on Metal:** the Qwen-VL tower TensorFold's 27B uses (`zig/src/core/qwen_vl/`), at Flash Next's 2,560-wide
  output, in bf16 on the GPU. Its rows replace the image tokens in all four hyper-connection streams, turned by
  interleaved M-RoPE in a frame shifted so text after the last image sits at its slot: decode, drafts and shared
  rounds run as for text, and text replies are token-identical to main's. PNG is decoded natively; JPEG by
  libjpeg-turbo 3.1.4.1, which `mac build` builds once into the cache and puts beside the server (the engine refuses
  any other release); WebP is refused.
- **Video:** frames sampled at 2 a second (4 to 256), sized over the clip's pixel budget and paired as transformers
  5.17's Qwen3-VL video processor does, each pair an image to the tower, written as `<X.Y seconds>` then its block.
  The served prompt is as long as the processor's on three clips (239, 387 and 8,215 tokens).
- **Two streams** on #606's shared sessions (`parallel`), with its load limit (`FLASHNEXT_LOAD_LIMIT_GB`), sampling
  and prompt passes stepped between the other streams' rounds.
- **Image and video chats in the prompt cache:** a follow-up turn with the same images resumes past them (a turn that
  adds an image reads its prompt again), and replies equal the cache off's.
- **Structured output:** `response_format` (`json_object`, `json_schema`), the `guided_*` fields (JSON, regex,
  choice, EBNF grammar), `tool_choice: "required"` and a named function, through xgrammar v0.2.8's `libtfgrammar`,
  which `mac build` builds once beside the server. A constrained row's other tokens are masked on the GPU before the
  draw; drafts stay on, and drafted, `"draft": false` and concurrent replies are the same.
- **#606's first token:** after each reply the shared host read the next turn ahead in the background even when the
  cache kept nothing of it, and the next request waited on it (0.28-0.31 s to the first token, drafts 126/73 tok/s);
  #986 skips such passes.

The previous engine, `7808bee8` (the fork's `flashnext-lane-parity`, on v1.0.7), had its own tower and an optional
1M window; the measurements below name the engine they were taken on.

Nothing to convert: the 6-bit checkpoint carries the tower (`vision_tower.*`, bf16, 1.7 GiB loaded with scratch).

## Measured on this Mac

Mac Studio M5 Ultra, 256 GB, checkpoint `45032a19`, greedy, thinking off, served by the launchd unit.

**The current engine** (`8e4ba6db`, the defaults: two 262k streams, an 8 GiB cache; 2026-10-17, the recipe harness's
speed plan twice on the recipe's build, beside the previous engine `7808bee8` served the same way that night):

| Engine | Streams | prose | code | structured | list | first token | drafted / plain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `8e4ba6db` | 1 | 138.3 / 142.4 | 204.3 / 206.1 | 173.7 / 173.4 | 207.1 / 211.5 | 0.11-0.12 s | 155 / 84 |
| `8e4ba6db` | 2, aggregate | 195.0 / 195.1 | 263.0 / 259.7 | 218.7 / 219.7 | 276.0 / 276.6 | 0.17-0.19 s | |
| `7808bee8` | 1 | 138.4 | 202.8 | 169.6 | 207.4 | 0.11-0.12 s | 154 / 83 |
| `7808bee8` | 2, aggregate | 196.2 | 253.3 | 220.6 | 268.8 | 0.18-0.20 s | |

The harness's structured check passed 18/18 on both runs (six grammar kinds, greedy, sampled and thinking; a bad regex
is a 400); on the previous engine and on `8a509ab5` it was 0/18 (refused). A grammar costs no decode speed: the harness's `structured_speed` check (greedy, thinking off, a 12-person `json_schema` reply and a `json_object` one, the same prompts with the grammar off and on) gave one stream 206 / 180 tok/s with the grammar (194 / 184 without; 92 / 93 undrafted), two grammar streams 263 / 236 tok/s in all (253 / 239 without), and a plain stream 129 / 123 tok/s beside a grammar stream (125 / 114 beside a plain one). On the served unit, drafted equals
`"draft": false` in all 12 greedy and sampled cases, a constrained reply beside a plain one leaves the plain one
unchanged, `tool_choice: "required"` and a named function give valid calls, and eight plain replies (greedy and
sampled) are token-identical to `8a509ab5`'s; the image and video gates pass on it again. On `8a509ab5`'s served unit:
the image gate 8/8 (`tools/vision_check.py`: small images 0.15-0.22 s to the first token, 10k-token image prompts
2.5 s, drafted equal to undrafted), the video gate 7/7 (`tools/video_check.py`: the 30-second 720p clip 3.2 s to the
first token; the previous engine missed its scenes' order that night), a JPEG through the built libturbojpeg, and
180.4 GiB wired at startup (the masks add 1.4 MiB at two streams). Against a server with the cache off, five-turn image and three-turn
video chats gave equal replies while follow-ups resumed 277 to 580 tokens past their images. The quality suite
(thinking off and on) was not run on this engine; the tables below are the previous engines'.

**Four streams** (`parallel = 4`, 262,144 tokens, 24 GiB prompt cache; the default until 2026-10-12, now past the GPU
budget below; engine `ddc17bb6`, [`evidence/2026-10-12-262k-cache/`](evidence/2026-10-12-262k-cache/)):

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
`160fed1c` runs its sampled rounds two deep as greedy ones are, 2-3% faster with the same replies (`add3da3`, now pinned, predates that
commit; it applies to `parallel = 1` only). A 255,674-token prompt read at 3,635 tok/s (70 s) and
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
| JSON mode (asked in the prompt) | 30/30 | 30/30 |
| IFEval loose / strict | 132 / 124 | 134 / 128 |
| GSM8K | 241/250 | 242/250 |

Video ([`video_check.txt`](evidence/2026-10-11-video/video_check.txt), `tools/video_check.py`, clips in
`tools/clips`, every drafted reply byte-equal to `"draft": false`): a yellow square going right in H.264 and VP9, a red
square going left along the bottom, a 30-second 720p clip's two scenes in order (8,206 prompt tokens, first token
5.0 s), a video beside an image, and a video and a text request at once each equal to their replies alone: 7/7.

## The 1M window and its problems

**Not in the current engine:** it has no YaRN, and `mac validate` refuses `context` past 262,144. What follows was
measured on the previous engines and stays for the record.

`context = 1048576` in `local.toml` (and `prompt_cache_gib = 0`, see the first point) served up to 1,048,576 tokens.
It worked, with a needle found at 982k, but it was the default only from 2026-10-11 to 2026-10-12 and these are why it
was not:

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

## GPU memory

The defaults serve two streams at 262,144 tokens with an 8 GiB prompt cache: 180.4 GiB wired at startup on the current
engine (about 189 GiB on the previous one with every buffer held), the rest left to macOS and other apps. Everything on
the GPU counts (an idle ComfyUI holds 8 GB; with a model loaded, more). Measured on this Mac, 2026-10-12, on the
previous engine with its residency fix below, 30 mixed single replies a run
([`evidence/2026-10-12-lane-parity/`](evidence/2026-10-12-lane-parity/)):

| Streams at 262k | Cache | Wired | Slow replies | Lone driver | Other checks |
|---|---|---|---|---|---|
| 1 | 6 GiB | ~181 GiB | 0 of 90 | (single-reply host) | idle first token 0.1 s |
| 2 (default) | 8 GiB | ~189 GiB | 0 of 90 | off (`FLASHNEXT_NO_LONE=1`) | two 99k prompts beside two chats; idle 0.15 s |
| 2 | 6 GiB | ~188 GiB | 3 of 60 | on | |
| 4 | 5 GiB | 202-209 GiB | 2 of 30 | on | 8 streams equal solo; 25 GiB free at worst |
| 4 | 5 GiB | ~206 GiB | 0 of 60 | off | two 99k prompts beside two chats; 26.5 GiB free at worst |

**The lone driver** (`add3da3`'s: a request alone runs the single-reply driver's rounds, handed to the shared rounds
when a second arrives) occasionally ran a reply at half speed here, with identical rounds; with it off a single
request runs the shared rounds (priced drafts) at the same speeds and none was slow, so the default turns it off.

**Residency.** The current engine keeps its buffers, its sessions' and the tower's in one residency set (#606's) that
the idle keepalive keeps wired; the slow-reply runs above and the idle table below were not repeated on it. The previous
engine's account follows. **Every buffer, kept whole** (`FLASHNEXT_RESIDENCY=all`, then the default `extra_env`). A
second or two after a request macOS unwires an idle process's GPU memory and may compress or drop it; the next request
then waits while it comes back (MTPLX's scheduler notes the same, ~9 ms a GiB). A residency set and the idle keepalive
keep it wired, but a set that misses some buffers made rounds that touched both kinds slower, and committing the set
mid-request stalled one. The engine (fork `7808bee8`) now holds every live Metal buffer: those made or freed during a
request join or leave the set staged, and the keepalive commits them between requests. `keep_warm` (a day) bounds how
long after the last request it keeps ticking.

| One stream at 262k, macOS defaults | Decode, 30 mixed replies | First token after 10-30 s idle | After 5 min idle | 25k-token prompt after 60 s idle |
|---|---|---|---|---|
| every buffer, kept whole (default) | 0 of 30 slow, twice | 0.09-0.10 s | 0.10 s | 5.4 s |
| no residency set | 0 of 30 slow | 2.1-2.4 s | | 24 s |
| engine and tower, set at startup | 10-30% at 50-100 tok/s | 0.1 s | | 5.9 s |
| weights only | a steady ~45% slower | 0.22 s | | 6.4 s |
| every buffer, a commit per buffer | new prompts stalled at 5 tok/s | | | |

`sudo sysctl iogpu.disable_wired_collector=1` (macOS keeps all GPU memory wired, for every app) gave the same speeds
with no set; it is not needed now, and it costs other apps the memory macOS could otherwise reclaim.

`mac validate` and `mac up` estimate the server's GPU memory (153.5 GiB for the engine, 7.5 GiB a 262k stream, the
prompt cache) and refuse settings that, with `gpu_headroom_gib` (12) kept for other apps, pass 74% of RAM, or
`iogpu.wired_limit_mb` when it is set. Two streams at 262k with 8 GiB: about 176.5 GiB of a 189 GiB budget (an estimate
from the engine's buffers; wired memory runs ~12 GiB above it). Four streams at 262k pass it; run them with
`gpu_headroom_gib = 0` and `prompt_cache_gib = 5` at the cost of the free memory and the slow replies above.

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

- One Apple Silicon Mac with 256 GB: the defaults keep about 189 GiB wired once the prompt cache fills
  (`prompt_cache_over_cap` lets the 8 GiB pass the engine's 70% share). See [GPU memory](#gpu-memory).
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
python3 -m mac restart             # once no request is in flight (the launchd unit when loaded); --force after --timeout
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
| `parallel` | 2 | replies at once on the shared lane path; 1 the single-reply driver; within the GPU budget |
| `context` | 262144 | the trained window, the current engine's most (the 1M window was the previous engine's) |
| `prompt_cache_gib` | 8 | kept prompt states (about 1 GiB a 32k-token conversation); 0 off, -1 what 70% of RAM leaves |
| `prompt_cache_over_cap` | true | `--prompt-cache-over-cap`: the cache may pass the engine's 70% share |
| `gpu_headroom_gib` | 12 | GPU memory kept for other apps; `validate` and `up` refuse settings past the budget |
| `load_limit_gb` | 240 | `FLASHNEXT_LOAD_LIMIT_GB`: how far the slots may take the server past 70% of RAM |
| `vision` | true | image input; false serves text only, without the tower |
| `vision_image_tokens`, `vision_max_images` | 4096, 4 | a request's image tokens in all, and images |
| `video` | true | video input: `mac build` builds a static FFmpeg 9.0.2 and libtfvideo once (about a minute) |
| `vision_video_tokens` | 16384 | the visual tokens a request's videos share (a 30 s 720p clip takes about 8,000) |
| `port` | 8888 | |
| `temperature` | `"0"` | the default for requests that send none: greedy; a request's own temperature samples |

## Not supported

- No image or video URLs (data URLs only), no audio.
- Structured output and `tool_choice: "required"` need the shared sessions: with `parallel = 1` the engine answers
  them with a 400.
- Without a `seed`, a sampled request's seed comes from its prompt (TensorFold's design), so the same request gives
  the same reply; send a `seed` for a different one.
- Nothing past 262,144 tokens: the current engine has no 1M window (see the 1M section).
- A turn that adds an image reads its prompt again (follow-ups with the same images resume); `--learn` needs
  `parallel = 1`. WebP images are refused (PNG and JPEG only).
- Past 2,048 tokens the sparse indexer scores blocks before the last image's end at their slot positions, not their
  multimodal ones (selection only; the long-context image checks pass with it).

## License

MIT ([LICENSE](LICENSE)). The model is under the Qwen Community License 1.0; TensorFold is Apache-2.0.
