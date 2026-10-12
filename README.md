# Qwen3.8-Flash-Next with image input on one M5 Ultra

Serve [TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP)
with its vision tower on one Apple M5 Ultra (256 GB) through [TensorFold](https://github.com/ashhart/TensorFold)'s
native Metal engine: text and images in the same OpenAI-compatible API (`image_url` parts), the full 262,144-token
window, MTP drafts, kept prompt states for text, a launchd unit.

Upstream TensorFold v1.0.7 reads Flash Next's images on CUDA only. This recipe builds the
[`flashnext-vision`](https://github.com/BobClawblaw/TensorFold/tree/flashnext-vision) branch, which is v1.0.7 plus
image input on Metal:

- **The tower** (`zig/src/families/flashnext/vision.zig`): transformers' `Qwen4ExpVisionModel` in bf16 on the GPU, 27
  blocks of 1152 with a learned position table resampled to each image and a 2x2 merger to the model's 2560. Checked
  against transformers on eight images: within bf16 noise of it (cosine to the fp32 tower 0.997-0.9996, the bf16
  reference's own 0.998-0.9998).
- **Preparation** (`image.zig`): ImageIO decode with EXIF orientation and alpha over white, `smart_resize` to multiples
  of 32 pixels, PIL's bicubic resampler, patches in merge windows. Within one 8-bit level of transformers' processor on
  PNGs; JPEGs differ by ImageIO's decoder.
- **The language model** (`vis_prompt.zig`, four hooks in `replay.zig`): the tower's rows replace the image tokens'
  embeddings, and image tokens rotate at transformers' multimodal positions (interleaved sections 11/11/10) in a frame
  shifted so the text after the last image sits at its own slot. Decode, drafts and GPU-side rounds are untouched:
  text replies are byte-identical to v1.0.7's.

Nothing to convert: the 6-bit checkpoint carries the tower (`vision_tower.*`, bf16, 1.7 GiB loaded with scratch).

## Measured on this Mac

Mac Studio M5 Ultra, 256 GB, engine `b103ecc2`, checkpoint `45032a19`, greedy, thinking off, one request at a time,
served by the launchd unit on 2026-10-11 ([`evidence/2026-10-11-launchd/`](evidence/2026-10-11-launchd/)).

Images, `tools/vision_check.py` (images drawn by the check itself; every drafted reply byte-equal to `"draft": false`):

| Check | Prompt tokens | First token | Reply |
|---|---:|---:|---|
| shapes and sides | 183 | 0.18 s | "There is a red circle on the left and a blue square on the right." |
| top and bottom | 145 | 0.15 s | "The shape on top is a green triangle." |
| counting five dots | 98 | 0.13 s | "5" |
| image, then 9k words with a fact | 10,299 | 2.8 s | "1) A red circle and a blue square. 2) Biscuit." |
| 9k words with a fact, then an image | 10,266 | 2.3 s | "1) Triangle 2) Biscuit" |
| two images in one request | 256 | 0.20 s | both described in order |
| a follow-up turn about the image | 178 | 0.16 s | "Below it is a yellow circle." |

A text reply after the image requests equals the one before them. A full-size photo costs more: a 2000x1500 image
takes 2,914 tokens and 1.5 s in the tower (its attention keeps fp32 scores in memory), 3.4 s to the first token.

Text, unchanged by the vision build (`bench/ruler.py`, 256-token replies, median of 3,
[`ruler-1user.json`](evidence/2026-10-11-launchd/ruler-1user.json)):

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 144.6 | 205.6 | 172.6 | 194.7 | 0.10-0.20 s |

The suite's core checks on the same server ([`suite/`](evidence/2026-10-11-launchd/suite/report.md)): drafted equals
undrafted 12/12 (144 against 83 tok/s), streaming 4/4, multi-turn 8/8, stop 3/3, thinking 10/10, copy 2/2, max_tokens
3/3, tools 59/60, JSON 30/30. The full suite, prompt passes to 261k tokens and the rest of the text numbers are the
text-only sibling's ([qwen38-flashnext-mlx6bit-m5-ultra](https://github.com/BobClawblaw/qwen38-flashnext-mlx6bit-m5-ultra)):
the same engine for text, IFEval 134/150, GSM8K 242/250, MMLU 282/342, HumanEval 157/164, prompts at 3,600-4,400 tok/s.

## Requirements

- One Apple Silicon Mac with 192 GB or more (the floor is 149.3 GiB plus the tower's 1.7 GiB and the kept prompt
  states), macOS with Xcode and its Metal toolchain, Zig 0.17.0, Python 3.11+ (stdlib only), the `hf` CLI.
- Disk: the checkpoint, 147 GiB, plus the weight packs the engine writes under `~/.cache/tensorfold/packs`.

## Quick start

```bash
git clone https://github.com/BobClawblaw/qwen38-flashnext-vision-m5-ultra.git && cd qwen38-flashnext-vision-m5-ultra
cp local.example.toml local.toml && $EDITOR local.toml   # host, port, paths if not the defaults
python3 -m mac validate            # the settings and the guards, touches nothing
python3 -m mac up                  # builds the engine once, finds or downloads the weights, waits for the API
python3 -m mac chat "Say hello in one sentence."
python3 -I tools/vision_check.py --url http://127.0.0.1:8888   # the image gate
python3 -m mac down
```

An image request is any chat request with an `image_url` part holding a data URL:

```bash
IMG=$(base64 -i photo.jpg)
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages": [{"role": "user",
  "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
              {"type": "text", "text": "What is in this photo?"}]}], "max_tokens": 200}'
```

Autostart: `launchd/com.tensorfold.qwen38-vision.plist` (install lines at its top). [HERMES.md](HERMES.md) is the
operations note for an agent managing the server on this Mac.

## Settings

As the text-only sibling's, plus:

| Setting | Default | Notes |
|---|---|---|
| `vision` | true | image input; false serves text only, without the tower |
| `vision_image_tokens` | 4096 | image tokens a request may hold in all: the tower's scratch is sized for it |
| `vision_max_images` | 4 | images a request may hold |
| `port` | 8888 | |
| `temperature` | `"0"` | the engine decodes Flash Next greedily only and refuses a request with a temperature |

## Not supported

- One reply at a time (upstream [#606](https://github.com/ashhart/TensorFold/pull/606) is bringing shared streams).
- Greedy only. No video, no image URLs (data URLs only), no grammars, no YaRN past 262,144 tokens.
- Image prompts are read whole and never kept: their positions depend on every image in them, so no prefix state
  stands for another prompt. Text prompts keep their states as before.
- Past 2,048 tokens the sparse indexer scores blocks before the last image's end at their slot positions, not their
  multimodal ones (selection only; attention over the chosen blocks is exact). Long-context checks above pass with it.

## License

MIT ([LICENSE](LICENSE)). The model is under the Qwen Community License 1.0; TensorFold is Apache-2.0.
