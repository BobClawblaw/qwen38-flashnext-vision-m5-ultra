# Changelog

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
