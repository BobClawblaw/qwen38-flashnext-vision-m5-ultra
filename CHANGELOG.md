# Changelog

## 2026-10-11

- First version: Qwen3.8-Flash-Next 6-bit with image input on one M5 Ultra, on the `flashnext-vision` branch of
  TensorFold (v1.0.7 plus the tower, image preparation and multimodal positions on Metal, commit `b103ecc2`). Image
  gate (`tools/vision_check.py`) passes 7/7 with drafted replies byte-equal to undrafted; text decode and the suite's
  core checks equal the text-only recipe's. Served on :8888 to the LAN without an API key, superseding MTPLX on this
  Mac, by the launchd unit; HERMES.md is the operations note. The launchd unit's PATH now holds /usr/sbin (`lsof`).
