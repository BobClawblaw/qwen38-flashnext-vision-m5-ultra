# Working on this recipe

Serve `TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP` with image input on one M5 Ultra with TensorFold's native Metal
engine, built from the BobClawblaw/TensorFold `flashnext-sampling` branch (vision, #606 streams, video, sampling, an optional 1M window). Everything
in this tree is original to this project: no code copied from other recipes (the Spark and GLM recipes included), no
harness dependencies, stdlib Python only.

- **Entry point:** `python3 -m mac up|down|status|logs|build|pull|info|command|chat|suite|ruler|validate`. Settings:
  `recipe.toml` defaults, `local.toml` per Mac (git-ignored), environment variables in upper case.
  `python3 -m mac validate` is the dry run; `python3 -m unittest discover tests` covers the guards.
- **6-bit is not a choice.** The native engine's Flash Next kernels take affine 6-bit, group 32, and nothing else
  (`zig/src/families/flashnext/affine.zig: checkFlashKernel`); 4-bit served gibberish on v1.0.2 and is refused at load
  since 2026-10-09. Do not pin another precision; a mixed layout is upstream engine work.
- **Sampling.** Shared sessions sample a request's temperature on the host (families/flashnext/sample.zig, the lane
  core's keyed sampler); `temperature = "0"` keeps requests without one greedy. `parallel = 1` samples too (rounds one at a time).
- **The engine is pinned by commit** (`tensorfold_sha`, flashnext-lane-parity `c507c94f`: sethforprivacy's `add3da3` on our flashnext-video, plus the top_k 0 sampler, run with `FLASHNEXT_RESIDENCY=all`: every buffer in the residency set, committed between requests) and built into a worktree under `tf_cache`; never build
  in the user's checkout. To try a branch, build it elsewhere and point `binary` at it; re-pin only after the suite and
  the ruler rerun.
- **The Spark recipe's suite and ruler are called, not copied.** `python3 -m mac suite` and `ruler` run them from a
  pinned clone under `.external/`. Bump `spark_recipe_sha` to pick up new checks. Its `evidence/` numbers come from a
  different checkpoint (INT4-Mixed AutoRound on two GB10s): a rough neighbourhood, not a target.
- **Measure before you claim.** Write the suite's and ruler's JSON under `evidence/<date>-<what>/`, commit it, and
  point the README row at the file. `profile = reference` (`--no-drafts`) is the exactness reference.
- **Memory is unified:** read it with `vm_stat` (`mac/serve.py: free_gib`), the floor comes from `tensorfold-native info`
  (149.25 GiB for this checkpoint). Kept prompt states take what is left by default; `prompt_cache_gib` bounds them.
  Nothing else of this size can be resident at the same time (GLM-5.3-Flash is 151 GiB).
- **Speed knobs to measure, not assume:** `FZ_LANE=1 FZ_GDN=2` (the speed-up recipe's faster decode kernels; replies
  are the same with or without), `--learn` for restart warm-up, `prompt_cache_gib`. A/B each with the ruler before it
  becomes a default.
- **Suite plumbing on this Mac:** give the ruler and suite absolute `--json`/`--out` paths (they run from the clone);
  HumanEval needs Docker (Colima here), `--sandbox-image python:3.12-slim`, and `TMPDIR` under `$HOME` (Colima shares
  only home); dataset downloads need `SSL_CERT_FILE` from certifi on python.org's Python.

- **Not served here:** image and video URLs, `--kv-dtype`, `--tool-system`, grammars, `--speed-up`; the guards refuse the flags.
- **Vision:** `tools/vision_check.py` is the image gate (stdlib, draws its own images) and `tools/video_check.py` the
  video gate (clips in `tools/clips`); run both after any engine or setting change. Image prompts are never kept by the prompt cache (their multimodal frame depends on every image).
  The tower's attention keeps fp32 scores in memory: a full-size photo takes ~1.5 s in it. A fused attention kernel
  is the obvious speed-up.
- **On this Mac** the launchd unit serves it on :8888 to the LAN, no API key, superseding MTPLX: see HERMES.md. Don't
  `mac up`/`down` while the unit is loaded.
- **Memory decides the shape:** every buffer is held (`FLASHNEXT_RESIDENCY=all`), so the server's memory stays wired.
  The default is two streams (`parallel = 2`, upstream #606's shared lane path, not merged) at the trained 262,144 with
  an 8 GiB prompt cache, ~189 GiB wired, and the lone driver off (`FLASHNEXT_NO_LONE=1`: it ran some replies at half
  speed). `mac validate` and `up` refuse settings past 74% of RAM less `gpu_headroom_gib` (config.gpu_check). Four
  streams worked at 202-209 GiB wired but left 25 GiB free and ran some replies slow. Never raise the wired limit or
  run anything large beside it without asking.
- **Prompt passes step between rounds** (backend.zig prefillStep, fair share with decoding streams);
  `tools/stall_check.py 280000` checks it. FZ_BATCH_LOG=1 times each chunk.
