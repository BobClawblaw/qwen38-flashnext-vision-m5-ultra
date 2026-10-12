# Design

One Mac, one engine process, one checkpoint served from the Hugging Face cache by folder. The orchestrator is
stdlib Python (`mac/`), written for this recipe.

- **Settings** (`mac/config.py`): `recipe.toml` defaults, `local.toml` per Mac, environment variables in upper case,
  then the guards. A profile fills only what nothing more specific set. `validate` is pure: it reads files and the
  environment and touches nothing else, so CI can run it on every profile and every refusal.
- **Engine** (`mac/engine.py`): TensorFold is pinned by commit (`tensorfold_sha`, v1.0.7). `build` adds a detached
  git worktree of the local checkout (or a clone) at that commit under `tf_cache/engine/<sha>` and runs
  `zig build native -Dcpu=apple_m1` there, so the user's checkout and its `zig-out` are never touched. `binary` points
  the recipe at a branch build instead.
- **Weights** (`mac/weights.py`): the pinned repository at its 40-hex revision. A `same_as` repository at the same
  revision is the same git tree, so a cached copy of either serves (none is known for the 6-bit checkpoint);
  completeness means every shard the index names resolves to a file. Download through the `hf` CLI.
- **Serve** (`mac/serve.py`): `info` gives the memory floor; `up` refuses when free plus inactive memory is under it
  plus a margin, starts the binary in its own session with the log appended under `tf_cache`, writes a pid file under
  `.run-state/`, and waits until `/v1/models` lists the served name. `--foreground` execs the binary instead, for
  launchd. `down` is SIGTERM, then SIGKILL after a grace.
- **Quality and speed** (`mac/external.py`): the Spark recipe's `quality/suite.py` and `bench/ruler.py` are
  stdlib Python against an OpenAI-compatible URL. They run from a shallow clone at a pinned commit under
  `.external/` with this server's URL and model name; nothing is copied into this tree.

What the Spark recipe has that this one does not, and why: two ranks (one Mac here; TensorFold's two-Mac speed-up
mode exists but is a different recipe), Docker and an image (a native binary), a converter (the engine reads this
checkpoint as published), `--kv-dtype`, `--tool-system` and grammars (GLM's or CUDA's on v1.0.7), and any precision
but 6-bit (the engine's Flash Next kernels; see AGENTS.md). The engine branch adds what v1.0.7 lacked here: four
streams (#606), images and video, and static YaRN to 1M, which is not the default: at 1M four streams leave no memory
for the prompt cache (README, "The 1M window and its problems").
