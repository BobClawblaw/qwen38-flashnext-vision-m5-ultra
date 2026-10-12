"""The guards: python3 -m unittest discover tests. Stdlib only; nothing here touches the engine or the network."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mac import config  # noqa: E402


def load(**env):
    return config.load(local_file="none", env={k.upper(): str(v) for k, v in env.items()})


class Defaults(unittest.TestCase):
    def test_defaults_load_and_serve_args(self):
        s = load()
        self.assertEqual(s.profile, "chat")
        self.assertTrue(s.drafts)
        self.assertEqual(s.context, 1048576)
        self.assertEqual(s.parallel, 4)
        args = s.serve_args("/m")
        self.assertEqual(args[:2], ["serve", "/m"])
        self.assertIn("--no-thinking", args)
        self.assertNotIn("--no-drafts", args)
        self.assertEqual(args[args.index("--temperature") + 1], "0")   # the engine decodes Flash Next greedily only
        self.assertNotIn("--prompt-cache-gib", args)     # -1: the engine's default
        self.assertIn("--dashboard", args)
        self.assertIn("--no-update-check", args)
        self.assertEqual(args[args.index("--parallel") + 1], "4")
        self.assertEqual(s.server_env()["FLASHNEXT_LOAD_LIMIT_GB"], "240")
        self.assertEqual(args[args.index("--vision-image-tokens") + 1], "4096")
        self.assertIn("--vision", args)

    def test_reference_profile_drops_drafts(self):
        s = load(profile="reference")
        self.assertFalse(s.drafts)
        self.assertIn("--no-drafts", s.serve_args("/m"))

    def test_explicit_setting_beats_profile(self):
        s = load(profile="reference", drafts="1")
        self.assertTrue(s.drafts)

    def test_env_overrides_and_summary(self):
        s = load(port="8090", temperature="0", prompt_cache_gib="0")
        self.assertEqual(s.port, 8090)
        self.assertEqual(s.url, "http://127.0.0.1:8090")
        args = s.serve_args("/m")
        self.assertEqual(args[args.index("--temperature") + 1], "0")
        self.assertEqual(args[args.index("--prompt-cache-gib") + 1], "0")
        self.assertIn("url=http://127.0.0.1:8090", s.summary())

    def test_candidate_repos_and_snapshot_dir(self):
        s = load()
        self.assertEqual(s.candidate_repos, ["TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP"])
        self.assertTrue(str(s.snapshot_dir()).endswith(f"models--TensorFold--Qwen3.8-Flash-Next-MLX-6bit-MTP/snapshots/{s.revision}"))

    def test_learn_flags(self):
        s = load(learn="1", learn_dir="~/learned", learn_gib="24", parallel="1")
        args = s.serve_args("/m")
        self.assertIn("--learn", args)
        self.assertTrue(args[args.index("--learn-dir") + 1].endswith("/learned"))
        self.assertEqual(args[args.index("--learn-gib") + 1], "24")

    def test_server_env(self):
        s = load(extra_env="TF_X=1 TF_Y=a=b", request_log="~/req.log")
        env = s.server_env()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TF_Y"], "a=b")
        self.assertTrue(env["TENSORFOLD_REQUEST_LOG"].endswith("/req.log"))


class Refusals(unittest.TestCase):
    def refuse(self, fragment, **env):
        with self.assertRaises(config.ConfigError) as cm:
            load(**env)
        self.assertIn(fragment, str(cm.exception))

    def test_context_bounds(self):
        self.refuse("YaRN x4", context="2097152")
        self.refuse("multiple of 256", context="1000")
        self.refuse("not a decimal integer", context="0100")

    def test_profile(self):
        self.refuse("must be one of", profile="concurrent")

    def test_owned_and_unused_flags(self):
        self.refuse("passes itself", extra_args="--port 1")
        self.refuse("passes itself", extra_args="--no-dra")          # a prefix
        self.refuse("passes itself", extra_args="--parallel 4")
        self.refuse("parallel=1", learn="1")
        self.refuse("not a Flash Next-on-Metal flag", extra_args="--kv-dtype int8")

    def test_vision_off_and_bounds(self):
        self.assertNotIn("--vision", load(vision="0").serve_args("/m"))
        self.refuse("outside", vision_image_tokens="32")
        self.refuse("passes itself", extra_args="--vision-max-images 8")
        self.refuse("not a Flash Next-on-Metal flag", extra_args="--vision-urls")

    def test_video(self):
        args = load().serve_args("/m")
        self.assertEqual(args[args.index("--vision-video-tokens") + 1], "16384")
        self.refuse("passes itself", extra_args="--vision-video-tokens 8")
        self.refuse("outside", vision_video_tokens="32")

    def test_learn_dir_needs_learn(self):
        self.refuse("learn=false", learn_dir="~/learned")

    def test_binary_must_exist(self):
        self.refuse("is not a file", binary="/nonexistent/tensorfold-native")

    def test_shapes(self):
        self.refuse("40-hex", revision="abc")
        self.refuse("not KEY=VALUE", extra_env="NOEQUALS")
        self.refuse("decimal", temperature="hot")
        self.refuse("fraction", compact_at="2")
        self.refuse("outside", port="70000")
        self.refuse("org/name", repo="nothing")

    def test_unknown_setting(self):
        with self.assertRaises(config.ConfigError):
            config.load(local_file=Path(__file__).with_name("bad.toml"), env={})


if __name__ == "__main__":
    unittest.main()
