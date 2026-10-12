"""The Spark recipe's quality suite and decode ruler, run from a pinned clone under .external/ (never copied here)."""

from __future__ import annotations

import subprocess
import sys

from .config import Settings
from .log import log


def ensure_clone(s: Settings) -> None:
    d = s.external_dir
    if (d / "quality" / "suite.py").is_file():
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == s.spark_recipe_sha:
            return
        log(f"{d} is at {head[:8]}; fetching the pinned {s.spark_recipe_sha[:8]}")
    else:
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "remote", "add", "origin", s.spark_recipe_repo], check=True)
    subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", s.spark_recipe_sha], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    log(f"Spark recipe at {s.spark_recipe_sha[:8]} under {d}")


def suite(s: Settings, passthrough: list[str]) -> int:
    """python3 -m quality.suite --url <ours> --model <ours> ...; HumanEval needs its Docker sandbox, skipped unless asked."""
    ensure_clone(s)
    args = [sys.executable, "-m", "quality.suite", "--url", s.url, "--model", s.served_name]
    if "--only" not in passthrough and "--skip" not in passthrough:
        args += ["--skip", "humaneval", "vision"]        # the sandbox is a Docker image; the Metal engine takes no images
    if "--out" not in passthrough:
        args += ["--out", str(s.state_dir / "quality-out")]
    return subprocess.run(args + passthrough, cwd=s.external_dir).returncode


def ruler(s: Settings, passthrough: list[str]) -> int:
    """python3 bench/ruler.py --url <ours> --model <ours> --concurrency 1 ...: Flash Next on Metal serves one reply at a time."""
    ensure_clone(s)
    args = [sys.executable, "bench/ruler.py", "--url", s.url, "--model", s.served_name]
    if "--concurrency" not in passthrough:
        args += ["--concurrency", "1"]
    return subprocess.run(args + passthrough, cwd=s.external_dir).returncode
