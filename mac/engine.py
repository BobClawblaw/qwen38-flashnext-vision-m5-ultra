"""The engine binary: a worktree of TensorFold at the pinned sha under tf_cache, built with Zig, or a given binary."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Settings
from .log import log


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> str:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd)} (exit {p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-4000:]}")
    return p.stdout


def ensure_source(s: Settings) -> Path:
    """A worktree at the pinned sha, from the checkout in `tensorfold_src` or a fresh clone of `tensorfold_repo`."""
    if (s.build_dir / "build.zig").is_file():
        head = _run(["git", "-C", str(s.build_dir), "rev-parse", "HEAD"]).strip()
        if head != s.tensorfold_sha:
            raise RuntimeError(f"{s.build_dir} is at {head[:8]}, not the pinned {s.tensorfold_sha[:8]}; remove it to rebuild")
        return s.build_dir
    s.build_dir.parent.mkdir(parents=True, exist_ok=True)
    src = s.tensorfold_src
    if (src / ".git").exists():
        have = subprocess.run(["git", "-C", str(src), "cat-file", "-e", f"{s.tensorfold_sha}^{{commit}}"], capture_output=True)
        if have.returncode:
            log(f"{src} lacks {s.tensorfold_sha[:8]}; fetching from {s.tensorfold_repo}")
            _run(["git", "-C", str(src), "fetch", s.tensorfold_repo, s.tensorfold_sha])
        log(f"Adding a worktree of {src} at {s.tensorfold_sha[:8]} under {s.build_dir}")
        _run(["git", "-C", str(src), "worktree", "add", "--detach", str(s.build_dir), s.tensorfold_sha])
    else:
        log(f"Cloning {s.tensorfold_repo} at {s.tensorfold_sha[:8]} into {s.build_dir}")
        _run(["git", "clone", "--no-checkout", s.tensorfold_repo, str(s.build_dir)])
        _run(["git", "-C", str(s.build_dir), "checkout", "--detach", s.tensorfold_sha])
    return s.build_dir


def ensure_binary(s: Settings, rebuild: bool = False) -> Path:
    """The binary to serve with: `binary` as given, or the pinned sha built once (zig build native)."""
    if s.binary:
        return s.run_binary
    if s.built_binary.is_file() and not rebuild:
        return s.built_binary
    src = ensure_source(s)
    if not s.zig.is_file():
        raise RuntimeError(f"zig={s.zig} is not a file: install Zig {(src / '.zig-version').read_text().strip()} there "
                           "or set `zig` in local.toml")
    want = (src / ".zig-version").read_text().strip()
    have = _run([str(s.zig), "version"]).strip()
    if have != want:
        raise RuntimeError(f"{s.zig} is Zig {have}; the checkout pins {want}")
    log(f"Building tensorfold-native at {s.tensorfold_sha[:8]} (zig build native -Dcpu={s.zig_cpu}); a few minutes")
    _run([str(s.zig), "build", "native", f"-Dcpu={s.zig_cpu}"], cwd=src, timeout=3600)
    if not s.built_binary.is_file():
        raise RuntimeError(f"the build left no {s.built_binary}")
    log(f"Built {s.built_binary}: {_run([str(s.built_binary), '--version']).strip()}")
    return s.built_binary


def info(s: Settings, model_dir: Path) -> str:
    """`tensorfold-native info <dir>`: family, format, weight bytes and the memory floor."""
    return _run([str(ensure_binary(s)), "info", str(model_dir)], timeout=300)
