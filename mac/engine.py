"""The engine binary: a worktree of TensorFold at the pinned sha under tf_cache, built with Zig, or a given binary."""

from __future__ import annotations

import hashlib
import shutil
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


# libjpeg-turbo's release, the one the engine reads JPEG with (it refuses JPEG with any other), and its tarball's hash
JPEG_VERSION = "3.1.4.1"
JPEG_URL = f"https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/{JPEG_VERSION}/libjpeg-turbo-{JPEG_VERSION}.tar.gz"
JPEG_SHA256 = "ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022"

# libwebp's release, the one the engine reads WebP with (Pillow 13.0.0's; any other is refused), and its tarball's hash
WEBP_VERSION = "1.6.0"
WEBP_URL = f"https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-{WEBP_VERSION}.tar.gz"
WEBP_SHA256 = "e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564"

# xgrammar's release the engine's structured output is built against (zig/build/grammar.zig pins the tag), and its commit
XGRAMMAR_TAG = "v0.2.8"
XGRAMMAR_COMMIT = "97787376faee5ed8466cfad57c99855e4ce2f6aa"
XGRAMMAR_REPO = "https://github.com/mlc-ai/xgrammar"


def ensure_binary(s: Settings, rebuild: bool = False) -> Path:
    """The binary to serve with: `binary` as given, or the pinned sha built once (zig build native), with libtfgrammar,
    libtfvideo, libturbojpeg and libwebpdecoder beside it."""
    if s.binary:
        return s.run_binary
    if not s.built_binary.is_file() or rebuild:
        build(s)
    if rebuild or not s.built_grammar.is_file():
        build_grammar(s)
    if s.vision and s.video and (rebuild or not s.built_video.is_file()):
        build_video(s)
    if s.vision and not s.built_jpeg.is_file():
        build_jpeg(s)
    if s.vision and not s.built_webp.is_file():
        build_webp(s)
    return s.built_binary


def build_jpeg(s: Settings) -> None:
    """libturbojpeg in lib/ beside the binary, where the server looks: libjpeg-turbo built once into tf_cache."""
    prefix = s.tf_cache / f"libjpeg-turbo-{JPEG_VERSION}"
    lib = prefix / "lib" / "libturbojpeg.0.dylib"
    if not lib.is_file():
        work = s.tf_cache / "libjpeg-turbo-src"
        work.mkdir(parents=True, exist_ok=True)
        tarball = work / f"libjpeg-turbo-{JPEG_VERSION}.tar.gz"
        if not tarball.is_file():
            log(f"Downloading libjpeg-turbo {JPEG_VERSION}")
            _run(["curl", "-sfL", "-o", str(tarball), JPEG_URL], timeout=600)
        got = hashlib.sha256(tarball.read_bytes()).hexdigest()
        if got != JPEG_SHA256:
            tarball.unlink()
            raise RuntimeError(f"{JPEG_URL} has SHA-256 {got}, not the pinned {JPEG_SHA256}")
        _run(["tar", "xzf", str(tarball), "-C", str(work)])
        log(f"Building libjpeg-turbo {JPEG_VERSION} into {prefix}; a minute")
        _run(["cmake", "-S", str(work / f"libjpeg-turbo-{JPEG_VERSION}"), "-B", str(work / "build"),
              "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}"], timeout=600)
        _run(["cmake", "--build", str(work / "build"), "-j", "8"], timeout=1800)
        _run(["cmake", "--install", str(work / "build")], timeout=600)
        if not lib.is_file():
            raise RuntimeError(f"the libjpeg-turbo build left no {lib}")
    s.built_jpeg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lib.resolve(), s.built_jpeg)


def build_webp(s: Settings) -> None:
    """libwebpdecoder in lib/ beside the binary, where the server looks: libwebp built once into tf_cache."""
    prefix = s.tf_cache / f"libwebp-{WEBP_VERSION}"
    lib = prefix / "lib" / "libwebpdecoder.3.dylib"
    if not lib.is_file():
        work = s.tf_cache / "libwebp-src"
        work.mkdir(parents=True, exist_ok=True)
        tarball = work / f"libwebp-{WEBP_VERSION}.tar.gz"
        if not tarball.is_file():
            log(f"Downloading libwebp {WEBP_VERSION}")
            _run(["curl", "-sfL", "-o", str(tarball), WEBP_URL], timeout=600)
        got = hashlib.sha256(tarball.read_bytes()).hexdigest()
        if got != WEBP_SHA256:
            tarball.unlink()
            raise RuntimeError(f"{WEBP_URL} has SHA-256 {got}, not the pinned {WEBP_SHA256}")
        _run(["tar", "xzf", str(tarball), "-C", str(work)])
        log(f"Building libwebp {WEBP_VERSION} into {prefix}; a minute")
        tools = ["ANIM_UTILS", "CWEBP", "DWEBP", "GIF2WEBP", "IMG2WEBP", "VWEBP", "WEBPINFO", "WEBPMUX", "EXTRAS"]
        _run(["cmake", "-S", str(work / f"libwebp-{WEBP_VERSION}"), "-B", str(work / "build"), "-DCMAKE_BUILD_TYPE=Release",
              f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DBUILD_SHARED_LIBS=ON"] + [f"-DWEBP_BUILD_{t}=OFF" for t in tools], timeout=600)
        _run(["cmake", "--build", str(work / "build"), "-j", "8"], timeout=1800)
        _run(["cmake", "--install", str(work / "build")], timeout=600)
        if not lib.is_file():
            raise RuntimeError(f"the libwebp build left no {lib}")
    s.built_webp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lib.resolve(), s.built_webp)


def build_grammar(s: Settings) -> None:
    """libtfgrammar beside the binary (structured output): xgrammar cloned once into tf_cache, then zig build grammar."""
    src = ensure_source(s)
    root = s.tf_cache / f"xgrammar-{XGRAMMAR_TAG}"
    if not (root / ".git").exists():
        log(f"Cloning xgrammar {XGRAMMAR_TAG} with its submodules into {root}")
        _run(["git", "clone", "--branch", XGRAMMAR_TAG, "--recurse-submodules", XGRAMMAR_REPO, str(root)], timeout=1800)
    head = _run(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    if head != XGRAMMAR_COMMIT:
        raise RuntimeError(f"{root} is at {head[:8]}, not xgrammar {XGRAMMAR_TAG}'s {XGRAMMAR_COMMIT[:8]}; remove it to clone again")
    log(f"Building libtfgrammar (zig build grammar -Dxgrammar={root})")
    _run([str(s.zig), "build", "grammar", f"-Dxgrammar={root}", f"-Dcpu={s.zig_cpu}"], cwd=src, timeout=3600)
    if not s.built_grammar.is_file():
        raise RuntimeError(f"the build left no {s.built_grammar}")


def build_video(s: Settings) -> None:
    """libtfvideo beside the binary: FFmpeg 9.0.2 built once into tf_cache (the checkout's script), then zig build video."""
    src = ensure_source(s)
    prefix = s.tf_cache / "ffmpeg-9.0.2"
    if not (prefix / "lib" / "libavcodec.a").is_file():
        log(f"Building FFmpeg 9.0.2 (static, decoders only) into {prefix}; several minutes")
        _run(["bash", str(src / "tools" / "zig" / "build_ffmpeg.sh"), str(prefix), str(s.tf_cache / "ffmpeg-src")],
             cwd=src, timeout=7200)
    log(f"Building libtfvideo (zig build video -Dffmpeg={prefix})")
    _run([str(s.zig), "build", "video", f"-Dffmpeg={prefix}", f"-Dcpu={s.zig_cpu}"], cwd=src, timeout=3600)
    if not s.built_video.is_file():
        raise RuntimeError(f"the build left no {s.built_video}")


def build(s: Settings) -> None:
    """tensorfold-native at the pinned sha."""
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


def info(s: Settings, model_dir: Path) -> str:
    """`tensorfold-native info <dir>`: family, format, weight bytes and the memory floor."""
    return _run([str(ensure_binary(s)), "info", str(model_dir)], timeout=300)
