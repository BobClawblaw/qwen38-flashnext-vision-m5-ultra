"""up / down / status / logs: one detached tensorfold-native process, a pid file, readiness on the API."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, engine, weights
from .config import Settings
from .log import log


def _get(url: str, timeout: float = 3) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ConnectionError, OSError):
        return None


def pid(s: Settings) -> int | None:
    try:
        p = int(s.pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(p, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return p


def port_busy(s: Settings) -> bool:
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{s.port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    return bool(out.strip())


def free_gib() -> float:
    """Free plus inactive pages, the memory a load can take without swapping (unified memory; no nvidia-smi here)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)\.", out)
        pages += int(m.group(1)) if m else 0
    return pages * page / 2**30


def gpu_readings() -> tuple[int | None, int | None]:
    """iogpu.wired_limit_mb (0 unset) and hw.memsize, None where sysctl does not say."""
    out = []
    for key in ("iogpu.wired_limit_mb", "hw.memsize"):
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        out.append(int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None)
    return out[0], out[1]


def memory_check(s: Settings, info_text: str) -> None:
    log(config.gpu_check(s, *gpu_readings()))
    m = re.search(r"memory floor:\s+([0-9.]+) GiB", info_text)
    floor = float(m.group(1)) if m else 0.0
    have = free_gib()
    if floor and have < floor + s.memory_margin_gib:
        raise RuntimeError(f"{have:.1f} GiB free, the engine's floor is {floor:.1f} GiB plus a {s.memory_margin_gib:g} GiB "
                           "margin (memory_margin_gib): close something or lower the margin")
    log(f"Memory: {have:.1f} GiB free, floor {floor:.1f} GiB")


def wait_ready(s: Settings, proc_pid: int, limit: int = 1800) -> None:
    log(f"Waiting for {s.url}/v1/models (log: {s.log_file})")
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            os.kill(proc_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f"the server exited; last lines of {s.log_file}:\n{tail(s, 30)}") from None
        models = _get(f"{s.url}/v1/models")
        if models and any(m.get("id") == s.served_name for m in models.get("data", [])):
            log(f"Ready in {time.time() - t0:.0f}s: {s.url}/v1 serves {s.served_name}")
            return
        time.sleep(2)
    raise RuntimeError(f"not ready after {limit}s; see {s.log_file}")


def command(s: Settings, download: bool = True) -> tuple[list[str], Path]:
    binary = engine.ensure_binary(s)
    snap = weights.ensure_snapshot(s, download=download)
    info_text = engine.info(s, snap)
    log("Engine: " + " ".join(line.strip() for line in info_text.strip().splitlines()))
    memory_check(s, info_text)
    return [str(binary)] + s.serve_args(snap), snap


def up(s: Settings, *, download: bool = True, foreground: bool = False) -> None:
    if pid(s):
        raise RuntimeError(f"already running (pid {pid(s)}): python3 -m mac down first")
    if port_busy(s):
        raise RuntimeError(f"port {s.port} is in use by another process")
    cmd, _ = command(s, download=download)
    s.tf_cache.mkdir(parents=True, exist_ok=True)
    s.state_dir.mkdir(exist_ok=True)
    env = s.server_env()
    log("Starting: " + " ".join(cmd))
    if foreground:                     # launchd: the server is the job
        os.execve(cmd[0], cmd, env)
    with open(s.log_file, "ab") as logf:
        logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=True)
    s.pid_file.write_text(str(p.pid))
    wait_ready(s, p.pid)


def down(s: Settings, grace: float = 30) -> None:
    p = pid(s)
    if not p:
        log("not running")
        s.pid_file.unlink(missing_ok=True)
        return
    log(f"Stopping pid {p}")
    os.kill(p, signal.SIGTERM)
    t0 = time.time()
    while pid(s) and time.time() - t0 < grace:
        time.sleep(0.5)
    if pid(s):
        log(f"still up after {grace:.0f}s, SIGKILL")
        os.kill(p, signal.SIGKILL)
        time.sleep(1)
    s.pid_file.unlink(missing_ok=True)
    log("stopped")


def status(s: Settings) -> str:
    p = pid(s)
    lines = [f"pid: {p or 'not running'}", f"url: {s.url}/v1"]
    health = _get(f"{s.url}/health")
    models = _get(f"{s.url}/v1/models")
    lines.append(f"health: {json.dumps(health) if health is not None else 'no answer'}")
    lines.append("models: " + (", ".join(m.get("id", "?") for m in models.get("data", [])) if models else "no answer"))
    if s.dashboard:
        stats = _get(f"{s.url}/stats")
        if stats:
            lines.append(f"stats: {json.dumps(stats)[:400]}")
    return "\n".join(lines)


def tail(s: Settings, n: int) -> str:
    try:
        lines = s.log_file.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return f"(no {s.log_file})"
    return "\n".join(lines[-n:])


def chat(s: Settings, prompt: str, max_tokens: int = 128) -> str:
    """One greedy chat completion, the smoke test after `up`."""
    body = json.dumps({"model": s.served_name, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{s.url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    if s.api_key:
        req.add_header("Authorization", f"Bearer {s.api_key}")
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read().decode())
    return json.dumps(out, indent=1)
