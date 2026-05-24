#!/usr/bin/env python3
"""Tail a Kaggle kernel's live log from the web UI.

Kaggle's public CLI/API does not expose stdout while a script kernel is
running (`kaggle kernels logs` returns empty until the kernel exits). The
web UI does — it uses a session cookie that the API token can't replace,
and modern Chromium-family browsers store that cookie encrypted with a key
that lives in xdg-desktop-portal, retrievable only by the running browser.

This script drives a headless `thorium-browser` against a snapshot of the
user's logged-in profile, navigates to the kernel page, and dumps the
visible progress lines. Same binary as the user's normal Thorium = same
portal app-id = it can read the existing session cookies.

Usage:
    .venv/bin/python scripts/kaggle_live_log.py <kernel-slug-or-url>
                                                [--watch SECONDS]

Examples:
    .venv/bin/python scripts/kaggle_live_log.py \\
        dmitriisamoilenko/uigen-kaggle-gateb-skin-c64
    .venv/bin/python scripts/kaggle_live_log.py \\
        https://www.kaggle.com/code/owner/slug --watch 60

The script reuses a running headless Thorium on port 9222 if present so
repeated invocations are fast; otherwise it starts one against
`/tmp/thorium_kaggle_clone` (a copy of ~/.config/thorium that survives
the user's interactive browser staying open).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
import websocket  # type: ignore[import-untyped]


CDP_HOST = "127.0.0.1"
CDP_PORT = 9222
THORIUM_BIN = "/usr/bin/thorium-browser"
SRC_PROFILE = Path.home() / ".config/thorium"
CLONE_PROFILE = Path("/tmp/thorium_kaggle_clone")
HEADLESS_LOG = Path("/tmp/thorium_kaggle_headless.log")
PID_FILE = Path("/tmp/thorium_kaggle_headless.pid")


def _cdp_up() -> bool:
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect((CDP_HOST, CDP_PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _ensure_clone() -> None:
    """Create or refresh the profile clone.

    We only mirror Default (the active profile) and Local State. The
    clone is shallow-refreshed each call so the session cookies stay
    current with what the user's interactive Thorium has.
    """
    if not SRC_PROFILE.exists():
        sys.exit(f"thorium source profile not found at {SRC_PROFILE}")
    CLONE_PROFILE.mkdir(parents=True, exist_ok=True)
    src_default = SRC_PROFILE / "Default"
    dst_default = CLONE_PROFILE / "Default"
    if dst_default.exists():
        shutil.rmtree(dst_default)
    shutil.copytree(src_default, dst_default, symlinks=True, ignore_dangling_symlinks=True)
    # Local State holds the per-profile preferences (not strictly required
    # for cookies, but Chromium expects it for a healthy launch).
    src_ls = SRC_PROFILE / "Local State"
    if src_ls.exists():
        shutil.copy2(src_ls, CLONE_PROFILE / "Local State")
    # Singleton locks would prevent the headless instance from starting.
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"):
        p = dst_default / name
        if p.exists():
            p.unlink()


def _start_headless() -> None:
    """Launch headless Thorium with CDP enabled if not already up."""
    if _cdp_up():
        return
    _ensure_clone()
    if HEADLESS_LOG.exists():
        HEADLESS_LOG.unlink()
    proc = subprocess.Popen(
        [
            THORIUM_BIN,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            f"--user-data-dir={CLONE_PROFILE}",
            f"--remote-debugging-port={CDP_PORT}",
            "--remote-allow-origins=*",
        ],
        stdout=open(HEADLESS_LOG, "w"),
        stderr=subprocess.STDOUT,
        # Detach so this child outlives us; PID_FILE makes it killable.
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _cdp_up():
            return
        time.sleep(0.5)
    sys.exit(f"headless thorium failed to start; see {HEADLESS_LOG}")


def _resolve_url(kernel: str) -> str:
    if kernel.startswith("http"):
        return kernel
    # Accept either "owner/slug" or just "slug" (the latter shouldn't happen
    # in practice, but be permissive).
    return f"https://www.kaggle.com/code/{kernel}"


def _attach_new_target(url: str) -> "websocket.WebSocket":
    r = requests.put(f"http://{CDP_HOST}:{CDP_PORT}/json/new?{url}", timeout=10)
    r.raise_for_status()
    tgt = r.json()
    ws = websocket.create_connection(
        tgt["webSocketDebuggerUrl"], timeout=30,
        header=["Origin: http://127.0.0.1:9222"],
    )
    return ws


_mid = 0
def _send(ws, method, params=None):
    global _mid
    _mid += 1
    ws.send(json.dumps({"id": _mid, "method": method, "params": params or {}}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == _mid:
            return m


def fetch_progress_lines(kernel_url: str, settle_seconds: float = 15.0) -> list[str]:
    """Open a fresh tab on the kernel URL, wait for JS to render, return the
    progress lines deduplicated and in step order."""
    ws = _attach_new_target(kernel_url)
    try:
        _send(ws, "Page.enable")
        _send(ws, "Runtime.enable")
        time.sleep(settle_seconds)
        # Pull every visible string from the DOM, then filter to lines that
        # look like trainer progress (start banner, step lines, end banner).
        expr = (
            "Array.from(document.querySelectorAll('*'))"
            ".map(e => e.innerText || '')"
            ".join('\\n')"
            ".split('\\n')"
            ".filter(t => /(\\[step\\s+\\d+\\/\\d+|training start:|trained V7Completer)/.test(t))"
            ".join('\\n')"
        )
        res = _send(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True})
        text = res.get("result", {}).get("result", {}).get("value", "") or ""
    finally:
        ws.close()
    return _dedupe(text.splitlines())


_STEP_RE = re.compile(r"\[step\s+(\d+)/(\d+)")


def _dedupe(lines: list[str]) -> list[str]:
    """Kaggle's UI splits the log across overlapping containers, producing
    triplicated step lines. Keep one entry per step number, preserving
    non-step lines."""
    seen_steps: set[int] = set()
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            if step in seen_steps:
                continue
            seen_steps.add(step)
        # de-dupe banner lines exactly
        if line in out:
            continue
        out.append(line)
    out.sort(key=_sort_key)
    return out


def _sort_key(line: str) -> tuple[int, int]:
    """Sort step lines by step number, banners come first/last by content."""
    if line.startswith("training start:"):
        return (0, 0)
    if line.startswith("trained V7Completer"):
        return (2, 0)
    m = _STEP_RE.search(line)
    if m:
        return (1, int(m.group(1)))
    return (1, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="Kaggle kernel as owner/slug or full URL")
    parser.add_argument("--watch", type=int, default=0,
                        help="If >0, repeat every N seconds (Ctrl-C to stop)")
    parser.add_argument("--settle", type=float, default=15.0,
                        help="Seconds to wait after page load for JS render")
    args = parser.parse_args()

    _start_headless()
    url = _resolve_url(args.kernel)

    def once() -> None:
        lines = fetch_progress_lines(url, settle_seconds=args.settle)
        if not lines:
            print("(no progress lines visible yet)")
            return
        for line in lines:
            print(line)
        # also surface the last step's deltas
        steps = [int(_STEP_RE.search(l).group(1))
                 for l in lines if _STEP_RE.search(l)]
        if steps:
            print(f"\n# {len(steps)} step lines, latest step {max(steps)}")

    once()
    while args.watch > 0:
        time.sleep(args.watch)
        print(f"\n--- {time.strftime('%H:%M:%S')} refresh ---")
        once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
