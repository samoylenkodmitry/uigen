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
    """Create a fully isolated profile clone.

    Hard-resets CLONE_PROFILE on every call. We deep-copy Default and
    Local State without following symlinks: a prior crashed run can
    leave behind Singleton* symlinks that point into the *interactive*
    Thorium's IPC socket directory (/tmp/.org.chromium.Chromium.XXXXXX).
    If we preserve those symlinks, the headless instance follows them
    and starts talking to the user's main Thorium instead of standing
    up its own — which crashes both. Wiping the whole clone every
    launch is cheap (a couple seconds, ~100 MB) and removes the entire
    class of problem.
    """
    if not SRC_PROFILE.exists():
        sys.exit(f"thorium source profile not found at {SRC_PROFILE}")
    if CLONE_PROFILE.exists():
        shutil.rmtree(CLONE_PROFILE)
    CLONE_PROFILE.mkdir(parents=True)
    src_default = SRC_PROFILE / "Default"
    dst_default = CLONE_PROFILE / "Default"
    # symlinks=False -> Python materializes symlink targets into real
    # files / dirs. Any Singleton* symlinks in src_default would resolve
    # to dangling targets and be skipped (since the user's main Thorium
    # owns those sockets); we don't want to bring them across anyway.
    def _ignore_singletons(_dir, names):
        return [n for n in names if n.startswith("Singleton") or n == "lockfile"]
    shutil.copytree(
        src_default, dst_default,
        symlinks=False,
        ignore=_ignore_singletons,
        ignore_dangling_symlinks=True,
    )
    src_ls = SRC_PROFILE / "Local State"
    if src_ls.exists():
        shutil.copy2(src_ls, CLONE_PROFILE / "Local State")


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
            # Be a quiet, non-interactive Chromium: no first-run UI, no
            # default-browser prompts. We deliberately do NOT pass
            # --password-store=basic or --use-mock-keychain — those would
            # block the portal key lookup we need for cookie decryption
            # (same-binary -> same portal app-id -> session cookies
            # decrypt). Isolation comes from a fresh user-data-dir each
            # launch (see _ensure_clone) plus the IPC socket dir being
            # derived from the user-data-dir path, so the headless's
            # socket can never collide with the user's main Thorium.
            "--no-first-run",
            "--no-default-browser-check",
        ],
        stdout=open(HEADLESS_LOG, "w"),
        stderr=subprocess.STDOUT,
        # Detach so this child outlives us; PID_FILE makes it killable.
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    # First-launch can take >30s on a cold cache because we just wiped
    # the clone and Chromium has to rebuild HSTS state / extension stubs.
    deadline = time.monotonic() + 60
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
            ".filter(t => /(\\[step\\s+\\d+\\/\\d+|training start:|trained V7Completer|\\bby_(mode|file|skin):)/.test(t))"
            ".join('\\n')"
        )
        res = _send(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True})
        text = res.get("result", {}).get("result", {}).get("value", "") or ""
    finally:
        ws.close()
    return _dedupe(text.splitlines())


_STEP_RE = re.compile(r"\[step\s+(\d+)/(\d+)")
_BREAKDOWN_RE = re.compile(r"^by_(mode|file|skin):")


def _dedupe(lines: list[str]) -> list[str]:
    """Kaggle's UI splits the log across overlapping containers, producing
    triplicated step lines. We dedupe by step number and group each step's
    breakdown sub-lines (by_mode / by_file / by_skin) under their owning
    step. Banner lines (training start / trained V7Completer) bracket the
    output."""
    step_lines: dict[int, str] = {}
    step_breakdowns: dict[int, list[str]] = {}
    banners_start: list[str] = []
    banners_end: list[str] = []
    current_step: int | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            current_step = step
            if step not in step_lines:
                step_lines[step] = line
            continue
        if _BREAKDOWN_RE.match(line):
            if current_step is None:
                continue
            slot = step_breakdowns.setdefault(current_step, [])
            if line not in slot:
                slot.append(line)
            continue
        if line.startswith("training start:"):
            if line not in banners_start:
                banners_start.append(line)
            continue
        if line.startswith("trained V7Completer"):
            if line not in banners_end:
                banners_end.append(line)
            continue
    out: list[str] = []
    out.extend(banners_start)
    for step in sorted(step_lines):
        out.append(step_lines[step])
        for b in step_breakdowns.get(step, []):
            out.append("  " + b)
    out.extend(banners_end)
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
