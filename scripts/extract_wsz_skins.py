#!/usr/bin/env python3
"""Extract canonical per-component BMP dirs from raw .wsz skins (skins_raw/).

A .wsz is a ZIP of Winamp BMPs (case-insensitive names). For each skin we read
the 11 trainable BMPs and normalize each to its canonical TRAINABLE_EXPORT_SPECS
size using the SAME validated rule as scripts/21_unpack_atlases_to_skin_dirs.py
(`_normalize_to_canonical`: crop-larger from top-left, pad-smaller with zeros,
else reject) — no new normalizer, no atlas round-trip. Skins missing a required
BMP, or where a BMP can't be brought to canonical / is too degenerate, are
SKIPPED (recorded in the manifest), never silently mangled.

Output (gen-compatible skin dirs, same layout as data_v7_16skin_completion):
    <out>/<skin_id>/<NAME>.bmp        11 canonical-size BMPs
    <out>/<skin_id>/_meta.json        per-file action trace (exact|cropped|padded)
    <out>/_manifest.json              per-skin accepted/skipped + reasons

Usage:
    python scripts/extract_wsz_skins.py --out data_v10_skins_all \
        --limit 0 --progress-every 100              # all skins
    python scripts/extract_wsz_skins.py --out data_v10_skins200 \
        --limit 200 --seed 0                        # diverse 200-skin subset
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS  # noqa: E402

# Reuse the EXACT validated normalizer from 21_unpack (digit-prefixed module).
_spec = importlib.util.spec_from_file_location(
    "_unpack21", REPO / "scripts" / "21_unpack_atlases_to_skin_dirs.py")
_unpack21 = importlib.util.module_from_spec(_spec)
sys.modules["_unpack21"] = _unpack21  # so the module's dataclasses resolve __module__
_spec.loader.exec_module(_unpack21)
_normalize_to_canonical = _unpack21._normalize_to_canonical

SPECS = [(s.file_name, s.w, s.h) for s in TRAINABLE_EXPORT_SPECS]


def _skin_id(wsz: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", wsz.stem.lower()).strip("_") or "skin"


def _extract_one(wsz: Path, out_root: Path, min_frac: float) -> dict:
    sid = _skin_id(wsz)
    try:
        z = zipfile.ZipFile(wsz)
    except Exception as e:  # noqa: BLE001
        return {"skin": wsz.name, "skin_id": sid, "status": "skipped", "reason": f"bad zip: {e}"}
    names = {n.split("/")[-1].lower(): n for n in z.namelist()}
    slots, normalized = [], {}
    for fn, cw, ch in SPECS:
        key = fn.lower()
        if key not in names:
            return {"skin": wsz.name, "skin_id": sid, "status": "skipped",
                    "reason": f"missing {fn}"}
        try:
            im = Image.open(io.BytesIO(z.read(names[key]))).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return {"skin": wsz.name, "skin_id": sid, "status": "skipped",
                    "reason": f"unreadable {fn}: {e}"}
        rw, rh = im.size
        if (rw * rh) < min_frac * (cw * ch):
            return {"skin": wsz.name, "skin_id": sid, "status": "skipped",
                    "reason": f"{fn} too small {rw}x{rh} < {min_frac} of {cw}x{ch}"}
        arr = np.asarray(im, dtype=np.uint8)
        norm, action = _normalize_to_canonical(arr, cw, ch, pad_smaller=True, crop_larger=True)
        if norm is None:
            return {"skin": wsz.name, "skin_id": sid, "status": "skipped",
                    "reason": f"{fn} normalize failed: {action}"}
        normalized[fn] = norm
        slots.append({"file_name": fn, "canonical_wh": [cw, ch],
                      "content_wh": [rw, rh], "action": action})
    # all 11 ok -> write
    dst = out_root / sid
    dst.mkdir(parents=True, exist_ok=True)
    for fn, arr in normalized.items():
        Image.fromarray(arr, "RGB").save(dst / fn)
    (dst / "_meta.json").write_text(json.dumps(
        {"skin_id": sid, "source_wsz": wsz.name, "slots": slots}, indent=2))
    return {"skin": wsz.name, "skin_id": sid, "status": "accepted",
            "actions": {s["file_name"]: s["action"] for s in slots}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skins-raw", default="skins_raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Max ACCEPTED skins (0 = all).")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed for diverse sampling.")
    ap.add_argument("--min-content-fraction", type=float, default=0.5)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    wszs = sorted(Path(args.skins_raw).glob("*.wsz"))
    random.Random(args.seed).shuffle(wszs)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    print(f"extract_wsz_skins: {len(wszs)} candidate .wsz -> {out_root} "
          f"(limit={args.limit or 'all'} min_frac={args.min_content_fraction})", flush=True)

    manifest, accepted, skipped = [], 0, 0
    t0 = time.time()
    seen_ids: set[str] = set()
    for i, wsz in enumerate(wszs):
        rec = _extract_one(wsz, out_root, args.min_content_fraction)
        if rec["status"] == "accepted":
            if rec["skin_id"] in seen_ids:  # dedupe id collisions
                rec = {**rec, "status": "skipped", "reason": "dup skin_id"}
            else:
                seen_ids.add(rec["skin_id"]); accepted += 1
        if rec["status"] != "accepted":
            skipped += 1
        manifest.append(rec)
        if args.progress_every and (i + 1) % args.progress_every == 0:
            dt = time.time() - t0
            print(f"[{i+1}/{len(wszs)}] accepted={accepted} skipped={skipped} "
                  f"{(i+1)/max(dt,1e-9):.0f} skins/s elapsed={dt/60:.1f}min", flush=True)
        if args.limit and accepted >= args.limit:
            print(f"reached limit {args.limit} accepted at candidate {i+1}", flush=True)
            break
    (out_root / "_manifest.json").write_text(json.dumps(
        {"accepted": accepted, "skipped": skipped, "records": manifest}, indent=2))
    print(f"DONE: accepted={accepted} skipped={skipped} -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
