"""V10 packaging: infer_v10.py produces a loadable skin.wsz even with NO trained
expert checkpoints (every BMP falls back to the default skin). Real-Cranamp
render is skipped (--no-render) to keep the test fast and engine-independent."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_infer_packages_loadable_wsz_with_default_fallback(tmp_path: Path):
    # Pick any image that exists in the repo (we letterbox to 960x1728).
    image = ROOT / "assets/default_skin/MAIN.bmp"
    out = tmp_path / "v10_infer"
    res = subprocess.run(
        [sys.executable, str(ROOT / "infer_v10.py"),
         "--image", str(image), "--checkpoints", str(tmp_path / "no_ckpts"),
         "--out", str(out), "--device", "cpu", "--no-render"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr[-1000:]
    skin_wsz = out / "skin" / "skin.wsz"
    assert skin_wsz.exists(), out
    with zipfile.ZipFile(skin_wsz) as zf:
        names = {Path(n).name for n in zf.namelist()}
    # Every TRAINABLE BMP + at least PLEDIT.TXT must be packaged.
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
    for s in TRAINABLE_EXPORT_SPECS:
        assert s.file_name in names, (s.file_name, names)
    assert "PLEDIT.TXT" in names
    # No trained experts -> everything fell back to default.
    used = (out / "experts_used.json").read_text()
    assert "DEFAULT" in used
