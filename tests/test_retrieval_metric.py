from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks


REPO = Path(__file__).resolve().parents[1]


def _load_retrieval_module():
    path = REPO / "scripts" / "11_eval_slotnet_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_slotnet_retrieval", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _zero_files() -> dict[str, torch.Tensor]:
    return {
        spec.file_name: torch.zeros(1, 3, spec.h, spec.w)
        for spec in TRAINABLE_EXPORT_SPECS
    }


def test_retrieval_distance_ignores_unsupported_export_pixels():
    module = _load_retrieval_module()
    masks = load_support_masks()
    pred = _zero_files()
    target = _zero_files()
    pledit = next(spec for spec in TRAINABLE_EXPORT_SPECS if spec.file_name == "PLEDIT.bmp")
    off = (~masks["PLEDIT.bmp"]).nonzero(as_tuple=False)
    assert off.numel(), "PLEDIT should contain unsupported pixels"
    py, px = off[0].tolist()

    target["PLEDIT.bmp"][..., py, px] = 1.0

    assert torch.isclose(module.exported_mae(pred, target, masks), torch.tensor(0.0))

    on = masks["PLEDIT.bmp"].nonzero(as_tuple=False)
    py, px = on[0].tolist()
    target["PLEDIT.bmp"][..., py, px] = 1.0

    assert module.exported_mae(pred, target, masks) > 0
