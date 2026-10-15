"""Observer for the unmodified author IDGL implementation.

Run from the official IDGL/src directory. The only runtime compatibility
shim supplies PyYAML's now-required Loader argument. A trace observer saves
the exact final relation and full-node logits at the end of the official test
epoch; it does not change model computations. ``--evaluate-existing DIR``
loads a completed official checkpoint and runs its ordinary test method only.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


OUT_DIR = Path(os.environ["PAPER2_AUDIT_OUT"]).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
captured_seeds = set()


def _numpy(value):
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trace(frame, event, arg):
    if event != "return" or frame.f_code.co_name != "_run_whole_epoch":
        return _trace
    loc = frame.f_locals
    seed = int(loc["self"].config.get("seed", -1))
    if loc.get("mode") != "test" or seed in captured_seeds:
        return _trace
    captured_seeds.add(seed)
    loader = loc["data_loader"]
    path = OUT_DIR / f"idgl_seed-{seed}.npz"
    np.savez_compressed(
        path,
        relation=_numpy(loc.get("cur_adj")),
        raw_relation=_numpy(loc.get("cur_raw_adj")),
        input_relation=_numpy(loader.get("adj")),
        logits=_numpy(loc.get("output")),
        labels=_numpy(loader.get("labels")),
        idx_train=_numpy(loader.get("idx_train")),
        idx_val=_numpy(loader.get("idx_val")),
        idx_test=_numpy(loader.get("idx_test")),
        seed=np.asarray(seed),
    )
    print(f"[audit observer] saved official IDGL relation/logits: {path}")
    return _trace


def _install_observer():
    source_root = str(Path.cwd().resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from core.model_handler import ModelHandler

    original_test = ModelHandler.test

    def observed_test(self):
        sys.settrace(_trace)
        try:
            return original_test(self)
        finally:
            sys.settrace(None)

    ModelHandler.test = observed_test


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: capture_idgl.py CONFIG [--multi_run] [--evaluate-existing DIR]")
    config_path = str(Path(sys.argv[1]).resolve())
    multi_run = "--multi_run" in sys.argv[2:]
    evaluate_existing = None
    if "--evaluate-existing" in sys.argv[2:]:
        pos = sys.argv.index("--evaluate-existing")
        evaluate_existing = str(Path(sys.argv[pos + 1]).resolve())

    _install_observer()
    old_load = yaml.load
    yaml.load = lambda stream, Loader=yaml.FullLoader: old_load(stream, Loader=Loader)
    if evaluate_existing:
        from core.model_handler import ModelHandler
        from main import get_config, set_random_seed

        config = get_config(config_path)
        config["out_dir"] = None
        config["pretrained"] = evaluate_existing
        set_random_seed(int(config.get("random_seed", 1234)))
        handler = ModelHandler(config)
        handler.test()
    else:
        sys.argv = ["main.py", "-config", config_path]
        if multi_run:
            sys.argv.append("--multi_run")
        runpy.run_path("main.py", run_name="__main__")


if __name__ == "__main__":
    main()
