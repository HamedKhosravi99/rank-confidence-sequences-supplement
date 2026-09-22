"""Turn the tinyBenchmarks release of Open LLM Leaderboard per-example results into item matrices.

Source: https://github.com/felipemaiapolo/tinyBenchmarks, file tutorials/data/lb.pickle
(Maia Polo et al., "tinyBenchmarks: evaluating LLMs with fewer examples", ICML 2024). It holds,
for 395 models of the Open LLM Leaderboard (snapshot January 2024), the per-item correctness on
six scenarios: MMLU (57 subjects, 5-shot), HellaSwag (10-shot), ARC-Challenge (25-shot),
Winogrande (5-shot), GSM8K (5-shot), TruthfulQA-MC (0-shot; a score in [0, 1], not binary).

Download (90 MB), then run this script from code/:

    curl -L -o ../data/raw/tinybenchmarks/lb.pickle \
      https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tutorials/data/lb.pickle
    python -m data.prepare_tinybenchmarks

Output: ../data/processed/<scenario>.npz with ``scores`` (items x models, float32 in [0, 1]) and
``models`` (names), plus ../data/processed/manifest.json with hashes and shapes. The pickle is
loaded with an unpickler that admits numpy array reconstruction only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import re
from pathlib import Path

import numpy as np

RAW = Path("../data/raw/tinybenchmarks/lb.pickle")
OUT = Path("../data/processed")
EXPECTED_SHA256 = "34f44d6a819512ef74d00a95288d252fa679288a10ca167cd97fdbc3aae66437"
SCENARIOS = {
    "mmlu": r"^harness_hendrycksTest_.*_5$",
    "hellaswag": r"^harness_hellaswag_10$",
    "arc": r"^harness_arc_challenge_25$",
    "winogrande": r"^harness_winogrande_5$",
    "gsm8k": r"^harness_gsm8k_5$",
    "truthfulqa": r"^harness_truthfulqa_mc_0$",
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy") and name in {"ndarray", "dtype", "_reconstruct", "_frombuffer", "scalar"}:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def main() -> None:
    raw = RAW.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"lb.pickle has sha256 {digest}, expected {EXPECTED_SHA256}")
    d = _NumpyOnlyUnpickler(__import__("io").BytesIO(raw)).load()
    models = [re.sub(r"^open-llm-leaderboard/details_", "", m) for m in d["models"]]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": digest, "n_models": len(models), "scenarios": {}}
    for name, pattern in SCENARIOS.items():
        keys = sorted(k for k in d["data"] if re.match(pattern, k))
        blocks = [d["data"][k]["correctness"] for k in keys]
        scores = np.concatenate(blocks, axis=0).astype(np.float32)
        if scores.shape[1] != len(models) or scores.min() < 0 or scores.max() > 1:
            raise SystemExit(f"{name}: unexpected shape or range")
        subject = np.concatenate([np.full(b.shape[0], i) for i, b in enumerate(blocks)]).astype(np.int16)
        np.savez_compressed(OUT / f"{name}.npz", scores=scores, models=np.array(models), subject=subject,
                            subject_names=np.array(keys))
        binary = bool(np.all((scores == 0) | (scores == 1)))
        manifest["scenarios"][name] = {
            "items": int(scores.shape[0]), "blocks": len(keys), "binary": binary,
            "accuracy_min": float(scores.mean(axis=0).min()), "accuracy_max": float(scores.mean(axis=0).max()),
        }
        print(f"{name:11s} items={scores.shape[0]:6d} blocks={len(keys):3d} binary={binary}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
