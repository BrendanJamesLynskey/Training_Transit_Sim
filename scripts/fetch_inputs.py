"""Fetch the experiment's results at the pinned commit into data/experiment/.

    python scripts/fetch_inputs.py

The inputs are DeMo_Fixed_Band_Experiment's ``results/tinystories-gpu/results.json`` (wire bytes per step)
and ``results.md`` (the steps-to-loss ladder, which only results.md holds), both at commit ``EXPERIMENT_COMMIT``.
They are vendored so the tests and CI need no network; ``SOURCE.json`` records the commit and each file's
SHA-256, and ``tests/test_inputs.py`` checks the files against it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_transit_sim.inputs import DATA, EXPERIMENT_COMMIT, EXPERIMENT_REPO, FILES  # noqa: E402


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    sums = {}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/{EXPERIMENT_REPO}/{EXPERIMENT_COMMIT}/results/tinystories-gpu/{name}"
        body = urllib.request.urlopen(url, timeout=60).read()
        (DATA / name).write_bytes(body)
        sums[name] = hashlib.sha256(body).hexdigest()
        print(f"{name}: {len(body):,} bytes, sha256 {sums[name]}")
    (DATA / "SOURCE.json").write_text(json.dumps(
        {"repo": EXPERIMENT_REPO, "commit": EXPERIMENT_COMMIT, "path": "results/tinystories-gpu/",
         "sha256": sums}, indent=1) + "\n")


if __name__ == "__main__":
    main()
