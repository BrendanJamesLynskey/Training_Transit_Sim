"""The inputs are the experiment's, at the pinned commit, and results.json was written from them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_transit_sim import inputs

ROOT = Path(__file__).resolve().parent.parent


def test_vendored_files_match_the_pinned_commit():
    src = json.loads((inputs.DATA / "SOURCE.json").read_text())
    assert src["commit"] == inputs.EXPERIMENT_COMMIT
    assert inputs.EXPERIMENT_COMMIT.startswith("b1f62d3")
    for name in inputs.FILES:
        assert hashlib.sha256((inputs.DATA / name).read_bytes()).hexdigest() == src["sha256"][name]


def test_setup_and_bytes_parsed():
    e = inputs.load()
    assert e.setup.params == 15_833_472
    assert (e.setup.workers, e.setup.seqs_per_worker, e.setup.context, e.setup.steps) == (8, 8, 256, 3000)
    assert e.row(inputs.LABEL_ADAMW)["payload_per_step"] == 31_666_944
    assert e.row(inputs.LABEL_TOPK)["wire_per_step"] == 27_705_720
    assert e.row(inputs.LABEL_BAND)["wire_per_step"] == 13_854_288
    assert e.bytes_per_param(inputs.LABEL_ADAMW) == 2.0
    assert e.bytes_per_param(inputs.LABEL_BAND) == 0.125


def test_ladder_parsed():
    e = inputs.load()
    assert [t for t, _ in e.targets] == [3.158, 3.124, 3.087, 1.785, 1.646]
    assert e.steps_to(inputs.LABEL_ADAMW, 3.158) == 400
    assert e.steps_to(inputs.LABEL_ADAMW, 1.785) == 2000
    assert e.steps_to(inputs.LABEL_ADAMW, 1.646) is None          # 1 of 3 seeds
    assert e.steps_to(inputs.LABEL_BAND, 3.158) == 2667
    assert e.steps_to(inputs.label_optics(10), 3.158) == 1600
    assert e.steps_to(inputs.label_optics(8, 4), 3.158) == 2600
    assert all(e.steps_to(inputs.label_optics(6), t) is None for t, _ in e.targets)


def test_results_json_written_from_pinned_numbers():
    res = json.loads((ROOT / "examples" / "results.json").read_text())
    e = inputs.load()
    assert res["experiment"]["commit"] == inputs.EXPERIMENT_COMMIT
    assert res["experiment"]["params"] == e.setup.params
    rows = {r["variant"]: r for r in res["inputs"]}
    assert rows["adamw"]["payload"] == e.row(inputs.LABEL_ADAMW)["payload_per_step"]
    assert rows["topk"]["experiment_wire"] == e.row(inputs.LABEL_TOPK)["wire_per_step"] == rows["topk"]["model_wire"]
    assert rows["band"]["model_wire"] == e.row(inputs.LABEL_BAND)["wire_per_step"]
    for lab, cells in res["ladder"].items():
        assert cells == [e.steps_to(lab, t) for t, _ in e.targets]
    assert res["enob_never"] == [6]


def test_readme_numbers_come_from_results_md():
    import re
    readme = (ROOT / "README.md").read_text()
    quoted = readme.split("<!-- results:start -->")[1].split("<!-- results:end -->")[0]
    md = (ROOT / "examples" / "results.md").read_text()
    nums = set(re.findall(r"\d[\d,]*\.?\d*", quoted))
    missing = [n for n in nums if n not in md]
    assert not missing, missing
