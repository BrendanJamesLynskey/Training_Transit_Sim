"""The experiment's measured inputs: bytes per step and the steps-to-loss ladder.

Source: DeMo_Fixed_Band_Experiment at commit ``EXPERIMENT_COMMIT`` (``results/tinystories-gpu/``), vendored in
``data/experiment/`` by ``scripts/fetch_inputs.py``. Nothing here is typed in by hand:

* ``results.json`` gives each configuration's payload and wire bytes per worker per step;
* ``results.md`` gives the setup (parameters, workers, sequences, context) in its header, and the steps-to-loss
  ladder in section 1 (``### Steps to reach a loss``), which results.json does not hold.

Steps to a loss are the first evaluation (every 200 steps) at or below it, mean over seeds, and only when every
seed reaches it. They were measured at 15.8M parameters with 8 workers and 3,000 steps: anything else is an
extrapolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

EXPERIMENT_REPO = "BrendanJamesLynskey/DeMo_Fixed_Band_Experiment"
EXPERIMENT_COMMIT = "b1f62d394ac79250df9ce555d3a55d85cf1cd517"
FILES = ("results.json", "results.md")
DATA = Path(__file__).resolve().parent.parent.parent / "data" / "experiment"

# The experiment's labels for the configurations this model uses (exactly as results.md prints them).
LABEL_ADAMW = "A  AdamW, all-reduce"
LABEL_TOPK = "B  DeMo, top-k, 2-D 64x64 chunks"
LABEL_BAND = "C  DeMo, fixed band, 1-D 256-point chunks"   # the exact band that E emulates


def label_optics(enob: int = 8, planes: int = 8) -> str:
    """The repaired optics runs: full scale re-set every 100 steps, feedback of the band sent."""
    return (f"E  DeMo, band via emulated optics, 1-D 256-point chunks, ENOB {enob}, {planes} planes, "
            "full scale re-set every 100 steps, feedback of the band sent")


ENOBS = (6, 8, 10, 12)          # measured at 8 planes
PLANES = (4, 8, 12)             # measured at ENOB 8


@dataclass(frozen=True)
class Setup:
    params: int
    workers: int
    seqs_per_worker: int
    context: int
    steps: int

    @property
    def tokens_per_worker(self) -> int:
        return self.seqs_per_worker * self.context


@dataclass(frozen=True)
class Experiment:
    setup: Setup
    rows: dict              # label -> results.json row
    targets: list           # [(loss, letter)], highest loss first, as the ladder's columns
    ladder: dict            # label -> {loss: steps or None}

    def row(self, label: str) -> dict:
        return self.rows[label]

    def steps_to(self, label: str, target: float) -> float | None:
        """Mean steps to reach ``target`` (a ladder column, to 3 decimals); None if not every seed reached it."""
        return self.ladder[label][round(target, 3)]

    def bytes_per_param(self, label: str) -> float:
        """Payload bytes per parameter per step: what scales with the model size."""
        return self.rows[label]["payload_per_step"] / self.setup.params


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def parse_setup(md: str) -> Setup:
    m = re.search(r"Model: ([\d,]+) parameters, .*?context (\d+)\. (\d+) simulated workers, (\d+) sequences each "
                  r"per step, (\d+) steps", md)
    if not m:
        raise ValueError("results.md header: setup line not found")
    return Setup(params=_num(m[1]), context=int(m[2]), workers=int(m[3]), seqs_per_worker=int(m[4]),
                 steps=int(m[5]))


def parse_ladder(md: str) -> tuple[list, dict]:
    """The '### Steps to reach a loss' table: columns 'to 3.158 (C)', cells '2,667 (5.28 GB)' or '–'."""
    sec = md.split("### Steps to reach a loss", 1)[1].split("\n## ", 1)[0]
    lines = [ln for ln in sec.splitlines() if ln.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    targets = []
    for c in head[1:]:
        m = re.fullmatch(r"to (\d+\.\d+) \((\w)\)", c)
        targets.append((float(m[1]), m[2]))
    ladder = {}
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        row = {}
        for (loss, _), cell in zip(targets, cells[1:]):
            m = re.match(r"([\d,]+) \(", cell)
            row[round(loss, 3)] = float(_num(m[1])) if m else None
        ladder[cells[0]] = row
    return targets, ladder


@lru_cache(maxsize=None)
def load(data: Path = DATA) -> Experiment:
    js = json.loads((data / "results.json").read_text())
    md = (data / "results.md").read_text()
    targets, ladder = parse_ladder(md)
    rows = {r["label"]: r for r in js["rows"]}
    return Experiment(setup=parse_setup(md), rows=rows, targets=targets, ladder=ladder)
