"""Bisection: a reported break-even point is the crossing, within the tolerance."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from disagg_sim.hardware import GB

from training_transit_sim.hardware import Module, Topology
from training_transit_sim.model import Config
from training_transit_sim.study import (bisect, evaluate, gap, knob_enob, knob_gflops, knob_inter_bw,
                                        measured_enob)

ROOT = Path(__file__).resolve().parent.parent
T2 = Topology("2x4", workers=8, dcs=2)
FREE = Module(gflops=99e9, transform_charged=False)


def test_bisect_finds_a_known_root():
    be = bisect(lambda x: x * x - 2.0, 0.1, 10.0, rel_tol=1e-10)
    assert be.x == pytest.approx(math.sqrt(2.0), rel=1e-9)
    assert bisect(lambda x: x - 3.0, 0.0, 10.0, log=False, rel_tol=1e-12).x == pytest.approx(3.0)


def test_bisect_reports_no_crossing():
    be = bisect(lambda x: x + 1.0, 0.1, 10.0)
    assert not be.found and be.f_lo > 0 and be.f_hi > 0


@pytest.mark.parametrize("x,y", [("band-switch", "adamw"), ("band-switch", "topk"), ("band-allreduce", "adamw")])
def test_bandwidth_crossing_is_the_crossing(x, y):
    f = gap(Config(x, T2, module=FREE), Config(y, T2, module=FREE), 3.158, knob_inter_bw)
    be = bisect(f, 0.1 * GB, 1000 * GB, rel_tol=1e-6)
    assert be.found
    assert f(be.x * (1 - 1e-5)) < 0 < f(be.x * (1 + 1e-5))     # X faster below, slower above


def test_gflops_crossing_is_the_crossing():
    slow = T2.with_inter(1 * GB)
    f = gap(Config("band-switch", slow, module=Module(gflops=99e9)), Config("adamw", slow), 3.158, knob_gflops)
    be = bisect(f, 1e9, 1e14, rel_tol=1e-6)
    assert be.found
    assert f(be.x * (1 - 1e-5)) > 0 > f(be.x * (1 + 1e-5))     # more budget, faster module


def test_enob_crossing_lands_on_a_measured_enob():
    slow = T2.with_inter(1 * GB)
    f = gap(Config("band-switch", slow, module=FREE), Config("adamw", slow, module=FREE), 3.158, knob_enob)
    be = bisect(f, 6.0, 12.0, log=False, rel_tol=1e-6)
    assert be.x == pytest.approx(8.0, abs=1e-4)
    assert math.isinf(f(7.99)) and f(8.0) < 0
    assert measured_enob(7.99) == 6 and measured_enob(8.0) == 8 and measured_enob(11.5) == 10


def test_enob_6_never_reaches_a_target():
    o = evaluate(Config("band-pluggable", T2, module=Module(enob=6)), 3.158)
    assert o.steps is None and math.isinf(o.time) and math.isinf(o.energy)


def test_recorded_breakevens_are_crossings():
    res = json.loads((ROOT / "examples" / "results.json").read_text())
    checked = 0
    for d in res["breakeven_bw"]:
        if d["bw"] is None or d["module"] != "free":
            continue
        cx = Config(d["x"], T2, module=FREE, exact_steps=d["exact_steps"])
        f = gap(cx, Config(d["y"], T2, module=FREE), 3.158, knob_inter_bw)
        assert (f(d["bw"] * (1 - 1e-3)) > 0) != (f(d["bw"] * (1 + 1e-3)) > 0)
        checked += 1
    assert checked >= 3
