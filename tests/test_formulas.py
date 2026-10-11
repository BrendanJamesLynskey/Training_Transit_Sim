"""Unit tests against hand formulas: ring all-reduce, all-gather, roofline, module time and energy."""

from __future__ import annotations

import math

import pytest
from disagg_sim.hardware import ACCELERATORS, LINKS

from training_transit_sim import inputs
from training_transit_sim.collectives import allgather, allreduce
from training_transit_sim.engine import simulate
from training_transit_sim.hardware import Module, Topology, band_kernel, sum_kernel
from training_transit_sim.model import Config, plan

B = 31_666_944.0


@pytest.mark.parametrize("n", [2, 3, 4, 8, 16])
def test_ring_allreduce_sends_2_n_minus_1_over_n(n):
    s = allreduce(Topology("t", workers=n), B)
    for w in range(n):
        assert s.sent_by(w) == pytest.approx(2 * (n - 1) / n * B, rel=1e-12)
    assert len(s.rounds) == 2 * (n - 1)


@pytest.mark.parametrize("n", [2, 3, 4, 8, 16])
def test_ring_allgather_sends_n_minus_1_payloads(n):
    P = 1_979_184.0
    s = allgather(Topology("t", workers=n), P)
    for w in range(n):
        assert s.sent_by(w) == (n - 1) * P


@pytest.mark.parametrize("W,D", [(8, 2), (8, 4), (8, 8), (6, 3), (12, 2)])
def test_hierarchical_totals_equal_the_flat_ring(W, D):
    t = Topology("t", workers=W, dcs=D)
    ar, ag = allreduce(t, B), allgather(t, 1000.0)
    for w in range(W):
        assert ar.sent_by(w) == pytest.approx(2 * (W - 1) / W * B, rel=1e-12)
        assert ag.sent_by(w) == pytest.approx((W - 1) * 1000.0, rel=1e-12)
    n = W // D
    wan_ar = sum(x.nbytes for x in ar.transfers() if x.link[0] == "wan")
    assert wan_ar == pytest.approx(D * 2 * (D - 1) / D * B, rel=1e-12)    # per DC: 2 (D-1)/D x B over its WAN link
    wan_ag = sum(x.nbytes for x in ag.transfers() if x.link[0] == "wan")
    assert wan_ag == pytest.approx(D * n * (D - 1) * 1000.0, rel=1e-12)   # per DC: n (D-1) payloads


def test_model_wire_reproduces_the_experiment():
    e = inputs.load()
    one = Topology("1x8", workers=8)
    for v, lab in (("adamw", inputs.LABEL_ADAMW), ("topk", inputs.LABEL_TOPK), ("band", inputs.LABEL_BAND),
                   ("band-pluggable", inputs.label_optics())):
        p = plan(Config(v, one))
        assert p.schedule.sent_by(0) == e.row(lab)["wire_per_step"]


def test_roofline_compute():
    t = Topology("t", workers=8, mfu=0.4)
    p = plan(Config("adamw", t))
    hand = 6 * 15_833_472 * (8 * 256) / (989e12 * 0.4)
    assert p.compute_s == pytest.approx(hand, rel=1e-12)
    pb = plan(Config("band", t))
    assert pb.compute_s == pytest.approx((6 * 15_833_472 * 2048 + 20 * 15_833_472) / (989e12 * 0.4), rel=1e-12)
    assert plan(Config("adamw", t, params=1e9)).compute_s == pytest.approx(6e9 * 2048 / (989e12 * 0.4), rel=1e-12)


def test_module_stage_times_by_hand():
    m = Module(gflops=10e9, planes=8)
    b = 31_666_944.0
    st = m.stage_times(b, band_kernel())
    assert st["module port"] == pytest.approx(b / 25e9)
    assert st["module optics"] == pytest.approx(b / 2 * 8 / 56e9)            # 8 bits per value at 4 x 14 Gb/s
    assert st["module compute"] == pytest.approx(b * (2.5 * 8 / 2) / 10e9)   # 10 FLOPs per BF16 byte
    assert st["FPGA"] == pytest.approx(b / 2 * (2 + 8 / 16) / 50e9)
    assert m.stream_time(b, band_kernel()) == (pytest.approx(1e-6 + b * 10 / 10e9), "module compute")
    free = Module(transform_charged=False)
    assert free.stage_times(b, band_kernel())["module compute"] == 0.0
    assert free.stream_time(b, band_kernel())[1] == "module optics"
    s = Module().stage_times(1000.0, sum_kernel(4))
    assert s["module optics"] == 0.0 and s["module compute"] == pytest.approx(1000 * 3 / 8 / 10e9)


def test_optics_rate_is_the_series_7_gb_s():
    assert Module().optics_rate == 7e9


def test_module_energy_by_hand():
    t = Topology("1x8", workers=8)
    m = Module(gflops=99e9)
    r = simulate(plan(Config("band-pluggable", t, module=m)))
    e = r.energy()
    dense = 2.0 * 15_833_472
    hand = 8 * (9.7 + 2.0) * r.step_time + 8 * dense * 8 * 2.76e-12
    assert e["module"] == pytest.approx(hand, rel=1e-12)
    acc = ACCELERATORS["h100"]
    assert e["compute"] == pytest.approx(8 * (acc.idle_w * r.step_time + acc.pj_per_flop * 1e-12 * r.plan.flops))
    wire = 7 * 1_979_184.0
    assert e["link"] == pytest.approx(8 * wire * 8 * LINKS["ib-ndr"].pj_per_bit * 1e-12, rel=1e-12)


def test_no_module_energy_without_a_module():
    r = simulate(plan(Config("adamw", Topology("2x4", workers=8, dcs=2))))
    assert r.energy()["module"] == 0.0


def test_unknown_inputs_rejected():
    with pytest.raises(ValueError):
        Config("nope", Topology("t"))
    with pytest.raises(ValueError):
        Topology("t", workers=8, dcs=3)
    with pytest.raises(KeyError):
        from training_transit_sim.model import steps_to_target
        steps_to_target(Config("band-pluggable", Topology("t"), module=Module(enob=7)), 3.158)
    assert math.isclose(Module().static_power, 11.7)
