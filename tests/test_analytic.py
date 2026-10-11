"""The analytic check: at zero load the simulated step equals compute + latency + bytes / bandwidth."""

from __future__ import annotations

import pytest

from training_transit_sim import analytic
from training_transit_sim.engine import simulate
from training_transit_sim.model import VARIANTS, Config, plan


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_simulated_step_equals_closed_form(variant, topo, module):
    p = plan(Config(variant, topo, module=module))
    r = simulate(p, syncs=3)
    assert r.step_time == pytest.approx(analytic.step_time(p), rel=1e-12)
    assert p.schedule.sent_by(0) == pytest.approx(analytic.wire_per_worker(p), rel=1e-12)


@pytest.mark.parametrize("variant", ["adamw", "topk", "band-switch"])
@pytest.mark.parametrize("H", [1, 10, 100])
def test_sync_every_h_steps(variant, H, topo):
    p = plan(Config(variant, topo, sync_every=H))
    assert simulate(p, syncs=2).step_time == pytest.approx(analytic.step_time(p), rel=1e-12)
    assert analytic.step_time(p) == pytest.approx(p.compute_s + analytic.sync_time(p) / H, rel=1e-12)


@pytest.mark.parametrize("variant", ["adamw", "band", "band-switch"])
def test_low_background_load_stays_close(variant):
    from training_transit_sim.hardware import Topology
    p = plan(Config(variant, Topology("2x4", workers=8, dcs=2)))
    r = simulate(p, syncs=200, background=0.001, seed=1)
    assert r.step_time >= analytic.step_time(p) * (1 - 1e-12)          # queueing only ever adds
    assert r.step_time == pytest.approx(analytic.step_time(p), rel=0.01)


def test_heavy_background_load_adds_queueing():
    from training_transit_sim.hardware import Topology
    p = plan(Config("adamw", Topology("2x4", workers=8, dcs=2)))
    assert simulate(p, syncs=100, background=0.6).step_time > 1.2 * analytic.step_time(p)


def test_flat_ring_closed_form_by_hand():
    from disagg_sim.hardware import LINKS

    from training_transit_sim.hardware import Topology
    p = plan(Config("adamw", Topology("1x8", workers=8)))
    lk = LINKS["ib-ndr"]
    hand = p.compute_s + 14 * (lk.latency + 31_666_944 / 8 / lk.bandwidth)
    assert simulate(p).step_time == pytest.approx(hand, rel=1e-12)
