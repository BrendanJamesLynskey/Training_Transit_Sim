"""Invariants: every schedule moves the right data, bytes are conserved, nothing is above 100% busy."""

from __future__ import annotations

import pytest

from training_transit_sim.collectives import (ScheduleError, Schedule, Transfer, allgather, allreduce,
                                              allreduce_keys, replay)
from training_transit_sim.engine import simulate
from training_transit_sim.model import VARIANTS, Config, plan


def _init(p):
    t = p.cfg.topo
    v = p.cfg.variant
    if v in ("adamw", "band-allreduce"):
        return {w: {k: frozenset({w}) for k in allreduce_keys(t)} for w in range(t.workers)}
    if v in ("topk", "band"):
        return {w: {("blk", w): frozenset({w})} for w in range(t.workers)}
    return {w: {("dense", w): frozenset({w})} for w in range(t.workers)}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_schedule_delivers(variant, topo, module):
    p = plan(Config(variant, topo, module=module))
    state = replay(p.schedule, _init(p))
    everyone = frozenset(range(topo.workers))
    for w in range(topo.workers):
        if variant in ("adamw", "band-allreduce"):
            assert all(state[w][k] == everyone for k in allreduce_keys(topo))
        elif variant in ("topk", "band", "band-pluggable"):
            assert all(state[w][("blk", o)] == frozenset({o}) for o in range(topo.workers))
        else:   # the switch: the global sum, and the worker's own band back for error feedback
            assert all(state[w][("s", x)] == everyone for x in range(topo.dcs))
            assert state[w][("band", w)] == frozenset({w})


def test_double_counting_is_caught():
    from training_transit_sim.hardware import Topology
    t = Topology("t", workers=4)
    s = allreduce(t, 1.0)
    bad = Schedule()
    bad.add("twice", [s.rounds[0], s.rounds[0]])        # the same reduce round twice
    init = {w: {k: frozenset({w}) for k in allreduce_keys(t)} for w in range(4)}
    with pytest.raises(ScheduleError):
        replay(bad, init)


def test_sending_missing_data_is_caught():
    s = Schedule()
    s.add("x", [[Transfer(0, 1, ("up", 0), 1.0, (("blk", 5),))]])
    with pytest.raises(ScheduleError):
        replay(s, {0: {}, 1: {}})


def test_incomplete_allgather_is_detected():
    from training_transit_sim.hardware import Topology
    t = Topology("t", workers=4)
    s = allgather(t, 1.0)
    s.phases[0] = (s.phases[0][0], s.phases[0][1][:-1])       # drop the last round
    state = replay(s, {w: {("blk", w): frozenset({w})} for w in range(4)})
    assert any(len(state[w]) < 4 for w in range(4))


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_bytes_conserved(variant, topo):
    p = plan(Config(variant, topo))
    r = simulate(p)
    sched = p.schedule
    nodes = {x.src for x in sched.transfers()} | {x.dst for x in sched.transfers()}
    assert sum(sched.sent_by(n) for n in nodes) == pytest.approx(sum(sched.received_by(n) for n in nodes), rel=1e-12)
    assert sum(r.bytes_on.values()) == pytest.approx(sum(x.nbytes for x in sched.transfers()), rel=1e-12)
    assert r.module_bytes == pytest.approx(sum(x.nbytes for x in sched.processes()), rel=1e-12)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("load", [0.0, 0.5, 0.95])
def test_no_resource_above_100_percent(variant, load, topo):
    r = simulate(plan(Config(variant, topo)), syncs=20, background=load)
    for kind in ("up", "down", "wan", "mod"):
        assert 0.0 <= r.utilisation(kind, background=True) <= 1.0 + 1e-12
    h = r.hot_spots()
    assert sum(h.values()) == pytest.approx(1.0, rel=1e-12)
    assert all(x >= 0 for x in h.values())
