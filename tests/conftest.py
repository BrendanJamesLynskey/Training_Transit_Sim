from __future__ import annotations

import pytest
from disagg_sim.hardware import LINKS

from training_transit_sim.hardware import WAN_LINKS, Module, Topology

TOPOS = [
    Topology("1x8", workers=8, dcs=1),
    Topology("2x4", workers=8, dcs=2),
    Topology("4x2", workers=8, dcs=4, inter=LINKS["eth-25g"]),
    Topology("8x1", workers=8, dcs=8),
    Topology("2x4-400g", workers=8, dcs=2, inter=WAN_LINKS["wan-400g"]),
    Topology("3x2", workers=6, dcs=3),
    Topology("1x1", workers=1, dcs=1),
]
MODULES = [Module(gflops=10e9), Module(gflops=99e9), Module(gflops=99e9, transform_charged=False),
           Module(gflops=99e9, planes=4), Module(gflops=99e9, caps_port=False)]


@pytest.fixture(params=TOPOS, ids=lambda t: t.name)
def topo(request):
    return request.param


@pytest.fixture(params=MODULES, ids=["low", "high", "free", "planes4", "nocap"])
def module(request):
    return request.param
