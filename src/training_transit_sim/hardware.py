"""Links, devices, the in-transit module and the topology.

Devices (``ACCELERATORS``) and the intra-DC and Ethernet link presets (``LINKS``) are imported from
Disaggregated_Inference_Sim (``disagg_sim.hardware``, pinned at 69ffd2b), not copied, so the series cannot drift.
This file adds only what that simulator does not have:

* two WAN presets, 400G and 800G, **illustrative** (round numbers; latency and pJ/bit as the 100 GbE preset);
* ``Module``: a photonic link module as a link kernel (a photonic link module, Kundu et al., arXiv:2608.21536);
* ``Topology``: W workers in D data centres.

Module figures and where they come from:

* port rate 25 GB/s: QSFP56 (200 Gb/s), the module's host port in the paper;
* optics rate 56 Gb/s: four modulators at 14 GBaud with NRZ (binary) input, the paper's section II. 4 x 14 Gb/s is
  the series' own arithmetic (7 GB/s), not a figure the paper states. With NRZ input a value quantised to
  ``planes`` bits takes ``planes`` symbols (one optical pass per bit plane, as in the experiment's emulation);
* compute budget 10 to 99 GFLOP/s: the series' bracket on the abstract's "tens of GFLOPs", a reading, not a
  measurement. Whether the optical transform is charged against it is not public, so it is a switch;
* static power 9.7 W: the paper's module power;
* dynamic 2.76 pJ/bit: the developer's website ("estimated ... on 45nm silicon photonics, excluding laser");
* laser power, FPGA rate, FPGA work per value and latency: **illustrative**.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from disagg_sim.hardware import ACCELERATORS, GB, LINKS, Link

# ── WAN presets (illustrative) ─────────────────────────────────────────────────────────────────────────
WAN_LINKS = {
    "wan-400g": Link("400G WAN (illustrative)", bandwidth=50 * GB, latency=LINKS["eth-100g"].latency,
                     pj_per_bit=LINKS["eth-100g"].pj_per_bit),
    "wan-800g": Link("800G WAN (illustrative)", bandwidth=100 * GB, latency=LINKS["eth-100g"].latency,
                     pj_per_bit=LINKS["eth-100g"].pj_per_bit),
}
ALL_LINKS = {**LINKS, **WAN_LINKS}

MFU_DEFAULT = 0.40      # illustrative: achieved fraction of the device's peak BF16 rate in training


def link(name: str, bandwidth: float | None = None, latency: float | None = None) -> Link:
    """A preset by name, optionally with its bandwidth (bytes/s) or latency (s) replaced."""
    lk = ALL_LINKS[name]
    if bandwidth is not None:
        lk = replace(lk, bandwidth=bandwidth, name=f"{lk.name} at {bandwidth / GB:g} GB/s")
    if latency is not None:
        lk = replace(lk, latency=latency)
    return lk


# ── The in-transit module as a link kernel ────────────────────────────────────────────────────────────
OP_TYPES = ("fft", "add", "mul", "mac", "fp_mac", "modular", "gather")
OPTICAL_OPS = ("fft",)                  # done by the optical transform (charged or not: ``transform_charged``)
DIGITAL_OPS = ("add", "mul", "mac", "fp_mac")   # charged against the module's compute budget
FPGA_OPS = ("modular",)                 # the paper: modular arithmetic runs in the FPGA driver logic


@dataclass(frozen=True)
class Kernel:
    """What the module does to a stream: operations per input byte by type, plus the digital work in the FPGA
    stage per input value. ``optical`` = the stream goes through the optics (bit planes on the modulators)."""

    name: str
    ops_per_byte: dict = field(default_factory=dict)  # op type -> operations per input byte
    fpga_ops_per_value: float = 0.0                   # FPGA-stage operations per input value (BF16: 2 bytes)
    optical: bool = True
    bytes_per_value: float = 2.0                      # BF16 in

    def __post_init__(self):
        bad = set(self.ops_per_byte) - set(OP_TYPES)
        if bad:
            raise ValueError(f"unknown operation types {sorted(bad)}")


def band_kernel(n: int = 256, keep: float = 1 / 16, planes: int = 8) -> Kernel:
    """A 1-D length-n transform of each chunk, keeping a fixed band (a gather, no arithmetic).

    The transform costs 2.5 log2 n FLOPs per value (FFTW's convention, as the hub's ``cit_budget.py``), so
    1.25 log2 n per BF16 byte. The FPGA stage (illustrative) does the digital work around the optics:
    quantising each value to ``planes`` bits (1 multiply), tracking the full scale by a sum of squares
    (1 multiply-accumulate) and recombining the ``planes`` bit-plane outputs of each kept value."""
    log2n = n.bit_length() - 1
    return Kernel(name=f"band, 1-D {n}, keep {keep:g}, {planes} planes",
                  ops_per_byte={"fft": 2.5 * log2n / 2, "gather": 0.0},
                  fpga_ops_per_value=2.0 + planes * keep)


def sum_kernel(streams: int) -> Kernel:
    """Summing ``streams`` BF16 streams in the network: streams - 1 adds per output value, i.e. per input byte
    (streams - 1) / (2 streams). Digital arithmetic, no optics."""
    return Kernel(name=f"sum of {streams} streams", ops_per_byte={"add": (streams - 1) / (2 * streams)},
                  optical=False)


@dataclass(frozen=True)
class Module:
    """A photonic link module (arXiv:2608.21536) as a link kernel: rates, budget, power and placement.

    A stream of b bytes takes ``latency + b * max(1/port, planes/(optics bits/s * values), ops/budget, fpga)``:
    the stages are pipelined, so the slowest one sets the rate, and the hot-spot report names it."""

    placement: str = "pluggable"            # "pluggable" (at a worker's port) | "switch" (at a DC's edge switch)
    port_rate: float = 25 * GB              # QSFP56, bytes/s
    optics_bits_per_s: float = 4 * 14e9     # 4 modulators x 14 GBaud NRZ (the series' arithmetic: 7 GB/s)
    planes: int = 8                         # bits per value through the optics (one pass per bit plane)
    enob: int = 8                           # effective bits of the readout; sets convergence, not time
    gflops: float = 10e9                    # compute budget, FLOP/s (the series' bracket: 10 to 99 GFLOP/s)
    transform_charged: bool = True          # is the optical transform charged against ``gflops``? (not public)
    fpga_ops_per_s: float = 50e9            # illustrative
    latency: float = 1e-6                   # illustrative, per stream
    static_w: float = 9.7                   # the paper
    laser_w: float = 2.0                    # illustrative
    pj_per_bit: float = 2.76                # the developer's website, 45 nm, excluding the laser
    caps_port: bool = True                  # a pluggable is in the data path: its port caps its worker's link

    def __post_init__(self):
        if self.placement not in ("pluggable", "switch"):
            raise ValueError(f"unknown placement {self.placement!r}")

    @property
    def optics_rate(self) -> float:
        """Optics throughput in bytes/s of the optics' own bit stream (7 GB/s at the defaults)."""
        return self.optics_bits_per_s / 8

    def stage_times(self, nbytes: float, kernel: Kernel) -> dict:
        """Seconds each pipelined stage needs for ``nbytes`` of input (latency excluded)."""
        values = nbytes / kernel.bytes_per_value
        charged = sum(v for k, v in kernel.ops_per_byte.items()
                      if k in DIGITAL_OPS or (k in OPTICAL_OPS and self.transform_charged))
        fpga_ops = values * kernel.fpga_ops_per_value \
            + nbytes * sum(v for k, v in kernel.ops_per_byte.items() if k in FPGA_OPS)
        return {
            "module port": nbytes / self.port_rate,
            "module optics": values * self.planes / self.optics_bits_per_s if kernel.optical else 0.0,
            "module compute": nbytes * charged / self.gflops,
            "FPGA": fpga_ops / self.fpga_ops_per_s,
        }

    def stream_time(self, nbytes: float, kernel: Kernel) -> tuple[float, str]:
        """(seconds, binding stage) for one stream through the module."""
        st = self.stage_times(nbytes, kernel)
        stage = max(st, key=st.get)
        return self.latency + st[stage], stage

    def ops(self, nbytes: float, kernel: Kernel) -> float:
        """Operations of every type applied to ``nbytes`` (for the operations-per-byte ledger)."""
        return nbytes * sum(kernel.ops_per_byte.values()) + nbytes / kernel.bytes_per_value * kernel.fpga_ops_per_value

    @property
    def static_power(self) -> float:
        return self.static_w + self.laser_w


# ── Topology ──────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Topology:
    """W workers in D data centres, W/D in each. Every worker has its own port to the DC fabric (``intra``, one
    direction each way); each DC has one link to the next DC on a ring of DCs (``inter``, per direction)."""

    name: str
    workers: int = 8
    dcs: int = 1
    intra: Link = LINKS["ib-ndr"]
    inter: Link = LINKS["eth-100g"]
    device: str = "h100"
    mfu: float = MFU_DEFAULT

    def __post_init__(self):
        if self.workers % self.dcs:
            raise ValueError("workers must divide evenly into data centres")

    @property
    def per_dc(self) -> int:
        return self.workers // self.dcs

    @property
    def accelerator(self):
        return ACCELERATORS[self.device]

    def dc_of(self, w: int) -> int:
        return w // self.per_dc

    def with_inter(self, bandwidth: float) -> "Topology":
        return replace(self, inter=replace(self.inter, bandwidth=bandwidth))
