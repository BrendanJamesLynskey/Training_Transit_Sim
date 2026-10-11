# Training_Transit_Sim

A [SimPy](https://simpy.readthedocs.io/) discrete-event model of **data-parallel training across data centres**, with
the gradient compression computed **in transit** by a photonic link module
([arXiv:2608.21536](https://arxiv.org/abs/2608.21536), Kundu et al.). It is Part 4 of the
[Compute in Transit](https://github.com/BrendanJamesLynskey/LLM_Hub_Compute_in_Transit) series, and it turns the
convergence results of the [DeMo fixed-band experiment](https://github.com/BrendanJamesLynskey/DeMo_Fixed_Band_Experiment)
into **wall-clock time and energy to a target loss**.

## The question

The experiment measured how many steps each way of syncing gradients needs to reach a loss, and how many bytes each
sends. It could not say how long training takes, or what a link module that computes on the traffic would add. This
model asks: across one or more data centres, when does computing a fixed frequency band **in the link** help, when is
the module itself the bottleneck, and what would it need (bandwidth, compute budget, ENOB) to break even?

Four measured variants and two hypothetical ones:

| | Variant | Collective | Steps to a loss from |
|---|---|---|---|
| A | AdamW, dense BF16 gradients | ring all-reduce | A's runs (3 seeds) |
| B | DeMo top-k (2-D 64x64 chunks), on the GPU | all-gather | B's runs (3 seeds) |
| C | DeMo fixed band (1-D 256-point chunks), on the GPU | all-gather | C's runs (3 seeds) |
| E | the band computed by each worker's **pluggable** module | all-gather | E's runs at the module's ENOB and bit planes (1 seed each) |
| C' | C's band summed by a ring all-reduce, no module (*hypothetical*) | all-reduce | C's (assumed) |
| E' | the band computed and **summed at the switch**, each worker's own band looped back (*hypothetical*) | in-network sum | E's (assumed) |

Every worker sends the same band positions and DeMo averages before taking the sign, so summing the bands gives the
same update as gathering them: that is why C' and E' reuse measured step counts. They were not run. E' needs a
switch-placed module (streams must meet to be summed) and an explicit return path: the experiment's fix for the
optics' errors feeds back the values actually sent, which an all-gather returns to each sender but an all-reduce
does not, so the switch sends each worker its own optical band back.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # installs disagg-sim from git, pinned at 69ffd2b
.venv/bin/pytest                                             # the verification ladder
.venv/bin/python examples/results.py                         # regenerates examples/results.md and results.json
.venv/bin/python scripts/sweep_bandwidth.py --gbps 1 10 100   # CSV: time to target against WAN bandwidth
```

```python
from training_transit_sim.hardware import Module, Topology
from training_transit_sim.model import Config
from training_transit_sim.study import evaluate

o = evaluate(Config("band-switch", Topology("2x4", workers=8, dcs=2), module=Module(gflops=99e9)), target=3.158)
print(o.steps, o.step_time, o.time, o.result.hot_spot, o.energy_per_step)
```

## The model

* **Inputs** (`src/training_transit_sim/inputs.py`): the experiment's `results/tinystories-gpu/results.json` (payload
  and wire bytes per step) and `results.md` (setup, and the steps-to-loss ladder), at commit `b1f62d3`, vendored in
  `data/experiment/` with their SHA-256 (`scripts/fetch_inputs.py`). Bytes scale with the parameter count by each
  variant's measured bytes per parameter.
* **Topology** (`hardware.py`): W workers in D data centres. Each worker has its own port to its DC fabric
  (InfiniBand NDR by default); each DC has one WAN link to the next on a ring of DCs. Devices and link presets come
  from [Disaggregated_Inference_Sim](https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim)'s
  `disagg_sim.hardware` (`ACCELERATORS`, `LINKS`), imported, not copied. 400G and 800G WAN presets are added here.
* **Compute per step**: a roofline, 6 x parameters x tokens per worker / (device peak x MFU). MFU 0.40.
  The top-k and band variants add their forward transform on the GPU (FFTW's 2.5 n log2 n convention).
* **Collectives** (`collectives.py`), modelled separately and hierarchically (intra-DC, then inter-DC, then
  intra-DC): ring all-reduce, 2(N-1)/N x B per worker; ring all-gather, (N-1) x P per worker; and the switch's
  in-network sum with the loopback. Schedules are data: each op names the data it carries, and a checker replays them
  on sets of contributors (every block everywhere; every sum counted exactly once).
* **The module as a link kernel** (`hardware.Module`, `Kernel`): it declares operations per byte by type (FFT, add,
  multiply, MAC, floating-point MAC, modular, gather), latency, pJ/bit and placement (pluggable or switch). A stream
  goes through four pipelined stages and the slowest sets its rate, which the hot-spot report names:
  * **port**, 25 GB/s (QSFP56);
  * **optics**, 4 x 14 Gb/s NRZ, 7 GB/s by the series' own arithmetic from the paper: with binary input, a value
    quantised to *p* bits takes *p* optical passes (bit planes), so the planes set throughput as well as accuracy;
  * **compute budget**, 10 to 99 GFLOP/s (the series' reading of the paper's "tens of GFLOPs"); whether the optical
    transform is charged against it is not public, so it is a switch (`transform_charged`);
  * **FPGA stage**, its own rate: modular arithmetic runs in the FPGA driver logic, not the optics; here it also
    does the quantising, full-scale tracking and bit-plane recombination.

  A pluggable sits in its worker's data path, so its port also caps that worker's link.
* **Power**: 9.7 W static (the paper) plus 2.76 pJ/bit dynamic (the developer's website, 45 nm, excluding the laser)
  plus a laser term. Devices: disagg-sim's static power and pJ per FLOP. Links: each preset's pJ/bit.
* **Engine** (`engine.py`): every link direction and module is a `simpy.Resource`; a sync runs round by round. Optional
  Poisson background traffic on the WAN links queues with the sync, which the closed forms cannot see.
* **Break-even** (`study.py`): bisection over inter-DC bandwidth, module GFLOP/s and ENOB. ENOB enters only through
  the measured E runs (6, 8, 10, 12), as a step function, so its break-even lands on a measured ENOB.
* **Sync every H steps** (DiLoCo-style) is an option, and an extrapolation: the experiment synced every step.

## Results

Every number here is quoted from [`examples/results.md`](examples/results.md) (with `results.json` beside it).
Target loss 3.158, the 2-D fixed band's final loss: the one ladder loss that the dense, top-k, exact-band and optics
variants all reach. Module settings: **[low]** 10 GFLOP/s, transform charged; **[high]** 99 GFLOP/s, transform
charged; **[free]** transform not charged.

<!-- results:start -->
Time to 3.158 in seconds (excerpt of section 2; 8 workers, H100s, 15.8M parameters):

| Variant | Steps | 1 DC x 8 | 2 DC x 4, 100 GbE | 8 DC x 1, 25 GbE | 2 DC x 4, 8 Gb/s |
| --- | --- | --- | --- | --- | --- |
| A  AdamW, dense ring all-reduce | 400 | 0.70 | 1.63 | 7.40 | 13.28 |
| B  DeMo top-k, all-gather | 400 | 0.45 | 0.93 | 3.80 | 6.75 |
| C  fixed band on the GPU, all-gather | 2,667 | 2.24 | 3.85 | 13.51 | 23.27 |
| E  band in transit (pluggable), all-gather [low] | 1,800 | 59.01 | 60.03 | 66.12 | 73.14 |
| E  band in transit (pluggable), all-gather [high] | 1,800 | 7.77 | 8.78 | 14.88 | 21.89 |
| E  band in transit (pluggable), all-gather [free] | 1,800 | 6.08 | 7.10 | 13.19 | 20.21 |
| C' fixed band on the GPU, ring all-reduce (hypothetical) | 2,667 | 1.87 | 2.16 | 5.02 | 7.02 |
| E' band in transit, summed at the switch (hypothetical) [low] | 1,800 | 59.29 | 59.64 | 61.78 | 62.91 |
| E' band in transit, summed at the switch (hypothetical) [high] | 1,800 | 8.04 | 8.39 | 10.54 | 11.67 |
| E' band in transit, summed at the switch (hypothetical) [free] | 1,800 | 6.35 | 6.71 | 8.85 | 9.98 |

* **At the experiment's scale AdamW's step is mostly communication, and the band needs many more steps.** On 2 DC x 4, 100 GbE, compute is 12.1% of AdamW's step. Top-k reaches 3.158 in 0.93 s and AdamW in 1.63 s (400 steps each); the fixed band on the GPU needs 3.85 s (2,667 steps) and the band in transit (E, ENOB 8) 1,800 steps.
* **With the transform charged against the budget, the module is the bottleneck.** At 10 GFLOP/s the band in a pluggable takes 33.348 ms a step against 1.443 ms for the band on the GPU, and module compute holds 95.0% of it. The budget would need 140.0 GFLOP/s before the optics, not the budget, set the module's rate, which is above the series' bracket of 10 to 99 GFLOP/s.
* **A pluggable also caps its worker's port**: module port holds 13.6% of the [free] pluggable step, because a 25 GB/s QSFP56 module sits on a 50 GB/s port.
* **The pluggable all-gather never breaks even with AdamW or top-k** over 0.1 to 1,000 GB/s of WAN (section 7a): an all-gather of the band still sends 13,854,288 bytes per worker, 0.25 of AdamW's 55,417,152, while it needs 1,800 steps to AdamW's 400.
* **Summing in the network is where a fixed band could gain.** E' (band computed and summed at the switch, each worker's own band looped back) beats AdamW below 1.57 GB/s (12.5 Gb/s) of WAN and top-k below 0.461 GB/s (3.69 Gb/s), with E's steps and the transform not charged. With the exact band's steps (no hypothesis) the AdamW crossing falls to 0.83 GB/s (6.64 Gb/s).
* **The same summation without a module does better.** C' (the band computed on the GPU and ring all-reduced, no module) beats AdamW below 6.58 GB/s (52.7 Gb/s); with both on the exact band's steps, E' never beats C' in range. Only E's single-seed step advantage (the hypothesis) lets E' beat C' at all.
* **ENOB.** ENOB 6 never reaches a ladder loss. Where an in-transit variant wins at all, the break-even ENOB is 8 (section 7c): the lowest measured ENOB that trains.
* **Energy is mostly the devices'.** The compute column (each device's static power for the whole step plus its energy per FLOP) is the largest in every row of section 5; the modules add 7.2% of E' [free]'s energy per step on 2 DC x 4, 100 GbE, and E' [free] uses 8.87 kJ to 3.158 against 1.95 kJ for AdamW.
* Every figure above rests on one model size, 8 workers, single seeds for E, and the series' own optics noise model; larger models and DiLoCo-style syncing are extrapolations (section 10).
<!-- results:end -->

## What is illustrative

* the MFU (0.40) and the devices' power coefficients (from disagg-sim, illustrative there);
* the 400G and 800G WAN presets, and the 8 Gb/s WAN share used to find crossings;
* the module's laser power (2 W), FPGA rate (50 G operations/s) and FPGA work per value, latency (1 us), and the
  10 and 99 GFLOP/s bracket on "tens of GFLOPs"; the 7 GB/s optics rate is the series' arithmetic, not a figure the
  paper states;
* the top-k cost on the GPU (about 2 comparisons per value);
* the background-traffic model (Poisson, 1 MB transfers);
* every extrapolation: models other than 15.8M parameters, worker counts other than 8, and syncing every H > 1 steps.

## Limits

* One model size (15.8M parameters), 8 workers and 3,000 steps; steps to a loss are only measured there.
* The learning rate was tuned only on top-k and AdamW and reused for every fixed-band variant.
* Single seeds for E and F; that the repaired optics band needs fewer steps than the exact band is a **hypothesis**,
  so every in-transit result is also given with the exact band's steps.
* The optics noise model is the series' own, not measured on the module.
* Steps to a loss are resolved only to the experiment's evaluation interval (200 steps), as results.md rounds them.
* Compute and communication do not overlap, and a pluggable's processing and its collective run one after the other:
  both conservative for the variants they touch.
* C' and E' were not run; they assume summing the band gives the same update as gathering it.

## Verification ladder

`pytest` runs:

* unit tests against hand formulas: ring all-reduce and all-gather bytes, the hierarchical totals, the roofline, the
  module's stage times and energy; the model's wire bytes reproduce the experiment's wire column exactly;
* invariants: every schedule delivers (replayed on contributor sets; double counting and missing data are caught),
  bytes are conserved, no resource is above 100% busy even under background load, hot-spot shares sum to 1;
* the analytic check: at zero load the simulated step equals compute + latency + bytes / bandwidth (to 1e-12,
  every variant, topology and module setting), and stays within 1% at 0.1% background load;
* bisection: each reported break-even is the crossing (the sign of the gap flips across it);
* the inputs: the vendored files match the pinned commit's SHA-256, and `results.json` holds the pinned numbers.

GitHub Actions runs ruff (E, F, W), the tests on Python 3.10 and 3.12, and checks that `results.md` and
`results.json` regenerate unchanged. The `Jenkinsfile` (in the style of
[Rust_DES_Kernel](https://github.com/BrendanJamesLynskey/Rust_DES_Kernel)'s) does the same, with JUnit reports and an
optional nightly bandwidth sweep.

## Licence

MIT. See [LICENSE](LICENSE).
