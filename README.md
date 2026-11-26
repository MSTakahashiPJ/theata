# THeatA: Transport- and Heating-aware QEC Simulation for 1D QCCD Trapped-Ion Systems

A compiler and simulator for syndrome measurement circuits on a one-dimensional QCCD (quantum charge-coupled device) architecture.
It compiles surface-code syndrome measurement circuits into transport-level instruction sequences, tracks the motional heating of every ion through transport and elapsed time, and evaluates logical error rates through a Stim/PyMatching noise pipeline whose gate error rates depend on the tracked phonon numbers.
Cooling strategies (threshold-triggered, predictive, and LIZ-resident coolants) are configurable compiler policies.

This repository accompanies the paper:

> *THeatA: Transport- and Heating-aware QEC simulation with Cooling Strategies for QCCD Trapped-Ion Systems* (2026). [arXiv:2611.30676](https://arxiv.org/abs/2611.30676)

## Quickstart

```bash
pip install -r requirements.txt
```

Two guided notebooks are the best entry points:

- `notebooks/hem3_demo.ipynb` — the whole pipeline in one short run: compile a d = 3
  surface-code circuit, watch per-ion phonon tracking, toggle a cooling policy with one
  config line, and measure the logical error rate.
- `notebooks/qec_simulation_walkthrough.ipynb` — how to bring your own QEC code,
  with the Steane [[7,1,3]] code generated end-to-end from its stabilizer list via Stim.

## Project Layout

```
src/                  Compiler core
circuits/             Input circuit definitions
experiments/          Experiment scripts
results/              Experiment outputs
plots/                Plot outputs
ThesisImages/         Paper and poster assets
```

## Using the Compiler

### Basic Workflow

```python
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from parallel_estimator import ParallelEstimator
from validator import validate_operations

# 1. Parse the circuit and build the DAG
program = parse(circuit_module.CIRCUIT)
dag = DAGBuilder(program, stabilizer_parallel=False).build()

# 2. Architecture and initial placement
arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=20, temp_capacity=20)
initial = {0: 1, 1: 1, 2: 5, ...}  # ion_id -> zone_id

# 3. Compile
config = CompilerConfig(eviction_strategy="balanced")
compiler = QCCDCompiler(arch, config=config)
result = compiler.compile(dag, initial)

# 4. Use the result (see below)
```

### Structure of CompileResult

`compiler.compile()` returns a `CompileResult` with the following fields:

| Field | Type | Description |
|-----------|-----|------|
| `timesteps` | `List[TimeStep]` | Parallelized instruction sequence (Scheduler output) |
| `flat_ops` | `List[Operation]` | Sequential instruction sequence before parallelization (MovementPlanner output) |
| `operations` | `List[Operation]` | Flattened view of `timesteps` (backward-compatibility property) |
| `stats` | `Dict[str, int]` | Counts per operation type (gates, splits, merges, ...) |
| `phonon_data` | `Dict` | Phonon-tracking results |
| `final_placement` | `Dict[int, int]` | Ion placement at the end of the round (ion_id -> zone_id) |
| `final_zone_contents` | `Dict[int, List[int]]` | Ion ordering inside each zone at the end of the round |
| `config` | `CompilerConfig` | Compiler configuration that was used |

### Working with CompileResult

#### Estimating Execution Time

```python
from parallel_estimator import ParallelEstimator

est = ParallelEstimator()
r = est.estimate(result.timesteps)
print(f"Parallel time: {r.total_time / 1000:.2f} ms")
print(f"Sequential time: {r.sequential_time / 1000:.2f} ms")
print(f"Speedup: {r.speedup:.2f}x")
```

#### Validating the Instruction Sequence

```python
from validator import validate_operations, validate_timesteps

# Sequential validation
seq_result = validate_operations(result.operations, initial, arch)

# Parallel validation (includes zone-conflict and ion-conflict checks)
par_result = validate_timesteps(result.timesteps, initial, arch)
```

#### Text Output (Human-Readable Instruction Sequence)

```python
from parallel_estimator import save_parallel_txt

save_parallel_txt(result.timesteps, "output.txt",
                  initial_placement=initial, arch=arch)
```

Output format (one operation per line; parallel operations expanded as a block):
```
TS   0 [  100.00μs  @    100.00μs]  Split(ion8) z1->z2
TS   1 [  100.00μs  @    200.00μs]  [2P]
  | Split(ion9) z5->z4                    100.00μs
  | Swap(0,1) z1                           20.00μs
```

#### Reading Phonon Numbers

```python
pd = result.phonon_data
print(f"Max axial phonon: {pd['max_axial']:.2f}")
print(f"Max transversal phonon: {pd['max_transversal']:.2f}")

# Phonon numbers at each MS gate
for gate_info in pd['at_gates']:
    print(f"{gate_info['gate']}: {gate_info['phonons']}")
```

#### Carrying the Placement Across Rounds

```python
# Round 1
result1 = compiler.compile(dag1, initial)

# Round 2: preserve the ion ordering of final_zone_contents
ordered_placement = {}
for zone_id in sorted(result1.final_zone_contents.keys()):
    for ion_id in result1.final_zone_contents[zone_id]:
        ordered_placement[ion_id] = zone_id

result2 = compiler.compile(dag2, ordered_placement)
```

#### Carrying Phonon Numbers Across Rounds

By default each round starts from a fresh `QCCDState`, so phonon numbers and the
planning clock are reset to zero at every round boundary -- the paper setting,
which makes the reported heating a lower bound. Pass `initial_phonons` and
`initial_clock_us` to `compile()` to carry the previous round's final values
over instead. Both default to `None`, which reproduces the reset behaviour
exactly.

```python
# Round 2: carry the phonon numbers and the clock over from round 1
pd1 = result1.phonon_data

result2 = compiler.compile(dag2, ordered_placement,
                           initial_phonons={
                               'axial': pd1['per_ion_axial'],
                               'transversal': pd1['per_ion_transversal'],
                               'transversal2': pd1['per_ion_transversal2'],
                           },
                           initial_clock_us=pd1['planning_clock_us'])
```

The two arguments are independent, but with a non-zero `heating_rate_*_per_s`
you normally want both: carrying only the phonon numbers restarts the planning
clock at `t = 0`, so the `time_us` recorded for every gate -- and anything
derived from it, such as a per-zone dwell time -- is cut at the round boundary.

The same option is exposed as `--carry-phonons` in
`experiments/exp_distance_scan.py` and as
`compile_rounds(carry_phonons=True)` in `hackathon/make_test_circuits.py`.

### CompilerConfig Options

| Option | Type | Default | Description |
|-----------|-----|-----------|------|
| `gate_selection` | str | `"min_id"` | Gate selection strategy |
| `memory_reorder` | bool | `False` | Lookahead reordering inside Memory during Measure |
| `measure_batch_size` | int | `1` | Measure batch size (2+ enables batching) |
| `interleave` | bool | `False` | Interleaved stabilizer execution |
| `eviction_strategy` | str | `"balanced"` | Eviction strategy out of the LIZ (`"balanced"` or `"nearest"`) |
| `cool_threshold_transversal` | float? | `None` | Cool the gate chain before an MS gate when its transverse phonon number exceeds this threshold |
| `cool_ancilla_after_measure_us` | float? | `None` | Deep re-cooling of the measurement ancilla right after each Measure (duration in μs) |
| `cool_ancilla_measure_interval` | int | `1` | Apply the post-measurement cooling once every N measurements |
| `merge_cool_threshold_transversal` | float? | `None` | Merge-averaging cooling: shuttle a cold ancilla into a hot memory chain when its mean phonon number exceeds this threshold |
| `predictive_cool_threshold_transversal` | float? | `None` | Predictive trigger (two-pass compile): insert cooling cycles when the predicted phonon number of upcoming data qubits exceeds this threshold |
| `predictive_cool_max_iters` | int | `1` | Maximum number of cooling cycles per trigger point |
| `predictive_cool_actuator` | str | `"ancilla"` | Which ion moves in a cooling cycle: the ancilla into the chain (`"ancilla"`) or the hot data qubit to the LIZ (`"data"`) |
| `resident_coolant_ids` | list? | `None` | LIZ-resident coolant ions; visiting data qubits cool passively by merge averaging |
| `coolant_recool_us` | float? | `None` | Batch re-cooling of the auxiliary ions (coolants + ancilla) whenever the LIZ holds only auxiliaries |
| `planner` | str | `"v1"` | Movement planner: `"v1"` (baseline) or `"v2"` (per-zone state machine with JIT staging) |
| `prefetch_depth` | int | `1` | v2: how many next-demand ions to stage ahead during a measurement window |
| `v2_sort_horizon` | int | `4` | v2: how many upcoming ions each memory zone pre-sorts toward the LIZ end |
| `v2_sort_trigger` | str | `"measure"` | v2: when pre-sorting is emitted (`"measure"`, `"delivery"`, or `"both"`) |

## Modules in src/

| File | Role |
|---------|------|
| `qccd_parser.py` | Parser for the custom DSL (built on Lark) |
| `ast_def.py` | Type definitions for operations (GateOp, SplitOp, ShuttleOp, CoolOp, ...) |
| `dag_builder.py` | Quantum circuit → DAG conversion, stabilizer detection, hook-error edge injection |
| `qccd_compiler.py` | MovementPlanner + Scheduler + QCCDState with phonon tracking and cooling policies (compiler core) |
| `planner_v2.py` | Alternative movement planner (per-zone state machine with JIT staging) |
| `per_liz_compiler.py` | Multi-LIZ compiler (per-LIZ instruction streams with planned synchronization) |
| `spatial_partitioner.py`, `stabilizer_partitioner.py` | Stabilizer-to-LIZ partitioning and placement for multi-LIZ configurations |
| `estimator.py` | Sequential execution-time estimation + TimeParameters definition |
| `parallel_estimator.py` | Parallel execution-time estimation + text output |
| `event_scheduler.py` | Opt-in event-driven time model (alternative to lock-step) |
| `sim_params.py` | Externalized parameter set (timing, phonon, and cooling values loaded from JSON) |
| `pauli_lookup.py` | Phonon-dependent MS-gate error lookup table with rounding modes |
| `ion_timeline.py` | Per-ion absolute-time timelines and idle-gap accounting for dephasing |
| `stim_recompiler.py` | Reconstruction of a multi-round Stim circuit from the compiled sequence |
| `validator.py` | Constraint validation of instruction sequences (sequential + parallel) |
| `formatter.py` | Shared text formatting of operations and timesteps |
| `memory_reorder_optimizer.py` | Ion-reordering optimization inside Memory during Measure |
| `css_code_utils.py` | Utilities for general CSS codes (circuit generation support) |
| `zone_visualizer.py` | Zone-state visualization |

## Circuit File Format

Each `.py` file in `circuits/` carries the following metadata and circuit string:

```python
QUBIT_COUNT = 10
DATA_QUBITS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ANCILLA_QUBITS = [9]

CIRCUIT = """
QUBIT 0 AT (0, 0)
...
RESET 9
R 0 1.5708 1.5708
MS 0 9 0.25
...
MEASURE 9
"""
```

Circuits for d=5 and d=7 can be generated with `d5_generator.py`:
```bash
cd circuits
python d5_generator.py 5 7    # generates reuse1, reuse2, and reverse_cnot for d5/d7
```

## Architecture

Seven-zone one-dimensional layout: `T(20) - M(20) - T(20) - L(4) - T(20) - M(20) - T(20)`

- T: Temporary zone (staging area during transport)
- M: Memory zone (ion storage)
- L: LIZ (laser interaction zone, where gates execute)

The accompanying paper does not distinguish the staging zones and denotes the same
layout `M-M-M-L-M-M-M`; the T/M distinction here is an internal detail of the compiler
(`QCCDArchitecture.simple_mlm`).

## Experiment Scripts

Experiment scripts live in `experiments/` and analyze the compiler output in various ways.
Each script accepts command-line options (see `--help`).

The main current scripts:

| File | Purpose |
|---------|------|
| `exp_cooling_distance_scan.py` | Distance scan (d = 3/5/7) with cooling policies, scaling-ratio (Λ) readout, and the phonon-number-cap sweep |
| `exp_cooling_pl.py` | Logical error rate of individual cooling policies |
| `exp_distance_scan.py` | Distance scan without cooling (also hosts the circuit registry); `--carry-phonons` keeps the phonon numbers and the clock across round boundaries |
| `exp_multi_round.py` | Multi-round compilation with placement carry-over (also hosts the greedy-alternating placement) |
| `exp_noise_injection.py` | Phonon-based Stim noise channels (library used by the scripts above) |
| `exp_multi_liz_pl.py` | Logical error rate of multi-LIZ (2-LIZ) compilations |
| `exp_per_liz_stream.py` | Per-LIZ stream compilation experiments |
| `exp_planner_v2.py` | Movement-planner v1/v2 comparisons |
| `exp_placement_bridge.py`, `exp_verify2_general.py` | Integration experiments with the external 1D qubit-mapping library (placement and ordering comparisons) |
| `generate_*.py` | Circuit generators (surface and general CSS codes) |

Earlier exploratory scripts (`exp_parallel.py`, `exp_placement.py`, `exp_interleave.py`, `exp_memory_reorder.py`, ...) are kept for reference.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
