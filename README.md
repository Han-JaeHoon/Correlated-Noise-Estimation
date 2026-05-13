
# Surface Code Fault Estimation Dataset Generator

This project generates syndrome and final-measurement datasets for a 17-qubit \(d=3\) rotated surface code under controlled CNOT fault schedules.

The goal is to study how correlated CNOT faults propagate through repeated stabilizer circuits and to save the resulting data as CSV files for later inverse estimation or ML classification.

---

## 1. Project Structure

```text
surface_code_fault_estimation/
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── surface_code_layout.py
│   ├── logical_state.py
│   ├── fault_schedule.py
│   ├── stabilizer_circuit.py
│   ├── simulator.py
│   ├── postprocess.py
│   ├── dataset_generators.py
│   └── io_utils.py
│
├── scripts/
│   ├── debug_run.py
│   ├── generate_fixed_cnot_patterns.py
│   └── generate_all_cnot_single_fault.py
│
└── data/
    ├── mid_measure_reset/
    ├── mid_measure_no_reset/
    ├── no_mid_measure_final_sample/
    └── no_mid_measure_final_probs/
````

---

## 2. Installation

Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages.

```bash
pip install pennylane numpy
```

Optional packages:

```bash
pip install matplotlib jupyter pandas
```

---

## 3. Surface Code Layout

The simulator uses a (d=3) rotated surface code with:

```text
9 data qubits
8 ancilla qubits
17 total qubits
```

Wire assignment:

```text
Data qubits: 0, 1, ..., 8
Z ancilla:   9, 10, 11, 12
X ancilla:   13, 14, 15, 16
```

Stabilizer measurement convention:

```text
Z stabilizer:
    data -> ancilla CNOT

X stabilizer:
    H on ancilla
    ancilla -> data CNOT
    H on ancilla
```

The stabilizer order per round is:

```text
[Z0, Z1, Z2, Z3, X0, X1, X2, X3]
```

---

## 4. Fault Model

A fault is inserted immediately after a directed CNOT.

Default fault:

```text
CNOT(control -> target)
    followed by
X on control
Z on target
```

Example fault schedule:

```python
fault_schedule = [
    {
        "round": 0,
        "control": 0,
        "target": 9,
        "error_wires": [0, 9],
        "error_types": ["X", "Z"],
    }
]
```

This means:

```text
At round 0,
after CNOT(0 -> 9),
apply X on wire 0 and Z on wire 9.
```

CNOT direction matters. `CNOT(0 -> 9)` and `CNOT(9 -> 0)` are treated as different locations.

---

## 5. Simulation Modes

The simulator supports four output situations.

### 5.1 Mid-measurement with reset

```python
mid_measure=True
reset_after_measure=True
return_probs=False
```

This is closest to the standard surface-code cycle.

```text
stabilizer interaction
ancilla measurement
ancilla reset
next round
```

Output:

```text
syndrome_bits
detection_bits
```

---

### 5.2 Mid-measurement without reset

```python
mid_measure=True
reset_after_measure=False
return_probs=False
```

The ancilla is measured but not reset. It continues in the collapsed state.

Output:

```text
syndrome_bits
detection_bits
```

---

### 5.3 No mid-measurement, final sample

```python
mid_measure=False
return_probs=False
shots=1
```

No intermediate ancilla measurement or reset is performed. The repeated stabilizer circuit is applied coherently, and the ancilla wires are measured only at the end.

Output:

```text
final_bits
```

---

### 5.4 No mid-measurement, final probability distribution

```python
mid_measure=False
return_probs=True
shots=None
```

Same as above, but returns the final probability distribution over the 8 ancilla wires.

Output:

```text
probabilities
```

---

## 6. Configuration

Default experiment settings are stored in:

```text
src/config.py
```

Important defaults:

```python
DEFAULT_N_ROUNDS = 4
DEFAULT_SHOTS = 1

DEFAULT_CONTROL = 0
DEFAULT_TARGET = 9
DEFAULT_ERROR_TYPES = ("X", "Z")
DEFAULT_FAULT_ROUND = 0
DEFAULT_INCLUDE_EMPTY = True
```

You can either edit `src/config.py` or override values from the command line.

---

## 7. Quick Debug Run

From the project root:

```bash
python scripts/debug_run.py
```

This checks:

```text
1. mid-measurement with reset
2. no-mid-measurement final sample
```

---

## 8. Generate Dataset: Fixed CNOT, All Round Patterns

This fixes one directed CNOT and sweeps all possible fault occurrence patterns over rounds.

Example for `n_rounds = 4`:

```text
[]
[0]
[1]
[2]
[3]
[0,1]
[0,2]
...
[0,1,2,3]
```

Run with default config:

```bash
python scripts/generate_fixed_cnot_patterns.py
```

Override from CLI:

```bash
python scripts/generate_fixed_cnot_patterns.py \
  --n-rounds 6 \
  --control 0 \
  --target 9 \
  --error-types XZ \
  --mode no_mid_sample
```

Available modes:

```text
all
mid_reset
mid_no_reset
no_mid_sample
no_mid_probs
```

Example output:

```text
data/no_mid_measure_final_sample/fixed_cnot_0_9_XZ_rounds_6_all_round_patterns.csv
```

---

## 9. Generate Dataset: All CNOTs, Single Fault

This sweeps all directed CNOT locations. For each CNOT, it inserts one fault at a fixed round.

Run with default config:

```bash
python scripts/generate_all_cnot_single_fault.py
```

Override from CLI:

```bash
python scripts/generate_all_cnot_single_fault.py \
  --n-rounds 6 \
  --fault-round 2 \
  --error-types XZ \
  --mode mid_reset
```

The current stabilizer circuit contains 24 directed CNOT locations, so this dataset has 24 rows per mode.

---

## 10. Output Folders

Generated CSV files are saved under `data/` by simulation situation.

```text
data/
├── mid_measure_reset/
├── mid_measure_no_reset/
├── no_mid_measure_final_sample/
└── no_mid_measure_final_probs/
```

You can change the output root:

```bash
python scripts/generate_fixed_cnot_patterns.py \
  --output-root ./my_data \
  --n-rounds 5
```

---

## 11. CSV Format

Each row is one simulation sample.

Important columns:

```text
sample_id
dataset_type
mode
n_rounds
shots

mid_measure
reset_after_measure
return_probs

control
target
stab_type
stab_index
cnot_index

error_rounds
occurrence_vector
error_types
error_wires

syndrome_bits
detection_bits
final_bits
probabilities

fault_schedule_json
raw_output_json
```

---

## 12. Column Meaning

### `control`, `target`

The directed CNOT location where the fault is inserted.

Example:

```text
control = 0
target = 9
```

means:

```text
CNOT(0 -> 9)
```

---

### `error_rounds`

Rounds where the fault occurs.

Example:

```text
0,2
```

means the fault occurs in rounds 0 and 2.

---

### `occurrence_vector`

Binary vector over rounds.

Example for `n_rounds = 4`:

```text
1,0,1,0
```

means:

```text
round 0: fault
round 1: no fault
round 2: fault
round 3: no fault
```

---

### `error_types`

Pauli errors applied to `error_wires`.

Example:

```text
error_wires = 0,9
error_types = X,Z
```

means:

```text
X on wire 0
Z on wire 9
```

---

### `syndrome_bits`

Used when `mid_measure=True`.

Flattened syndrome sequence.

Logical shape before flattening:

```text
(shots, n_rounds, 8)
```

For `shots=1`, `n_rounds=4`, this contains 32 bits.

---

### `detection_bits`

Used when `mid_measure=True`.

Detection events are computed by:

```python
detection_events = syndrome[:, 1:, :] ^ syndrome[:, :-1, :]
```

Logical shape before flattening:

```text
(shots, n_rounds - 1, 8)
```

---

### `final_bits`

Used when:

```python
mid_measure=False
return_probs=False
```

Final ancilla measurement bitstring.

There are 8 ancilla wires, so this is an 8-bit string when `shots=1`.

---

### `probabilities`

Used when:

```python
mid_measure=False
return_probs=True
```

Final probability distribution over the 8 ancilla wires.

There are (2^8 = 256) probabilities.

---

## 13. Manual Python Example

```python
from src.fault_schedule import make_fixed_cnot_fault_schedule
from src.simulator import make_repeated_stabilizer_qnode
from src.postprocess import (
    reshape_mid_measure_syndrome,
    compute_detection_events,
    print_single_shot_sequence,
)

n_rounds = 4
shots = 1

qnode = make_repeated_stabilizer_qnode(
    n_rounds=n_rounds,
    shots=shots,
    mid_measure=True,
    reset_after_measure=True,
)

fault_schedule = make_fixed_cnot_fault_schedule(
    control=0,
    target=9,
    error_rounds=[0, 2],
    error_types=("X", "Z"),
)

raw = qnode(fault_schedule=fault_schedule)

syndrome = reshape_mid_measure_syndrome(
    raw=raw,
    n_rounds=n_rounds,
    shots=shots,
)

detection_events = compute_detection_events(syndrome)

print_single_shot_sequence(
    syndrome=syndrome,
    detection_events=detection_events,
    shot_idx=0,
)
```

---

## 14. Notes

* `shots=1` is usually enough for deterministic Clifford + Pauli fault signatures.
* Use `shots=None` when `return_probs=True`.
* `mid_measure=True, reset_after_measure=True` is the most surface-code-like setting.
* `mid_measure=False` is a toy setting for studying coherent propagation without intermediate syndrome measurement.
* CNOT direction is important throughout the code.
