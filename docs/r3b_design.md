# R3b Decoder — Design Note

This note pins down the precise specification of the R3b ("Round-by-round
window decoder") scenario added in branch `decoder-add-analysis`. It exists
because the choice of decoder spec materially shapes what the experiment
actually tests, and the README §13.7 hypothesis is only meaningful relative
to a concrete decoder definition.

## 1. Setting

Same as R1 (README §12.1) except for the decoder:

| Aspect | R1 | R3b |
|---|---|---|
| Fault occurrence | Every round, every CNOT, Bernoulli | (same) |
| Per-CNOT rate | `p_bg`, one elevated `p_high` | (same) |
| Pauli per event | Uniform 15-Pauli pool (II excluded) | (same) |
| Observable | `(T, 8)` syndrome stream | `(T, 8)` syndrome stream |
| Target | Dominant CNOT id | (same) |
| **Decoder** | **IdentityDecoder (no-op)** | **LookupDecoder, window=1** |

The decoder consumes one round's detection event and applies a Pauli
correction to data qubits before the next round.

## 2. Why window_size = 1

The README §13.7 hypothesis is about whether *applying any correction at all*
breaks the per-round-distribution equivalence that creates the ambiguity
groups. Window size 1 is:

- the simplest reading of "window decoder"
- the one most directly tied to the hypothesis (each round's detection event
  is consumed immediately)
- the case where the decoder is essentially a single-fault hypothesis test on
  one round at a time

Larger windows are deferred to follow-up work; if window=1 already breaks the
X-stab interior groups, the hypothesis is confirmed in the strongest sense.

## 3. LookupDecoder specification

### 3.1 Inputs the decoder has

- The pre-computed **24 × 15 single-fault syndrome lookup**
  (`compute_single_round_lookup(reset=True)` from `src/ceiling.py`), which
  maps (CNOT k, Pauli α) → 8-bit syndrome `σ_{k,α}` for a logical-zero start.
- The current round's detection event `d ∈ F₂⁸` (raw round syndrome under
  reset mode is identical to the detection event since the previous round's
  syndrome was zero after correction; see §3.4).

### 3.2 Inverse table built at decoder construction

From the lookup we construct

```
inverse[d] = list of (k, α, residual_data_xz)  for every (k, α) with σ_{k,α} = d
```

where `residual_data_xz` is the (X-part, Z-part) Pauli on the 9 data qubits
that this single fault leaves at the **end** of the round it was injected in.
This is what the decoder needs to apply as a correction (negated; see §3.3).

The residual is precomputed for every (k, α) by Pauli-frame propagation
through the rest of the round's circuit (§4 below).

### 3.3 Round-by-round decode

For each round t:
1. Observe d_t = (round-t syndrome XOR round-(t-1) syndrome). Under reset
   mode with a clean decoder, the post-correction syndrome at round t-1 is
   zero, so d_t = round-t syndrome directly (see §3.4 for the precise claim).
2. If d_t == 0: apply identity correction; move on.
3. Else: look up `candidates = inverse[d_t]`.
   - If empty: no single-fault hypothesis explains d_t. Default policy:
     identity correction (treat as a multi-fault event we won't try to
     localize from one round; the residual remains uncorrected, becomes
     part of the frame for next round, possibly producing a non-zero
     detection event there that we *can* explain).
   - If non-empty: pick one candidate by tie-break (§3.5) and apply its
     `residual_data_xz` as the correction (this exactly cancels the residual
     in the Pauli frame if the single-fault hypothesis is correct).

### 3.4 Reset mode and "syndrome = detection event"

In reset mode with no decoder (R1), the round-t syndrome equals the
detection event because each round starts from the same ancilla state |0⟩
and the data-qubit Pauli frame accumulates only via faults. Under R3b with a
**perfect** correction (single-fault hypothesis correct), the post-round-t
frame is exactly cancelled by the round-t correction, so round-(t+1) starts
from logical-zero again, and the round-(t+1) syndrome is again equal to
round-(t+1)'s detection event. Under an **incorrect** correction, the frame
carries a residual into the next round, and the next round's syndrome
contains both the new fault's contribution and the old residual's. This is
exactly the mechanism the §13.7 hypothesis predicts will distinguish the
ambiguity-group members.

For simplicity and exactness, the runner will track the data-qubit frame
explicitly and pass it into each round's QNode call via `initial_error_list`.
The decoder receives the **round syndrome** (which is exactly the detection
event under reset mode), not a "decoder-corrected" detection event.

### 3.5 Tie-break policy

If multiple (k, α) candidates produce the same observed d_t, the decoder must
pick one residual to apply. We use the deterministic default agreed with the
user:

- **Lexicographic minimum (k, α)**.

Rationale: deterministic, reproducible, easy to reason about. The hypothesis
in §13.7 does not depend on the tie-break being "smart"; it predicts that
distinct true residuals (even under the same correction) will eventually
diverge in observable syndromes.

A weighted/probabilistic tie-break (e.g. by `p_high` prior on k, by Pauli
prior on α) is left as a follow-up after the unweighted version is
characterized.

### 3.6 Multi-fault rounds

A round with two or more faults will generally have a detection event that
is not in the lookup. The default policy (§3.3) is identity. This is a
conservative choice: it preserves frame state and lets subsequent rounds
gather more information rather than over-committing. At `p_bg ≈ 1e-3`,
multi-fault rounds are rare (~24·1e-3 = 2.4% per round), so this should
have small effect.

## 4. Pauli-frame propagation through a round

This is the core piece of new physics-faithful bookkeeping.

### 4.1 Round gate sequence (deterministic, no randomness)

From `src/stabilizer_circuit.py:stabilizer_round`:

```
for i in 0..3:                              # Z stabilizers, anc_z[i]
    for q in Z_stabilizers[i]:
        CNOT(q, anc_z[i])
        [possible fault at (q, anc_z[i])]
for i in 0..3:                              # X stabilizers, anc_x[i]
    H(anc_x[i])
    for q in X_stabilizers[i]:
        CNOT(anc_x[i], q)
        [possible fault at (anc_x[i], q)]
    H(anc_x[i])
measure_and_reset(all ancillas)
```

### 4.2 Symplectic propagation rules

State on 17 qubits represented as `(X, Z) ∈ F₂¹⁷ × F₂¹⁷`. Each gate updates
in-place:

- `CNOT(c, t)`: `X_t ^= X_c`, `Z_c ^= Z_t`.  (X spreads c→t, Z spreads t→c)
- `H(q)`: swap `X_q` and `Z_q`.
- `measure(q, reset=True)`: ancilla measurement outcome reads off `Z_q`
  (relative to the frame), then `X_q = Z_q = 0` (reset). We don't need the
  outcome here — only the post-round data-qubit frame.

### 4.3 Implementation plan

`src/pauli_frame.py` (new):
- `class PauliFrame`: `(x: np.ndarray[17], z: np.ndarray[17])` over F₂.
- `apply_cnot(frame, c, t)`, `apply_h(frame, q)`, `reset(frame, q)` —
  in-place updates.
- `from_pauli_string(types, wires)` → constructor for a single Pauli.

`src/round_propagation.py` (new):
- `propagate_single_fault(control, target, pauli_pair) -> (x_data, z_data)`:
  initialize an empty frame, run the round gate sequence, inject the fault
  at the matching CNOT, finish the round, return the data-qubit X and Z
  residuals at the end of the round (after H but before ancilla reset; reset
  doesn't touch data qubits).
- `build_single_fault_residual_lookup() -> np.ndarray (24, 15, 2, 9)`:
  precompute (k, α) → (x_data, z_data).

`src/decoder.py` (extend):
- `class LookupDecoder(Decoder)` with `window_size = 1`,
  `decode_window(d) -> Correction` per §3.

`src/sequence_runner.py` (extend §72):
- Window-by-window path that:
  1. Maintains a `PauliFrame` over 9 data qubits between rounds.
  2. For each round t: sample faults, call 1-round QNode with `initial_error
     _list = frame_to_pauli_strings(frame)`, get round syndrome.
  3. Pass syndrome to `decoder.decode_window(d_t)`, get `Correction`.
  4. Update frame: frame ⊕= (this round's faults' residuals) ⊕ correction.
     - Faults' residuals come from `build_single_fault_residual_lookup` keyed
       on (k, α) per fault entry.
     - The CIRCUIT already applied these faults in-line, so the syndrome
       reflects them; we add them to the frame for next-round bookkeeping.
  5. Decoder's correction has the opposite sign convention: applying it
     should cancel the residual, so frame ⊕= correction.
  6. Optionally also apply the correction to the CIRCUIT next round
     (i.e. include it in `initial_error_list` for round t+1), which is
     what physically happens.

> Subtlety: step 4 says we add the faults' residuals to the frame ourselves.
> But the circuit already applied those faults to a fresh logical-zero
> state, so the next-round circuit starting from a fresh logical-zero +
> `initial_error_list = (frame + residuals)` gives the correct physical
> state. The frame book-keeping is the *running record* of what residual is
> on the data qubits as we enter each new round's circuit.

### 4.4 Correctness check (planned in §5 of NIGHT_LOG)

Inject a single fault at round 0 only, with IdentityDecoder vs LookupDecoder,
and verify:
- With IdentityDecoder, the round-1 syndrome contains the propagated effect
  of the round-0 fault.
- With LookupDecoder (correct correction), the round-1 syndrome is zero.

This validates both the lookup inversion and the frame-propagation
bookkeeping.

## 5. What we expect to find

If §13.7 hypothesis holds:
- For the X-stab interior groups (G₂, G₃, G₄): R3b ceiling per-class accuracy
  should be measurably higher than R1's 0.75. Group accuracy stays at 1.0.
- For G₁ (CNOT 6, 7 in 15-Pauli pool): unclear, depends on whether the
  decoder's tie-break and the carry-over interaction breaks the symmetry.

If hypothesis fails:
- R3b ceiling stays at 0.75. Then the ambiguity is even deeper than
  per-round-distribution equivalence — it's a property of the full
  conditional law on syndrome streams, not just one-round marginals. Still
  a valuable result (negative is informative).

## 6. Out of scope for this overnight pass

- Larger windows (>1)
- Probabilistic / soft-output decoders
- Non-reset mode (the round-iid argument breaks; would need separate
  treatment)
- Pauli pool other than 15
- Decoder priors derived from `p_bg / p_high`
