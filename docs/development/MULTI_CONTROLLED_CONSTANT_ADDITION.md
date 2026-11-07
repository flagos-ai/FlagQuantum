# Multiple-controlled constant addition

`compiler/registers/multi_controlled_constant_addition.py` ports the flat QBit
control path of Qualtran v0.7.0 ControlledViaAnd(AddK), pinned at
`bcb17bb25360ed68ce3bc084645db3e929799389`. It composes the existing control
predicate, positive controlled AddK and measured predicate erasure. The inverse
switches only the middle AddK to its exact measured inverse. No reference source
or existing primitive emitter is modified.

The caller supplies m>=2 controls with binary activation values, n target bits,
m-1 clean predicate wires, n clean constant wires, and n-1 clean carry wires.
Predicate work is ordered as m-2 prefix junk followed by the final predicate.
Only the target changes. Controls are preserved, and every work wire returns to
zero. All wires and the two phases' measurement slots must be disjoint.

The concrete source resource model reuses existing predicate and single-control
AddK models, including their seven categories and selected QEC profile. It counts
target wires as bystanders during predicate compute/erase, and controls plus
prefix junk as bystanders during addition. The resulting peak is 2m+3n-2 for the
compared concrete domain. Model Clifford categories remain distinct from emitted
primitive Clifford counts; T and measurement totals close independently.

Tests compare original operand/feedback dependencies for 2–4 controls, mixed and
uniform polarities, forward and inverse. Small coherent control/target/spectator
states are projected through every measurement history without normalization,
checking relative phases, probabilities and clean work. Resource fields and peak
match the original in both profiles. Replacement tests swap shared models without
changing consumers. Input ownership failures are tested before execution.

This batch adds 83 tests and passes 1,223 concentrated checks without skips.
Strict mypy, formatting, lint and architecture checks pass. This slice covers
flat QBit controls and unsigned bit-pattern targets; grouped integer control
registers and symbolic resource shapes remain separately unaccepted paths.

Run:

```sh
python -m examples.algorithms.multi_controlled_constant_addition
python -m pytest -q tests/unit/test_multi_controlled_constant_addition.py
```

The example uses `circuit.append(op)`, activates on control 10, computes
1-1=0 modulo four, checks all four work bits and prints the logical model
separately from the emitted primitive counts.

Both emission and resource composition delegate to the existing generic
`controlled_component_ir` / `controlled_component_cost` helpers. This module
only binds the AddK body and validates the caller's complete storage ownership;
it does not maintain a second predicate schedule or bystander-width formula.
