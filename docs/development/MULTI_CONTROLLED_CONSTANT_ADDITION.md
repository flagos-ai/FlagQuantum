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

## Grouped integer controls (G2-A49)

`grouped_controlled_add_constant_ir` binds existing `ControlRegister` descriptors
and per-group flat wire tuples. Word bits are MSB first; words in shaped source
registers flatten in row-major order. The source dtype/shape adapter remains
outside the compiler. The native descriptor retains the activation word values
and their unsigned, two's-complement or ones-complement encoding, rather than
adding a new schema. Stored Montgomery/bounded words use the unsigned bit pattern.

All nonempty control configurations are supported by this concrete entry. One
control bit needs no predicate work; the generic wrapper owns its positive or
negative polarity. The existing flat entry keeps its m>=2 contract and delegates
to the same grouped implementation. Grouped resource queries likewise reuse the
same control-component authority with logical target_width=n.

Reference tests cover QBit arrays (including two-dimensional shapes), QAny,
unsigned, signed, ones-complement, stored Montgomery and bounded integer groups,
forward/inverse and both cost profiles. Small signed/ones-complement and single-bit
control cases enumerate all coherent input and measurement histories, including
an entangled spectator. Wire overlap, wrong group sizes, invalid encoding values,
empty controls and overlapping measurement slots fail explicitly.

This adds 56 tests; 1,359 concentrated checks pass without skips. Six numerical
warnings are retained in the evidence log from the pre-existing LCU reference
probe, not the grouped AddK tests. Reference Cirq 1.7.0 is restored in an isolated
directory to provide its CY interface; project dependency files are unchanged.
The five relevant source blobs match the pinned inventory. Symbolic grouped
shapes remain outside this concrete implementation.

Run `python -m examples.algorithms.grouped_controlled_constant_addition` to activate
on a signed two-bit control of -1, subtract one from the target, verify clean work,
and compare logical model counts with emitted primitive counts.

The grouped concrete model explicitly requires a Python integer constant;
source symbolic constants use a different graph-width fallback and are not
accepted as if they had executable loaded/carry lifetimes. That symbolic
resource path remains open. The preceding flat-control limitations describe
G2-A48; G2-A49 adds the grouped concrete paths documented here.
