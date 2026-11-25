# Native coherent Fourier control

The user authorized native counterparts for all five source categories and
successive local batches. This unreleased language extension admits
`control(condition, lambda: qft(target))` in a root Python qfunc. It reuses
Compiler-owned `qreg.control_qft`, verification, controlled instruction
construction, scheduling, budgets and execution. It adds no public symbol,
signature, IR operation, dependency or executor. Release authorization is false.

The condition is an initialized scalar port or an existing bounded comparison
over at most eight bits. The target is an initialized whole QNum or QArray[QBit]
port. Target and every condition input must be disjoint. Const targets,
uninitialized storage, slices, host callbacks, nested controls and mutable-helper
control remain outside this admission. Comparison workspace is uncomputed and
released after the transform; QFT cannot modify a comparison input. Existing
phase and Hadamard control behavior is unchanged.

The Hadamard-test examples keep their four-bit target as an explicit native
output and select only `expectation_value` for probability readout. This spells
out ownership without pretending that Qmod `drop` cleans an entangled register.
The five-qubit joint state remains available and charged to budgets. This
adaptation does not claim native lexical storage, drop/free, reset, zero hidden
target, or a mixed-state-to-statevector conversion. Readout marginalization is
the existing Program behavior. Source targets remain hidden in the test-only
original-source adapter; their full joint states must nevertheless agree.

## Acceptance

- Compare every small Fourier column, asymmetric coherent inputs and absolute
  global phase against the Fourier definition in an independent Circuit.
- Cover scalar controls and bounded predicates, uncomputed predicate workspace,
  target/condition overlap, Const and initialization errors and actual budgets.
- Run the unchanged four-bit Hadamard-test algorithm and tutorial originals;
  compare full five-qubit states and ancilla marginal probabilities 5/8, 3/8.
- Preserve original identities/resources and distinguish native program coverage
  from complete Notebook workflows. No remote CI or PR before full acceptance.
