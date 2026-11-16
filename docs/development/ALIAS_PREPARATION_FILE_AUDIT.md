# Coherent alias preparation source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The notebook, implementation, and
test blobs are `081add39d9e5a13a0e215483e429b6f0d9e3ff84`,
`4850ef95d9ebf58b0e062505a3482db2f5307443`, and
`c5141c0e4beadb9a890dc3fc035601252a6c0454`.

## Native owners

| Source responsibility | FlagQuantum owner |
| --- | --- |
| Dense probabilities, sparse dictionary, and dense-to-sparse input | The concrete alias table factories preserve input row order, integer alternate and keep data, and normalization. |
| Table-free dense and sparse constructors | `AliasPreparationShape` owns both precision formulas and resource declarations. The dense formula depends on normalization. The sparse formula uses the full domain with normalized tolerance. |
| Signature and logical junk | The alias PREPARE emitter binds selection then sigma, optional sparse row, alternate, keep, and comparison bit. Words are big-endian. The junk is a logical output and can remain entangled with selection. |
| Composite and adjoint | Shared uniform preparation, sigma Hadamards, QROM, measured comparison, and full-word controlled swap. The adjoint reverses their order. |
| Temporary lifecycle | QROM and comparison share cleared temporary wires and classical slots. PREPARE restores temporary work. PREPARE followed by its adjoint restores the selection and logical junk. |
| Independent estimates | Shared uniform, QROM, and comparison models declare their gate categories and live-width peaks. The alias model composes them with logical bystanders. |

The native [tutorial](../../examples/notebooks/state_preparation_alias_sampling.ipynb)
retains both source sections, all five original factories, their signatures,
and the concrete depth-one call graphs. Its final cell appends the dense and
sparse examples with `circuit.append(op)`.

## Verification

The two concrete sparse input forms produce the same table and operation. The
original dense, dictionary-sparse, and diagram fixtures match the source
component order, wire order, and primitive streams. The native interpreter
checks the complete complex output and clean work against the unchanged source
Cirq action, and runs the emitted adjoints to restore all wires. Each larger
fixture uses three seeded measurement histories. Existing small fixtures retain
their exhaustive measurement histories. The original Ising probability grid,
uniform low-precision case, and both sparse probability cases also run through
the native append entry. The original source suite passed 17 cases including
its notebook, with three documented skips and three expected failures.

The two exact symbolic factory declarations match modern and legacy resource
categories and logical width in both directions. The original T-complexity
test expands the model's CSwap and And categories. Its sparse Clifford expression
includes ten additional Cliffords for each declared CSwap and And under that
conversion.

| Original fixture | Emitted T/T† | Emitted CX | Emitted measures | Reserved wires | Source CSwap | Source And | Source Clifford | Source measures | Source logical wires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense, four rows | 42 | 117 | 7 | 19 | 2 | 7 | 80 | 7 | 19 |
| Sparse, four support rows over nine indices | 56 | 144 | 7 | 25 | 4 | 7 | 91 | 7 | 25 |

## Remaining boundaries

- Q-0700 remains open for executable symbolic binding and generic source
  register subclasses. Native table-free models and fixed concrete bindings
  provide the covered paths.
- The locked source has sparse singleton target/call failures and a symbolic
  two-row width-query failure. Native constant singleton handling and explicit
  model rejection retain separate behavior and tests for these cases.
- Source object serialization has expected failures. Two strict typed-edge
  assertions in the original source suite skip on BQUInt/QAny connections.
  The native migration verifies concrete register widths and operand order.
