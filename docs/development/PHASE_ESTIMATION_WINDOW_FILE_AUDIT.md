# Complete phase-estimation window base and rectangular preparation

Q-0627 maps frozen Qualtran v0.7.0
`phase_estimation/qpe_window_state.py`, blob
`075147c04eef8ff5e47067d992c9af4c47372099`. Q-0628 maps its test file,
blob `0ca33cf4ab254b52dac9a605e0efe1a1bda98d84`, at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

Algorithms owns `phase_estimation_window.py`. The window base owns the phase
width and RIGHT-only unsigned fixed-point output, with all bits fractional.
Physical words use big-endian order, so the first wire is the half-unit bit.
The rectangular window reuses Compiler `on_each_ir` and the existing named
BlackBoxPrepare binding. The caller supplies clean physical output storage.
There are zero borrowed work wires, result slots or measurements. Core owns the
register descriptor and precision arithmetic, while Compiler owns graph
aggregation. Symbolic widths remain declarations and fail before physical binding.

The precision/delta and standard-deviation factories preserve the complete source
ceil/log2/pi expressions, parameter order and failure behavior. The source base's
`m_bits`, `m_qdtype` and `m_register` map to the same native properties.

## Complete source responsibilities and evidence

Both original example declarations map to the same numeric and symbolic native
factories. Source drawing labels use the existing register declaration layer.
All original base properties, factories, output declarations and the complete
OnEach(H) preparation body have an executable native counterpart. The caller
uses `circuit.append(window.bind().make_on(selection=wires, junk=()))`.

At widths 1, 2, 4 and 5, the frozen Cirq decomposition is captured and mapped
onto reversed physical wire layouts. The exact H stream and operands match the
native emitter. Full complex amplitudes match the frozen tensor, including a
complex spectator. The spectator's unused computational branch stays zero for
clean input. Appending the exact inverse restores the coherent input. Replacing
the Compiler emitter with the captured source stream preserves the consumer and
state. No measurement branches occur in this source.

The complete original test suite reports 4 passed and 2 original skips in
`g2-window-file-source.log`. Concentrated native PREPARE, reflection, window and
GQSP regression reports 274 passed. Black, Ruff, strict mypy for both new Python
files, Algorithms scope, architecture, multi-level IR and protected API checks
pass. Logs use the `g2-window-file-` prefix in the external reference workspace.

## Separate emitted inventory and declared graph

For positive width n, the actual emitted primitive inventory is exactly n H,
zero measurements, zero result slots and zero borrowed work wires. The caller
owns n clean output wires. The frozen QEC model independently reports n Clifford
and zero T. The source declaration graph retains Allocate(phase) once and
OnEach(H) once, with n H leaves. Allocation metadata is not counted as a physical
gate. Symbolic n remains symbolic in the same graph.

## Preserved source boundaries

The original autotest skips decomposition counts for the small example and
strict type checking because Allocate's QFxp(5,5) output and OnEach's QAny(5)
input are incompatible. Native execution uses the existing explicit physical
binding while preserving the fractional output descriptor. This does not certify
the upstream strict type check. At width zero, the frozen tensor raises
`ValueError: not enough values to unpack (expected 2, got 0)`.
The native declaration and zero-count graph remain available, while physical
the Algorithms binding refuses this standalone source-unsupported body explicitly.
Q-0628 was initially retained for these original test boundaries. Q-0627 closes the complete source responsibility
mapping with those boundaries recorded. No reference source was modified.

## Complete original test-file closure, November 25

Both original test functions and both examples now have independent observations
under every original autotester mode. The complete n=4 tensor and numeric/symbolic
construction, decomposition and serialization assertions retain native evidence.
Original count checks remain UNVERIFIED with only decomposition counts. Strict
typing remains UNVERIFIED for QFxp -> QAny. No skipped assertion becomes a pass.
See [the complete test matrix](WINDOW_SOURCE_TEST_ACCEPTANCE.md).

Approved Proposal 116 admits unsigned empty fixed-point storage. The old native
constructor witness incorrectly expected its earlier storage rejection. A
standalone physical circuit now reaches the existing one-physical-wire guard.
With a live borrowed spectator, the empty RIGHT output binds and emits zero
gates. The unmodified source H/window/T context and native append agree on their
full two-by-two complex operator at absolute tolerance 2e-15, without phase
alignment. Every allocation has empty storage and the spectator remains live.
The standalone source tensor failure and Algorithms bind refusal are unchanged.

Q-0628 closes its actual complete original assertions and source-protocol
disposition. The contextual proof explains the current failure stage without
weakening storage, phase or output assertions.
