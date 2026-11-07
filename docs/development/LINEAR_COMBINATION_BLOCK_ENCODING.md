# Default-oracle linear-combination block encoding

G2-BE07 composes the unchanged Qualtran v0.7.0 `LinearCombination` default
oracles through native operations. Locked commit:
`bcb17bb25360ed68ce3bc084645db3e929799389`; parent source blob:
`a3d8aa7fafa7a6f3fcaf095826751209427cd255`. Copyright and source are retained.

`LinearCombinationBlockEncoding` belongs to algorithms/primitives. It reuses
`LinearCombinationPreparation`, `LinearCombinationSelect`, `BlackBoxPrepare`,
shared partitioning, bound operation composition and component cost calls.
There is no new graph, circuit IR, arithmetic lowerer or execution backend.

Packed register order follows the original:

- system: common constituent system;
- ancilla: coefficient selection, then maximum constituent signal ancilla;
- resource: maximum constituent resource, then coefficient preparation junk.

The operation emits PREPARE, signed SELECT, PREPARE-adjoint. The block adjoint
keeps PREPARE forward and reverses only SELECT's child/traversal direction.
A positive external control controls SELECT alone; preparation remains
unconditional, including when the control is in a superposition. Caller-owned
private scratch can be shared across sequential PREPARE/SELECT lifetimes.
Selection, signal and logical junk are meaningful outputs, not scratch promised
to clear after the forward block. The exact inverse restores the complete input.

Metadata retains the source implementation. Alpha is sum(abs(lambda_i)*alpha_i).
Epsilon uses the normalized padded coefficient array times max(child epsilon),
which differs from the source docstring's unnormalized sum(abs(lambda_i)) bound.
A regression distinguishes them explicitly. Identity signal preparation has no
invented L1 normalization. Symbolic alpha remains inspectable as metadata;
the default probability factory requires concrete block normalizations.

Resource models consume explicitly paired directional SELECT, PREPARE and
PREPARE-adjoint models. All inactive through ports are live bystanders; only
sequential scratch peaks are shared. Estimation does not emit or execute a
circuit. It preserves caller-selected QEC profiles rather than inferring a
formula from an IR count. The integrated default-profile source comparisons
use the existing forward/adjoint alias model; alternative profiles need their
corresponding supplied preparation models. Original generic controlled child
resource failures remain visible as recorded in the signed SELECT dependency.

Validation covers 32 complete original configurations: two/three terms,
1/8 coefficient bits, mixed auxiliary widths, external control present/absent
and both directions. Original primitive dependencies, operands, feedback,
metadata, all QEC categories and peak width match. Reference graph traversal
retains each component's scope. Classical storage names bind measured wires
within PREPARE and SELECT scopes; they are not guessed from a flattened
parent's scratch index. The untouched parent supplies the partitions and order.

Twelve small full-matrix tests apply arbitrary complex states to every logical
port, including nonzero and entangled preparation junk and control. Every
measurement history is compared as an unnormalized complete complex vector,
including private work, without phase alignment or partial trace. Those state
checks use one-bit coefficient precision; eight-bit fidelity is established by
complete original gate flow, not a claimed exponential state simulation.

Run:

```sh
python -m examples.algorithms.linear_combination_block_encoding
```

The example uses circuit.append(op) and encodes (X-3Z)/4 with alpha=4. It verifies
the projected complex block, clean private work, and full inverse restoration,
retaining every failure amplitude. Model and emitted counts remain separate:

| Quantity | Original default model | Emitted IR |
|---|---:|---:|
| T equivalents / T and T-dagger | 16 | 22 |
| Measurements | 2 | 2 |
| Peak logical qubits / reserved qubits | 8 | 8 |

Both PREPARE calls use the exact seven-T Fredkin decomposition, while the
original abstract model charges four T equivalents per Fredkin. That explains
the six-T difference; the native primitive sequence matches the original.

Custom PREPARE/SELECT overrides, symbolic probability construction and additional
control specifications are not certified by this default-oracle slice. The
original integer-array normalization failure is retained in the coefficient
module. The original parent is not marked wholly accepted by this slice.
