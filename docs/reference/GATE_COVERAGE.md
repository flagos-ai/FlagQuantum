# Gate coverage

FlagQuantum separates **logical expressibility** from **end-to-end support**.
The presence of a builder method alone does not mean that every simulator,
compiler target, ecosystem adapter, or remote provider can execute that gate.

This page records the built-in operator surface after API Change 098.
Provider inventories and hardware-native gate sets remain device facts; they are
not inferred from this table.

## Current built-in surface

The operator registry declares 38 canonical unitary gates. API Change 098 added
the complete fixed two-qubit batch shown below:

| Family | Canonical opcodes |
| --- | --- |
| Fixed one-qubit | `i`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg` |
| Parameterized one-qubit | `rx`, `ry`, `rz`, `phase`, `u1`, `u2`, `u3` |
| Fixed two-qubit | `cx`, `cy`, `cz`, `swap`, `iswap`, `iswapdg`, `ch`, `csx`, `csxdg`, `dcx`, `ecr` |
| Parameterized two-qubit | `crx`, `cry`, `crz`, `cphase`, `rxx`, `ryy`, `rzz` |
| Fixed three-qubit | `ccx`, `cswap` |

It also declares four channel instructions: `bit_flip`, `phase_flip`,
`depolarizing`, and `amplitude_damping`. Public aliases include `id`,
`hadamard`, `p`, `u`, `cnot`, `toffoli`, and `fredkin`.

Both `Circuit` and `DynamicCircuit` receive the same schema-installed fluent gate
methods. `DynamicCircuit` adds measurement, reset, and classical conditions; those
are instructions rather than unitary gates.

## What “supported” means

A built-in gate is complete only when every applicable row below has evidence.

| Layer | Required evidence |
| --- | --- |
| Construction | Fluent methods on `Circuit` and `DynamicCircuit`, exact argument validation, typing, and IR round-trip |
| Semantics | A canonical operator schema, matrix or exact decomposition, adjoint, control, and power behavior |
| Local execution | Statevector, MPS, tensor-network, density-matrix, batching, precision, and device paths either execute or fail closed |
| Quantum AI | Parameter binding, autograd, parameter-shift metadata, and distributed-gradient behavior are tested where applicable |
| Compilation | Optimization preserves semantics and every advertised target either accepts the opcode or lowers it exactly |
| Interoperability | Import and export adapters map the opcode without changing qubit order, parameters, or global-phase-sensitive behavior |
| Remote execution | The selected provider's device-scoped capability snapshot admits the post-routing program |

`Circuit.any(...)` permits an explicitly supplied dense operator, and
`Circuit.control(...)` can synthesize supported controlled operations within its
documented size bound. These mechanisms make the language expressive, but they do
not replace named-gate validation, portable serialization, target legalization,
or provider capability evidence.

## Coverage assessment

The current registry is a strong backend-neutral algorithmic core. It includes a
universal one- and two-qubit basis, standard variational rotations, common
controlled rotations, Pauli interactions, and reversible three-qubit gates.
Ordinary algorithm construction therefore does not depend on a missing named
gate.

The surface is not yet a complete cross-hardware vocabulary. The most useful
gaps are grouped by ownership rather than added indiscriminately to Core:

| Priority | Gap | Intended treatment |
| --- | --- | --- |
| A | `ecr`, `ch`, `csx`/`csxdg`, `dcx` | Implemented with exact matrices, lowering, static/dynamic builders, and explicit adapter support boundaries |
| B | General controlled unitary (`cu`) and convenient multi-control spellings | Separate design because global phase, parameter convention, decomposition cost, and ancilla policy are observable |
| C | `fsim`, phased-X, Mølmer-Sørensen, `gpi`, `gpi2`, and other hardware-native operations | Target/profile or extension ownership first; promote to Core only with two concrete consumers |
| D | Additional channels already available as noise factories | Promote only when IR materialization and every advertised channel executor agree |
| E | `barrier`, `delay`, pulse operations, and global phase | Model as directives or program properties, not as ordinary gates |

## Approved boundary

[API Change 098](../development/API_CHANGE_PROPOSAL_098_GATE_COVERAGE_BATCH_A.md)
defines four implementation slices. All seven names are now registered public
FlagQuantum opcodes. This logical support does not claim that a provider or QPU
executes any of them natively; remote admission still uses post-routing,
device-scoped capability facts.
