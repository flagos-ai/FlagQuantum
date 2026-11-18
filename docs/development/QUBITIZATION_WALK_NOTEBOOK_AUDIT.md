# Complete qubitization walk notebook migration

Q-0657 maps the entire Qualtran v0.7.0
`qubitization/qubitization_walk_operator.ipynb` at frozen commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its unchanged Git blob is
`a31dafeb427e4d7e13bfb0eb9bc903b38a15a316`. The native artifact is
`examples/notebooks/qubitization_walk.ipynb` and retains Google LLC 2023/2024
Apache-2.0 attribution, including the two copied integral-fixture algorithms.

## Complete original responsibility mapping

The first sixteen cells preserve the original Markdown/code sequence. Two added
cells supply a runnable append example and independent resource accounts.

| Original cells | Native responsibility |
|---|---|
| 0–1 | Walk spectrum/normalization description and native compiler, algorithms, Core and Drawer imports. |
| 2–3 | SELECT semantics and existing named directional provider / `SelectBlockEncoding` boundary. |
| 4–5 | Existing `PrepareOracle` declaration and accepted Identity specialization. |
| 6–7 | Existing `QubitizationWalk`, shared reflection, inverse/control and junk lifetime contracts. |
| 8–9 | Original four-site Ising factory at epsilon .2, complete normalization and typed signature. |
| 10 | Original THC dimensions 152/450, ten PREPARE bits, twenty angle bits, RandomState(7) symmetric integrals and block factor 32. Native prepared data and complete typed walk signature are constructed. |
| 11 | Original eight-spin-orbital sparse fixture, eight rotation bits and twelve PREPARE bits. Native integral fixture and existing sparse walk factory preserve normalization and signature. |
| 12–13 | All three graphical signatures use accepted shared Drawer signature / Graphviz owners. |
| 14–15 | Original one-level SELECT block / prepared-reflection graph and leaf multiplicities. Independent native Ising resource query. |
| Added 16–17 | Small X/Z walk through ordinary append, complete complex state and clean phase anchor, emitted primitives and separate source model. |

The native notebook constructs physical Ising and sparse declarations through
existing factories. The THC cell supplies the original prepared data and a typed
signature. The original notebook uses that large factory for metadata drawing.
Its physical binding follows the existing chemistry contract requiring explicit
rotation networks.
The accepted THC module retains its own small-system numerical and resource
fixtures. Its larger physical admission remains with the chemistry parents.

The source `QI(L)` minimizer and the existing forward block minimizer at target
width one have the same objective up to a constant. Their chosen block factor is
compared directly at the original L=526. No new optimizer is introduced.

## Verification and executable workflow

The original notebook passes its unchanged pytest in an isolated reference kernel.
The native test executes both complete notebooks, verifies their frozen source
hash and cell coverage, and compares every integral array exactly. THC alias
columns and normalization match the original factory. All three signatures retain
original names, widths, shapes and sides. Native Ising normalization and QEC/peak
model match the source. One-level child identities and multiplicities match.

Native cell execution rejects Qualtran, Cirq and OpenFermion imports. The reference
fixture imports declare a known upstream `notebook` marker, which is registered
locally by the test without changing reference assertions. The native notebook
produces eight display objects including four SVG graphs, all rendered to PNG and
visually inspected for readable tables and complete edges.

Run the notebook from top to bottom. Its final example constructs an identity
signal and a two-row X/Z SELECT provider, then uses

```python
op = small.make_on(
    selection={"selection": (0,)}, target=(1,), reflection_workspace=(2,))
circuit = Circuit(3, dtype=torch.complex128).h(0).h(1)
circuit.append(op)
# Complete amplitudes are (1, 0, 1, 0, -1, 0, 1, 0)/2.
```

The single unitary branch returns the borrowed anchor to zero. The example emits
two Identity, six X, one CX, two CZ and one phase instruction. Its independently
queried source declaration has zero T equivalents and peak three logical qubits.
No measured gate count is inferred from that declaration. Larger measured walk
branches, cleanup and inverse behavior reuse the accepted module evidence.

The original slow walk suite reached its 180-second deadline after one successful
case. Its remaining cases stay unverified and are recorded in
`QUBITIZATION_WALK_FILE_AUDIT.md`. Controlled-LCU signature and typed physical
admission boundaries remain with Q-0658/Q-0659.

The batch regression passes 282 cases without skips or warnings. Strict mypy
covers the native test and all extracted notebook code. Black, Ruff, Algorithms
scope, architecture, multi-level IR and protected API checks pass.
