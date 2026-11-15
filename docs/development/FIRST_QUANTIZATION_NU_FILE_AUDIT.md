# First-quantization momentum-transfer preparation audit

The frozen Qualtran v0.7.0 reference is at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization`.

| Inventory | Source | Git blob |
| --- | --- | --- |
| Q-0365 | `prepare_nu.py` | `ae1e4e6bc279596c33c16b791ffe4d7fbe3ca788` |
| Q-0366 | `prepare_nu_test.py` | `6cf67ef6fb128fc137201d18d8ea17da1bd270b5` |

## Registers and lifetime

The native Algorithms module owns source signatures. The enclosing
component has through words `mu`, a three-axis `nu`, `m`, and `flag_nu`.
Momentum and m words retain the source widths and opaque encodings.
Five declaration-only leaf signatures preserve the original direction of
each temporary flag. The decomposition binds the three flags as follows.

| Forward producer port | Inequality consumer port |
| --- | --- |
| `flag` | `flag_mu_prep` |
| `flag_minus_zero` | `flag_minus_zero` |
| `flag_nu_lt_mu` | `flag_ineq` |

The adjoint reverses those connections. Both directions connect the
inequality's through `succ` to the caller's `flag_nu`. The resource model
derives direct port widths from the signatures and checks the flags' live
bystanders against the locked composite. The analytical call graph omits
m preparation, while the composite schedules it on its own word.
At `m_param=1` the source still declares a zero-width m word. Its full
composite rejects that binding, and the native model retains that limit.

## Measured child and analytical model

`prepare_nu_m_on` binds the original uniform superposition substep through
`circuit.append(op)`. It delegates gate emission to Compiler's existing
uniform preparation. The actual substep in the frozen composite was
bound on a reordered target with a spectator. Both directions match
every complex amplitude and clear workspace in all 32 measurement
histories. The three-value child emits 20 T/T† and five measurements
with four clean work wires. Forward and adjoint primitive and feedback
recipes reuse the independent shared uniform-preparation comparison.

The original two tests passed. Their forward-plus-adjoint analytical
comparison at p=6 and M=32768 gives 615 after dividing its total T
equivalents by four. It uses the formula in the locked source.
No whole-component emitted primitive count is available. The original
mu preparation, nu superposition, minus-zero flag, range flag and final
inequality all lack gate decompositions in this release.

Q-0365 and Q-0366 close the source declarations, composite metadata,
executable m child and original tests. End-to-end momentum-transfer
execution still requires physical implementations of the five source leaves.
