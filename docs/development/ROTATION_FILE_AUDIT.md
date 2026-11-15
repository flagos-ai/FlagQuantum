# Basic rotation source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain their
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0250 `rotation.ipynb` | `79016d2d85080d494b121afc47600772b4ca9953` | Closed by the native Rotation tutorial. |
| Q-0251 `rotation.py` | `69d2f19d636db2de0105919fb6b53421ba5b94b4` | Concrete matrix and decomposition paths and independent resource declarations are verified. Foreign adapters and object protocols remain open. |
| Q-0252 `rotation_test.py` | `63480cc29919427e516c71f7453e8bdfa7cf01a3` | The unchanged suite runs. Native coverage retains the external adapter, source formatting and generic factory test boundaries. |

The 50 original cells present ZPow, CZPow, Rz, CRz, XPow and YPow in that
order. The native notebook retains the six instances, the two source symbols,
six graphical signatures, four call graphs with leaf counts and two
decomposition scores. The 52 native cells add a complete append example.

## Phase, binding and resource accounting

The three Pauli powers preserve their source global phase relative to radian
rotations. A controlled application makes that phase relative to the inactive
control block. The example compares full complex matrices, explicitly checking
the Pauli-power phase against the three native radian rotations. Numerical
execution uses finite bound angles. Symbolic angles and precision remain
visible in resource declarations.

`CZPow` binds the two entries of `q` in source order and computes an AND on
one clean work wire, applies ZPow to the work wire, then uncomputes with
measurement and feedback. Existing tests compare the actual sequence against
the original for both outcomes, including full unnormalized complex states,
work reset and the inverse. The notebook checks both normalized trajectories
with one clean work wire and one classical slot.

`CRz` binds scalar `ctrl` then `q`. Its order is positive half-angle ZPow,
CX, negative half-angle ZPow, CX. Existing tests compare the source sequence
and precision halving. The example checks the complete controlled Rz matrix.

| Path | Executed stream | Source declaration |
|---|---|---|
| CRz at 0.37 radians | 2 phase, 2 CX, no work or measurement | 2 rotations, 2 Cliffords, peak 2 |
| CZPow at exponent 0.123 | 3 H, 2 T, 2 Tdg, 6 CX, 1 S, 1 phase, 1 measurement, 1 CZ, 1 reset, one work wire | 1 AND, 1 rotation, 1 measurement, 1 Clifford, peak 3 |
| XPow and YPow | A phase conjugated by basis changes, with scalar shift handled by the existing emitter when supplied | Each is an atomic leaf in the teaching graph |

The locked tests cover angle conversion, special exponents, tensor and
adjoint identities, controlled dispatch, the external Cirq and PennyLane
operations, source strings and the factory harness. Existing native tests
cover the concrete matrix relationships, primitive flow, classifications and
measurement cleanup. The native tutorial uses native labels and does not
claim external object or factory test compatibility.

Both notebooks ran in independent kernels. The native kernel rejected
Qualtran, Quimb and Cirq imports. All ten SVGs and two score PNGs were
inspected. The locked test suite recorded 36 passes and 14 mode skips.
The 115-case direct regression and 110-case integration profile passed.
Strict typing, formatting, team scope, architecture, multilevel Program IR
and the public API baseline also passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_rotation.ipynb
```
