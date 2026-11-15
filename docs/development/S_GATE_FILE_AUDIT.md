# S and S-adjoint source family

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The two files retain the
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0253 `s_gate.py` | `6111677a1e55ccab022e988598366a0f1bdcfc94` | Native execution, tensor and resource parity verified. External S-adjoint conversion remains open under the current interop contracts. |
| Q-0254 `s_gate_test.py` | `98ed87f00f6788bd20a54eeed8b3770e738f13a8` | Original test scenarios mapped. Full native reproduction of the original Cirq diagram through the S-adjoint adapter remains open. |

Both directions have the scalar `q` THRU signature, emit one `s` or `sdg`
instruction and preserve the source complex matrices. The adjoint toggles
the direction and the score helper renders the corresponding `S†` label.
Each version has one atomic graph leaf and one modeled Clifford. Neither
direction requires measurement, feedback or work wires. Existing control
tests compare full complex states, clean work and all tested measurement
histories for multi-control variants.

| Original test | Native evidence |
|---|---|
| `test_s_gate` | The locked factory binds to the native diagonal tutorial and executes as one atomic gate. The original harness skips decomposition, counts and decomposition-dependent dtype modes. |
| `test_call_graph` | The native graph and the locked graph each retain one leaf and zero edges for S and S-adjoint. |
| `test_to_cirq` | The native H-and-S prefix round-trips through the bridge as the original ordered operations. The full native H/S/S-adjoint state agrees with the original Cirq circuit. Native `sdg` export currently raises `unsupported_operation`, so exact adapter reproduction of the three-operation diagram remains open. |
| `test_pl_interop` | The native S export produces the same PennyLane operation as the locked source, round-trips, and matches the full complex matrix. Native S-adjoint export raises `unsupported_operation`. |
| `test_tensors` | Full complex S and S-adjoint matrices are checked against the locked tensors. The unchanged source suite also compares both against Cirq. |

The Cirq and PennyLane contracts currently classify `sdg` as unsupported.
The new reference tests verify that boundary explicitly. Enabling `sdg` in
either converter needs the corresponding interop contract change.

The unchanged source suite recorded six passes and three mode skips. The
98-case native regression included the Cirq and PennyLane conformance paths
and passed without skips. Strict typing, formatting, Compiler scope,
architecture, multilevel Program IR and public API checks also passed.

Use the existing tutorial to append and verify S, its inverse, its square,
and both controlled phases:

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_diagonal_gates.ipynb
```
