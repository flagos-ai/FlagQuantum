# Source workflow mapping and acceptance

Source: `algorithms/foundational/quantum_teleportation/quantum_teleportation.ipynb`
at `b69c4893d8264d1e59ce0909df2bd88efed25e00`. Cell hashes are in
`provenance.json`.

| Source cell | Native workflow |
| --- | --- |
| 8 | Frozen native coherent feedback in place of private measurement and corrections. |
| 10 | Original H-prepared plus state; angle parameters also allow complex test states. |
| 12 | Allocate Bob, Alice and Bell, prepare entanglement, send, drop private outcomes. |
| 14 | Native synthesis, actual resources and text circuit diagram. |
| 16 | Actual Bob readout, 1000 local observations and reduced density inspection. |

Only Bob's output channel is equivalent to the measured protocol. Two dropped
outcome qubits remain a retained environment, uniformly distributed over four
outcomes. They are not clean workspace. Generic measurement/reset and physical
classical communication remain outside this implementation. Source sampling has
no explicit shot count; local 1000-shot sampling and seed 3 are adaptations.
No identical SDK default or remote sample sequence is claimed.

Acceptance compares complete absolute complex purification with an independent
Circuit and Bob's density with independently measured/corrected branches, retaining
global and relative phase checks. Original plus and two nonzero complex inputs,
external entangled-reference identity channel, basis outputs, named readout,
repeatability, invalid inputs and five resource refusals are checked.

Actual resources depend on bound angles: the original plus input needs 3 qubits,
0 clean work bits, 9 operations and depth 6; each complex test uses 4 qubits,
1 clean work bit, 12 operations and depth 9. Clean work must return to zero;
private outcome qubits must not be relabeled as clean work.
Classification: `single_device_fast_path`.

## Recorded local acceptance outcome

44 focused pytest cases passed, including the retained entangled-reference channel
checks. A fresh local Jupyter kernel executed four unchanged native code cells
and one evidence-only cell; the CLI also ran the original plus input with 1000
observations. Its counts were 524 zero and 476 one. Complete absolute purification
errors were below 1.1e-16 and independent measured-density errors below 4.5e-16
across the original input and two nonzero complex inputs.

All 216 original replay outcomes and resource rows, and all six approved
revisions, were unchanged. The full default tier had 19619 passing tests and the
same prior 24 failed and 6 error nodes; none were added or removed. This does not
certify the whole repository as passing. All 3872 Python inputs were frozen
through that run; only catalog assertions changed after acceptance.
