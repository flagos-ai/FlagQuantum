# Complete register wiring tutorial

Q-0943 binds the complete frozen `tutorials/quantum-programming/1-wiring-up.ipynb`
at Qualtran v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`.
The Git blob is `061fa74a7a2a4213bd406ba9fdd16cbad07276c2`.
`examples/notebooks/register_wiring.ipynb` maps all 19 original cells in order
and appends one provenance and execution-scope cell. It retains the original
Apache-2.0 attribution.

## Complete source mapping

| Original cells | Native responsibility and observable behavior |
| --- | --- |
| 0–2 | Introduce composition, declare the eight-bit signed Core signature, and construct the identity circuit |
| 3–4 | Validate the identity IR and its named logical signature |
| 5–6 | Draw the identity register dependency with the shared Drawer |
| 7–9 | Encode positive and negative eight-bit integers and explain fixed-width complement plus increment |
| 10–12 | Append shared BitwiseNot and AddK(1) operations to the same target, with explicit caller-owned work and classical slots |
| 13–14 | Validate the complete emitted IR and the actual concatenated component bodies |
| 15–16 | Draw the two-step x dependency and report emitted primitive counts, the independent AddK model and physical capacity separately |
| 17–18 | Use the shared classical composition to evaluate x=5 as -5 |

The native API is `circuit.append(op)`. A small notebook-local immutable append
adapter carries existing native instructions. It adds no production abstraction,
resource model, quantum algorithm or simulator. The native circuit API replaces
the source builder and its implicit allocator. Core owns signed register metadata
and structural validation. Compiler owns complement and literal constant-add
emission. Simulation owns the value codecs and classical composition. Drawer
owns register layout and rendering. IPython and Graphviz remain optional tutorial
presentation dependencies.

## Evidence and limits

The acceptance test executes the unchanged original notebook and every native
code cell. Native execution rejects imports from Qualtran, Cirq and Kickmix.
All 256 eight-bit signed inputs and bit encodings agree with the original,
including the minimum signed integer wrapping to itself. The identity preserves
every input. Both actual graph renders were inspected. The negation graph has
exactly the two original calls, one connecting x edge, and named input/output
boundaries.

The actual eight-bit native primitive stream matches the unchanged original
composite after explicit wire and measurement-slot binding. Gate multiplicities,
per-wire dependency order, operands and feedback are checked. Existing complete
negation and constant-add evidence is reused through concentrated regression.
The small three-bit coherent test adds a spectator entangled with all input
values, enumerates all four carry-measurement histories, compares every complex
unnormalized amplitude, and requires exact zero support on all five work wires.
Each history has amplitude factor 1/2. Eight-bit circuit construction and
classical evaluation do not claim eight-bit exhaustive quantum simulation.

| Resource category | Observed or independently modeled value |
| --- | --- |
| Eight-bit emitted primitives | 168 total, comprising X 10, H 21, T 14, T-adjoint 14, CX 81, S 7, measurement 7, conditional CZ 7 and reset 7 |
| Independent AddK model | 7 AND blocks, 48 Clifford, 7 measurements, 28 T equivalents, peak logical width 23 |
| Complement outside AddK model | 8 emitted X gates |
| Physical IR capacity | 8 target wires plus 8 constant-load and 7 carry wires |
| Classical feedback ownership | 7 caller-owned measurement slots |
| Small exhaustive coherent check | 3 target, 5 clean work, 1 spectator, all 4 measurement histories |

This tutorial makes no physical hardware estimate. The abstract model's Clifford
count is distinct from the decomposed primitive count. The existing one-bit
signed constant-load defect remains outside the fixed eight-bit tutorial. Prior
source and native arithmetic admission boundaries remain documented by their
owning audits.

Concentrated regression passes 116 cases without skips. Fourteen warnings come
from the unchanged source deprecation path. Strict mypy covers the complete new
test and all extracted native code. Native notebook cells are individually
formatted with Black because the environment lacks Black's optional Jupyter
formatter dependency. Ruff checks the notebook directly. Scope, architecture,
IR and protected API checks are recorded alongside the batch logs.

Persistent evidence is under `ftqc-reference-env/g2-wiring-*`. Source hashing,
complete-cell execution, all signed values, primitive feedback, models and
exhaustive small histories live in `tests/unit/test_register_wiring_notebook.py`.
The dashboard's default-density arithmetic discrepancy remains independently
open in `PHYSICAL_RESOURCE_DASHBOARD_AUDIT.md`.
