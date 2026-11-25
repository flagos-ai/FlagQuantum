# Grid kinetic energy file audit

This batch covers Q-0427 and Q-0428, the kinetic implementation and its tests, from Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0428 closes with the original examples and their native comparison. The initial audit kept Q-0427 open for the source phase and cleanup boundaries below; the complete source constructor and branch closure is recorded at the end.

The source declares one opaque through register `system` shaped `(electrons, 3)`. Each axis word has `(grid_points - 1).bit_length() + 1` wires. The source documentation calls the extra bit a sign bit. The actual `SumOfSquares` accepts the axis words as unsigned values, which the native circuit also uses. FlagQuantum binds wires in electron, axis, then big-endian word order. The native scratch and sum register belong to the caller and are restored by the operation.

The locked top-level decomposition sends each electron's three axes to `SumOfSquares`, routes its result to `QuantumVariableRotation`, and passes the returned phase register to `Free`. The direct source comparisons check both shaped endpoints, axes, result/phase widths, child order and operands. They include the three original parameter cases, the documentation example and smaller sizes, and compare gate classes and logical peaks. The unchanged original suite returned six passes and two skips.

The native dynamic operator supports `circuit.append(op)`. It uses exponentiated unsigned phase weights supplied by the caller and reapplies the sum computation to erase its result. Forward and adjoint on the two-electron, one-bit grid preserve complex amplitudes and clear scratch on all 16 measurement histories. The two-bit test covers all coordinate values with sampled histories and verifies independent ownership of an occupied classical slot.

The two-electron fixture emits 16 T-family instructions and four measurements. The source budget estimates two Toffoli and a peak of ten logical qubits. Its QVR leaves give rotation multiplicity without fixed angles. Its source `Free` is also a different operation from the native uncomputation. Together these boundaries prevent comparison against a complete locked primitive stream.

The shared chemistry drawer now preserves the fixed `U_T(dt)` root and every
indexed axis TextBox. `test_grid_energy_symbols.py` compares the complete
register layout to the immutable source at one, two and three electrons and
one, three and eight grid points. JSON round trips and Matplotlib rendering
preserve the labels and geometry. This presentation mapping adds no execution
gates and does not close the unspecified-angle or cleanup boundaries.


## Complete source constructor contract before implementation

Compiler owns the original opaque system word shaped (num_elec,3), with
coordinate width (num_grid-1).bit_length()+1. Array order is electron, axis,
then MSB-first word storage. The actual original recipe calls SumOfSquares,
QVR and clean Free for each electron and returns its updated system handles.
The RIGHT square result is transferred to QVR, then consumed by Free. It is
not a private lease that can be silently erased instead. Source constructors
must retain this literal recipe and bodyless source leaves; no numerical
coefficient is supplied by the root. Algorithms keeps its explicitly selected
compute/absolute-phase/uncompute operation and existing positive admission.

Actual source libraries at depths1/2 already resolve through the generic stored
graph resolver. Only depth0 lacks a constructor. The new factory must produce
the same normalized typed call graph as those actual libraries, reusing the
existing builder and source child factories rather than introducing adapters.
Symbolic electron counts may remain logical declarations when no finite loop
exists; symbolic grid width remains unsupported by the original bit_length.


## Complete five-definition source and dependency closure

| Frozen definition or asset | Owned implementation and evidence |
|---|---|
| KineticEnergy and num_elec/num_grid fields | Compiler source factory retains original constructor metadata; it does not add a numerical phase coefficient. |
| KineticEnergy.signature | Shared grid_kinetic_signature owns the original opaque electron/axis shape and bit_length sizing. Existing Algorithms positive-width signature delegates without changing admission. |
| KineticEnergy.wire_symbol | Existing chemistry drawer retains U_T(dt), all indexed axis boxes, complete source geometry, JSON and rendered labels. Prior whole drawing evidence is reused. |
| KineticEnergy.build_composite_bloq | New shallow factory emits the literal per-electron SumOfSquares, QVR and clean Free using existing typed builder and child owners. Entire call/operand/result/return SSA graphs, child signatures and Free body equal actual original stored graphs after names alone are normalized. |
| _kinetic_energy and _KINETIC_ENERGY | Original factory and DocSpec retain 12 electrons/17 grid points, shape (12,3) and six-bit coordinate words; source/test blobs are verified. |

All five original definitions and the DocSpec have explicit disposition. Four
profiles, including the full original example, bind actual source libraries at
depths 0/1/2. Existing generic stored-graph resolution remains unchanged; the
new constructor fills only the missing shallow-metadata route. No new dtype
views or registry mechanism are introduced. Empty electron arrays preserve the
original identity without unused bodyless helpers. Symbolic electron arrays
retain a logical declaration; the original finite-loop failure remains explicit.
Symbolic grid sizing is refused because the original uses integer bit_length.
Factory construction works with reference SDK imports blocked.

Original SumOfSquares and QVR remain bodyless and physical binding refuses the
literal source Program. Original clean Free still consumes the QVR output.
This source recipe is not silently rewritten as the independent native
square/phase/square operation. The source supplies no numerical coefficient,
complete primitive stream or cleanup guarantee for arbitrary computed values.
Those limitations remain explicit; no source operator or gate equality is
claimed across missing bodies. Native numerical behavior has its own complete
contract and evidence below.

## Complete native diagonal-channel proof

For one/two electrons and one/two bits per coordinate, both forward and adjoint
are checked on every system basis word and an unchanged coherent spectator.
The largest case has 4096 system words and 8192 spectator-extended columns.
Each complex phase equals exp(plus-or-minus i * coefficient * sum(axis**2));
all result and work bits return to zero. These complete phased columns establish
the action on arbitrary complex superpositions over the admitted input span.

The certificate checks actual literal And compute/erase primitives and every
erased conjunction. Each erase history has the same positive local scalar;
composition proves the diagonal channel for every combined raw history. The
largest profile has 32 measurements. This proves all 2**32 histories without
claiming to enumerate them. Existing forced-history small tests and independent
sparse complex trajectories are reused. Eight full native CircuitIR and model
comparisons against published 0c0bfad2c are identical: the numerical operation
has not been rewritten as part of source constructor binding.

| Two-electron profile | Measured inventory and clean storage | Separate original model |
|---|---|---|
| One bit/coordinate | 72 CX, 12 H, 8 T, 8 Tdg, 4 S, 4 measure, 4 CZ, 4 reset, 8 phase; four clean result bits and one work bit; 11 circuit wires excluding spectator | 2 Toffoli, 8 generic rotations, peak 10 |
| Two bits/coordinate | 380 CX, 96 H, 64 T, 64 Tdg, 32 S, 32 measure, 32 CZ, 32 reset, 12 phase; six clean result bits and eight work bits; 26 circuit wires excluding spectator | 18 Toffoli, 12 generic rotations, peak 18 |

Measured primitive counts and actual bound storage are not substituted for the
original model categories or peak estimate. Four directional-independent count
rows are retained in g2-grid-kinetic-source-resources-20261126.json.

Final concentrated regression passes 243 cases in each Cirq lane; local
integration passes 145. Strict five-file typing, formatting, lint, both scopes,
language, links, architecture, Program IR and public API checks pass. The real
Kickmix 0.1 Simulator and both frozen source/test files are verified. Original
six passes/two source skips and complete drawing evidence are reused. Q0427
closes its original source capabilities and directly required native execution
proofs; source unknown-angle/body/Free limits remain registered, not invented.
Receipts use g2-grid-kinetic-source-*-20261126.
