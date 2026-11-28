# Generalized QSP complete source responsibility audit

The frozen v0.7.0 reference is
`bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0654 source blob is
`5848dcb4785697563fca6173e499015e504b233b`; Q-0655 test blob is
`c925da2a851888c4ceb3c8dbc1893c93e3c7602d`.

Algorithms owns the operation declaration, named append binding and declared
source call graph. Simulation owns complementary coefficients and phase factors.
Compiler owns physical signal emission and generic graph aggregation. Core owns
typed register signatures and symbolic arithmetic. Shape-only coefficient values
reuse Simulation's existing `SymbolicArrayShape` and `value_length`.

This batch fills the existing finite operation's shape-only constructor/factory,
symbolic classification, synthesis-precision metadata, typed signature composition
and source graph declaration. Actual coefficient arrays retain the accepted
execution path. Symbolic coefficients and powers fail before physical emission.
The source symbolic factory discards the supplied precision and uses the direct
constructor default. That quirk remains explicit and tested. Complete source
members, original tests and the notebook receive individual dispositions below.

## Complete member and asset disposition

| Source member | Native owner and acceptance |
|---|---|
| `qsp_complementary_polynomial` | Accepted Simulation root solver, retaining root order, unit-pair tests, reciprocal checks and degenerate source failures. |
| `qsp_phase_factors` | Accepted Simulation phase recursion with full complex rotation matrices and source safe-angle threshold. |
| `_to_tuple`, polynomial validators | Existing finite tuple snapshots extended to preserve `SymbolicArrayShape`; concrete first-dimension degree checks and symbolic unknown lengths match source. |
| `signature` | `source_gqsp_signature` prepends a QBit signal to complete typed descriptors. Instance scalar binding signatures use existing opaque layouts. Typed declarations remain separate from physical port admission. |
| `from_qsp_polynomial` | Accepted finite grid/complement chain and shape-only source branch. Shape-only calls skip verification/numerics and preserve the source's ignored precision argument. |
| `_qsp_phases` | Existing cached `_phases`; shape-only values fail, finite coefficients with symbolic Laurent power can still expose phases. Native error text identifies its own descriptor. |
| `_eps_per_rotation` | `precision_per_rotation` retains precision divided by coefficient count plus one, including symbolic arithmetic. |
| `signal_rotations` | Accepted `signal_rotation_calls` and Compiler SU2 emission retain numerical angles, initial lambda and child precision. |
| `decompose_from_registers` | Accepted `make_on`/`emit`, stored/shaped/entangling/measured unitary providers and explicit scratch/slot ownership. Symbolic coefficients/powers fail before invoking the unitary callback. |
| `is_symbolic` | Shared Core classifier on P, Q and Laurent power, with the same exclusion of precision alone. |
| `build_call_graph` | `source_generalized_qsp_graph` uses shared symbolic graphs and Core Max/Min/zero arithmetic, preserving equality-based child merging and first occurrence order. Existing finite ComponentCost model retains explicit child pricing and widths. |
| Three example factories | All original coefficient pairs, X unitary and powers 0/1/5 match source through ordinary append, every full operator column, a coherent complex state and clean idle. |
| DocSpec | Native complete notebook maps all three factories, signatures, graph and append/resource example. |
| Q-0648 initializer | Frozen blob `15780de77b1637deab59d60722c84f30b3d8e75d` contains only the copyright/license header, with no imports, definitions or exports. Native domain owners retain attribution. Its empty export responsibility closes. |

## Original test responsibility

| Original group | Evidence and disposition |
|---|---|
| Example autotests and Ising factory | All original factories execute; three original serialization xfails retain source reasons. Native factory examples and accepted Ising/Hamiltonian consumers cover binding. |
| Root complements, real complements | Original degree 100 slow cases included in the complete run. Accepted native root/phase conformance reused. |
| Real/complex random-unitary GQSP | Complete original cases run. Native stored, shaped, entangling and measured providers retain full amplitudes, operand/dependency order and cleanup. |
| Concrete and shape-only call graphs | Original cases run; all twelve shape/power combinations and symbolic expressions compare native leaf multiplicities, with caller-key replacement and child merging. |
| `SymbolicGQSP` and symbolic-signal tests | Original algebraic matrix/upper-bound tests, including degree 10 slow cases, run. Native phases already match unchanged rotations exactly. This helper is a reference assertion asset. |
| Jacobi-Anger complements | Original supported cases run. Three precision 1e-9 cases retain upstream Issue #860 skips. Their intended acceptance remains open. |
| Notebook | All twelve original cells and native fourteen-cell artifact execute. Original isolated-kernel result is recorded in the notebook audit. |

The complete non-notebook original run passes 135, retains three Issue #860 skips
and three original serialization xfails, deselects only the notebook, and emits
one original TestGWRAtom collection warning. Frozen source and test blobs are
verified. The native batch passes 458 cases with one IPython temporary-directory
warning. Strict mypy covers both owners, both tests and every extracted notebook
code cell. Black/Ruff, Algorithms scope, architecture, multi-level IR and
protected API checks pass. Four notebook graphs render and pass visual inspection.

## Resource accounts and remaining boundaries

The numerical owners emit zero quantum instructions. The source example's
ordinary append emits 6 X,4 phase,4 Rz,2 Rx,2 H and1 controlled phase instruction.
It borrows zero scratch, has zero measurements and preserves one idle zero qubit.
The independently chosen child model declares 6 continuous rotations and 3 Clifford
gates at peak 2 logical qubits. This model prices each source SU2 as three rotations
and the open-controlled X as three Clifford gates. The physical phase inventory
and modeled synthesis categories retain their separate accounting.

Q-0654/Q-0655 remain open for the original Issue #860 high-precision requirements,
degenerate root/source numerical failures and complete generic typed physical
admission. Verification with no explicit FFT provider retains the previously
documented native refusal. A generic foreign SDK MatrixGate lowering may use a
different primitive synthesis than the named native provider. Comparable source
children, controls, operands, feedback and full amplitudes use the accepted
provider evidence. Physical synthesis prices remain caller-owned. The symbolic
factory precision quirk is retained and explicitly exposed. These dispositions
preserve the original acceptance requirements.

## Complete migration acceptance review, 2026-11-28

This review supersedes the earlier open-parent disposition under the explicitly
approved faithful-migration closure rule. Q-0654 and Q-0655 are accepted once the
batch receipt below is published. Issue #860's three original skips, degenerate
root failures and original serialization xfails remain source defects, with
the original inputs and receipts retained; they are not correctness certificates.

The remaining native implementation gap was logical composition above a genuinely
undecomposable U. The unchanged source successfully builds
`GeneralizedQSP(THCRotations(2,4,8), P=(0,1), Q=(0,0))` with three calls. The
published native baseline f2d387e79 refused it before composition. The shared
component now retains typed declarations while physical binding still refuses
missing bodies. Available negative-control wrappers and SU2 children are fully
preserved, not replaced by opaque higher-level placeholders. Four Laurent powers
0/1/2/5 compare the actual source root and all available descendants, every
operand and return, exact source SU2 parameters, source-free reconstruction,
seven-category models and logical peak 14. No whole physical inventory or
numerical cleanup is claimed for the opaque THC unitary.

The executable path additionally covers a seeded three-target matrix U at
powers 0/1/5, using the existing arbitrary-width matrix atom owner. Complete
complex operators, coherent spectators, reversed operands, append/JSON and clean
private work match the unchanged source. This does not assert a MatrixGate QEC
synthesis model. Existing scalar, packed/shaped, entangling and measured U
execution evidence remains separate and is reused. A supplied exact inverse or
control role is never inferred from a matching signature.

The frozen source/test blobs above and real Kickmix Simulator are checked in
`test_gqsp_logical_source_closure.py`. The original complete non-notebook source
receipt remains 135 passed, three source skips and three source xfails, including
all original slow numerical cases. The final G2 receipt records concentrated
regressions in both SDK profiles, strict checks and independent primitive/model
accounts. Q-0540's actual Hubbard composition and generic controlled-Hamiltonian
ownership remain a separate open parent; no closure is inferred for it.
