# Controlled register XOR

`compiler/registers/controlled_register_xor.py` owns two concrete entries:
`controlled_xor_ir` binds one positive control; `grouped_controlled_xor_ir` binds
existing `ControlRegister` groups around that body. Both preserve the source word
and XOR it into any equal-width target. Registers use MSB-first order. The
operation is self-adjoint and accepts arbitrary coherent target data.

The positive body reuses `cnot_with_controls_ir` for each source/target pair.
Structured activation reuses `controlled_component_ir`: compute a single
predicate, apply all word gates, and erase the predicate. For m controls, reserve
max(0,m-1) initially zero work wires and the same number of distinct measurement
slots. Controls, source, target and work must be globally disjoint. Quantum work
returns to zero; classical slots retain measured results. Single-bit controls
need no work or measurement. Signed encodings require no arithmetic conversion:
XOR acts on the stored bit strings.

The pinned, unmodified source is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`: `Xor` in
`qualtran/bloqs/arithmetic/bitwise.py`, generic `Controlled` in
`qualtran/_infra/controlled.py`, `ControlledViaAnd`, and CNOT's Toffoli control
specialization. Positive `Controlled(Xor)` threads the control through the
original per-bit CNOTs; structured controls compute the predicate once around
that complete body. The same shared FlagQuantum owners implement these steps.

## Common gate set and resource interpretation

Upstream Toffoli is an atomic compilation target: `Toffoli.decompose_bloq`
explicitly raises `DecomposeTypeError`. The common executable comparison gate set
therefore retains atomic CCX alongside Clifford/T and measured predicate erasure.
No relative-phase Toffoli substitution or invented source decomposition is used.
`source_grouped_xor_cost` composes width-many logical Toffolis with the existing
predicate model and both live data registers. T-equivalent conversion is a model
query, not an emitted T gate measurement.

Run the append example:

```sh
python -m examples.algorithms.controlled_register_xor
```

`circuit.append(ControlledRegisterXor())` preserves control word 2 and source 3,
changes target 1 to 2, and clears predicate work. The example checks 16 actual
trajectories; unit tests prove all small measurement histories separately.

| Two-bit example | Executable instructions | Original resource model |
|---|---:|---:|
| Atomic CCX | 2 | 2 Toffoli |
| T + TDG outside CCX | 4 | 1 And, converted to 4 T |
| Measurements | 1 | 1 |
| Work / total qubits | 1 / 7 | Peak 7 |
| Clifford | 4 X, 3 H, 6 CX, 1 S, 1 conditional CZ | 1 Clifford, with Toffoli/And separate |
| Reset | 1 | Part of measured predicate erasure |
| Model T equivalents | No claim of emitted total T | 12 at 4 T per Toffoli and And |

## Acceptance

G2-BIT04 adds 106 tests. Ninety original-graph configurations compare operand
bindings, gate dependency order, outputs, measured feedback, all seven QEC
categories in both profiles and QubitCount: ten control specifications, three
widths (1/2/5), and unsigned/two's-complement/Montgomery data words. The source
walker retains original Split/Join routing and atomic Toffoli identities.

Eight coherent tests cover all input amplitudes including an entangled spectator,
both single-bit polarities and two/three-bit activation, every erasure history,
zero work, source preservation and double-application restoration. Eight negative
tests reject invalid widths and overlapping/invalid data ports. The source is
not patched. Compiler work only emits/counts instructions; numerical execution
remains in existing simulation/runtime paths.

The concentrated acceptance log records 435 passing control/bitwise/AddK tests,
without skips. Twelve existing controlled-Toffoli/LCU tensor warnings are preserved without suppression or a new LCU
numerical claim. Symbolic widths and a full Clifford+T decomposition of the atomic
upstream Toffoli are outside this slice. The parent migration entry remains open.
