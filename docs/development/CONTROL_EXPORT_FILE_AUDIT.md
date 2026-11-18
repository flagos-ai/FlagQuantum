# Complete control package exports

Q-0543 covers every import in `qualtran/bloqs/mcmt/__init__.py` at locked
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`23b8caf45e0b6c83ead160e6cfac5ab3d7d10280`. This initializer contains seven
re-exports and no executable gate body. The native test verifies its complete
order, absence of aliases and exact object identity with every defining module.

| Original export | Authoritative native owner | Existing consumer evidence |
| --- | --- | --- |
| And | Compiler `registers/primitives/and_compute.py` | `test_and_file_audit` and arithmetic consumers preserve four control polarities, exact gate/operand/feedback and compute/erase branches. |
| MultiAnd | Compiler `registers/control/multi_control.py` | `test_and_file_audit`, predicate and MCX consumers retain ordered logical prefix outputs and reverse erasure. |
| ControlledViaAnd | Compiler `registers/control/controlled_component.py` and component predicate owner | `test_controlled_via_and_file` retains grouped controls, nested attachment, positive child and measured cleanup. |
| CtrlSpecAnd | Compiler `registers/control/control_predicate.py` and shared predicate resource owner | `test_control_predicate_file` preserves integer encodings, row-major groups and MSB-first words, signatures and all seven factories. |
| MultiControlX | Compiler `registers/control/multi_control.py` | `test_register_multi_control` and `test_multi_pauli_file` compare occupied XOR targets, control polarities, all small erasure histories and work reset. |
| MultiControlZ | Same multi-control owner | The Pauli suite preserves relative phase, zero/one/many controls, full coherent spectators and feedback. |
| MultiTargetCNOT | Compiler `registers/bits/bitwise.py` fanout | The Pauli and classical controlled-bit suites retain actual logarithmic-depth fanout, dirty targets and original callback behavior. |

The user workflow remains the existing native `circuit.append(op)` examples in
`controlled_via_and.py` and `control_predicate_catalog.py`. Compiler owns binding
and gates. Simulation owns exact tensor/channel factors and numerical callbacks.
Core owns typed signatures and Program. No public package mirror or additional
control reducer is introduced. The shared export assertion now recognizes an
optional module docstring before the exact import-only body.

## Reused evidence and remaining gaps

All admitted control words retain big-endian bit order. Compute returns its
logical predicate histories, which are consumed by the measured adjoint. Prefix
outputs are not clean work until that adjoint completes. Callers supply clean
scratch and distinct measurement slots. Full complex branch states and scratch
zero are checked by the existing direct-dependency suites in this batch.

The initializer emits no primitives, allocates no work and owns no measurement
branches. Physical And compute emits four T/TDG. Erasure has its own measurement,
feedback and reset inventory. Independent QEC And/Clifford/measurement/peak
queries remain separate, including controlled-H's modeled rotations versus an
atomic physical CH instruction. No package aggregate is inferred.

This closes Q-0543's entire export responsibility. Original And symbolic
polarities/diagrams, typed integer and generic child admission, attachment
boundaries and original skips/xfails remain under Q-0545/Q-0550/Q-0553/Q-0556.
The existing `AND_FILE_AUDIT`, `CONTROLLED_VIA_AND_FILE_AUDIT`,
`CONTROL_PREDICATE_FILE_AUDIT` and `MULTI_PAULI_FILE_AUDIT` retain those details.
No reference source or test expectations were changed.

The shared batch passed 792 concentrated native/reference cases without skips
or warnings. All four changed Python test files passed strict mypy, Black and
Ruff. Compiler scope, architecture, multilevel IR and protected API checks
passed. The source initializer blobs and child reference files were unchanged.
Logs are `ftqc-reference-env/g2-package-exports-*.log`.
