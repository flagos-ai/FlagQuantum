# Checked static quantum callbacks for native QPE

Ordinary QPE in the unchanged quantum-counting, Grover-QPE and user-defined QMC
originals uses a reversible quantum subprogram rather than one literal matrix.
The native interface already accepts a zero-argument callable syntactically;
its unreleased admission now broadens from one static matrix call to one static
positional captured operation or qfunc call in that lambda. Host Python calls
are never executed. No new public name or signature is required.

Capture the callback through existing native validation and mutable-helper
borrowing. Require nonempty pure effects on initialized writable storage,
disjoint from the allocated unused one-to-eight-bit all-fractional counter.
Reject allocations, hidden ownership changes, Output publication, escapes,
Const mutation, counter use, nonunitary effects and arbitrary host callbacks.
Root pure within/adjoint composition retains these rules. Generic signed/scaled
helper borrowing and runtime classical feedback remain separate.

Extract the existing ordinary source-QPE controlled-effect transformation into
Compiler ownership. Both native and Qmod frontends call that implementation;
source parsing remains Ecosystem-owned. Reuse existing phase, controlled-X,
Pauli, indexed preparation and matrix effects and inverse Fourier. Preserve
conditional global phases and the original source controlled-CRY sequence.
The single-matrix callback retains its compact checked matrix-power path.
Repeated generic callbacks remain subject to the native 4096 effect bound and
all final resource/memory constraints. Unsupported nested controls refuse.
No new IR opcode, numerical kernel or executor is introduced.

Independent acceptance compares dense matrix powers and Fourier against all
complex amplitudes, including absolute phase, signed numeric labels, retained
environments and clean work. Test primitive/helper callbacks, every input
column for small operators, nonzero variations, inverses, conditional constant
phases, illegal captures/aliases/escapes, effect bounds and real budgets. The
three named original programs need complete source/native/Circuit acceptance
before coverage credit. QPE for molecules additionally uses power-dependent
Suzuki repetition counts; ordinary repeated callbacks do not admit it.

This successor admission preserves the earlier matrix-QPE contract as a
historical record. Standing authorization is the complete native corpus
migration. Only local implementation is authorized; full Notebook workflows,
release, hardware and performance claims remain outside this acceptance.
