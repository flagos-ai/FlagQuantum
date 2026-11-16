# Native time marching with complete QSVT phases and coherent counter

Status: accepted locally. 23 new time-marching tests and 31 exponential
regressions pass, along with 29 catalog/API tests and strict typing of five
Python files. Final fresh 216-original hashes, outcomes and resources exactly
match prior evidence; approved six pass and all 29 production pins are unchanged.
Native coverage is 203/216; Notebook workflows remain 24/213.

Native: 13 physical bits, one compiler work bit, one explicit clean marker,
18,344 operations, depth 15,762. Original: 15 physical bits, four work bits,
5,118 operations, depth 4,355. Independent-reference errors are below 2.1e-13
and all clean marker/work leakage is below 1e-22. Native gate overhead is a
retained optimization gap.

Retain the complete frozen time-marching program: two-bit index and time words,
block bit, QSVT signal/selector bits, unused amplification bit and three-bit
counter. Initialize index uniformly and counter to four. Apply four intervals
[0,0.5], [0.5,1], [1,1.5], [1.5,2], each with all seven even/eight odd phases,
the final selector-controlled block and projected phase, then decrement the
counter modulo eight on the exact five-bit success predicate. Do not substitute
a final-state table or change phase sequences, interval count or unused storage.

Reuse native real bindings, raw views, rotations, phase polynomials, checked
adjoint, pure helper coherent control and modular decrement. Input selector
preparation is RY(pi/2) with inverse RY(-pi/2), as in the source's in-place
product preparation; Hadamard has the same action only on zero input and is not
a valid replacement of this borrowed unitary. Time-register Hadamards remain H.

An explicit clean predicate marker may implement the five-bit success flag with
H/phase/H before and after controlled decrement. Prove marker zero and compiler
work zero separately. This avoids unsupported raw-array predicate expressions
without changing their semantics. No production API, cap or ownership rule
changes are planned.

Independent Circuit acceptance must use explicit controlled-Y rotations,
projector-X/RZ/X matrices, inverse order and conditionally permuted counter.
Compare the complete complex state with the frozen original, including global
phase, register layout and every clean work/marker amplitude. Exercise changed
nonzero time scale and phases, nonzero coherent counter wrap and borrowed-block
inverse restoration; reject overlapping actuals and resource-budget excesses.
Preserve frozen original resources and report native overhead honestly.

After acceptance run a fresh 216-original plus approved-six replay, archive
immutable evidence and update catalog/ledger. Notebook workflows remain separate
at 24/213. Classification is `single_device_fast_path`. No push, integration PR
or full remote CI before all 216 native quantum-program acceptances are complete.
