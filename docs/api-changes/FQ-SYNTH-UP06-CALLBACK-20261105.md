# UP-06 closed quantum callback specialization

Private Qmod adaptation admits borrowed `qfunc(qbit[])` and
`qfunc(qnum,qbit)` parameters in addition to the existing real callback profile.
Named functions or nonempty closed lambdas are statically specialized into the
existing first-order statement representation before oracle normalization.
No callable enters owned IR and no user host code executes. Quantum callback
ports borrow initialized unsigned values; Output, consumption, quantum captures,
aliases, signature mismatches and recursive expansion refuse. Expansion is
bounded by 4096 statements and sixteen call levels, including unused wrapper
validation. Callback arrays and arbitrary captures remain outside this profile.

The quantum wrappers may compose Hadamard transforms and existing oracle calls.
A within preparation consisting solely of borrowed Hadamard transforms is
self-inverse and expands as preparation, apply, reverse preparation. This does
not authorize general gate scopes or changes of ownership. Borrowed numeric
widths are inferred from the caller, never guessed. Const numeric predicates
reuse existing checked XOR-to-phase normalization; integer min/max evaluation
uses the existing local eight-input and polynomial budgets.

Accept unchanged hidden_shift.qmod and deutsch_jozsa.qmod against independent
public Circuit and analytic full-state references, without phase alignment.
Check source numeric bit order, clean workspace and real resource budgets.
Reject captures, Output escape, aliasing, recursion, unused invalid wrappers,
empty callbacks and wrong arity/kinds. Preserve existing real callbacks and
prior resources. Recheck all 216 original sources and retain source/license
hashes and original AO/PF baselines. Update delivery evidence only after local
acceptance; no integration PR/full CI before complete corpus acceptance.
