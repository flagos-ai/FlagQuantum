# UP-06 scalar-controlled Pauli evolution and classical angle expressions

Private Qmod adaptation admits one scalar qbit/one-bit unsigned qnum or a static
qbit-array element as the control of pure Pauli evolution branches. A bare bit,
bit==0/1 or bit!=0/1 selects one branch; else selects its complement coherently.
Each branch specializes through existing static evolution/helper lowering and
must emit only previously uncontrolled qreg.pauli_rotation effects. Existing
control_index represents the result; no opcode, physical template or public
export is added. Negative controls use paired X conjugations. Original term
order, all repetitions, finite-angle bounds and all resource caps remain active.
The identity Pauli word becomes a control-only phase and must never be dropped.

Control and the complete target storage must be disjoint and initialized.
Nested controlled Pauli effects, allocation, Output escape, initialization,
mixed gate/Pauli branches and unsupported effects refuse. Existing reversible
X-helper controls keep their established route and semantics. Empty/time-zero
Pauli effects retain owned initialization/alias checks before optimization.

Gate angles and phase coefficients admit finite closed classical +, -, *, /
expressions, unary minus, parentheses, pi, bound real/integer parameters and
existing checked static array elements. Division by zero, quantum/unknown
values and intermediate/final magnitude beyond finite +/-2**53 refuse. Unary
minus keeps ordinary expression precedence. Numeric matrix/preparation literals
keep their existing scanner and restrictions, moved to one private lexical
helper rather than broadening those input domains.

Accept the byte-identical bpde.qmod at delta_e=0.37 and -0.81 against independent
public Circuit and matrix algebra full-state references, without phase alignment.
Keep time=4, order=2, all eight Trotter steps and all source Hamiltonian terms.
Test that removing identity terms changes interference, not just global phase.
The dropped evolved state is semantic data: retain its physical state, never
claim it is zero workspace or reset it. Check actual workspace separately.
Test I/X/Y/Z and multi-bit numeric order, positive/negative controls and else,
helper borrowing, overlap/capture/escape refusals, initialization, time-zero,
finite angle arithmetic, operation/width/depth/memory budgets and source caps.
Provide runnable Qmod/native phase-composition/Circuit examples, explicitly
recording any existing native entry limitations. Recheck all 216 originals,
preserve prior source/license/AO/PF/resource evidence and the separately tracked
partial-exponential amplitude mismatch. No integration PR/full CI before the
complete original-corpus acceptance gate.
