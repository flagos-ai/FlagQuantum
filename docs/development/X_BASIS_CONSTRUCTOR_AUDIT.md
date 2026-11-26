# X-basis constructor dependency closure

The complete frozen x_basis.py and x_basis_test.py at Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389 retain the complete disposition in
[the file audit](X_BASIS_FILE_AUDIT.md). Compiler's existing basis-vector format
owner now maps PlusState, MinusState, PlusEffect and MinusEffect explicitly.
No SDK callback or numerical execution is placed in Compiler.

Each constructor retains the scalar q/QBit signature, RIGHT for preparation and
LEFT for effect. The preparation Program calls the shared zero/one allocator
and then the existing typed Hadamard leaf. It emits H for Plus and X then H for
Minus, exactly as the unchanged source Cirq consumer. Allocation uses the
caller's clean output wire; the output belongs to the caller. There is no
private workspace, measurement, feedback or reset. n other than one retains the
original NotImplementedError; malformed metadata and signatures are refused.

Effects retain bodyless logical declarations. Original as_cirq_op refuses both
covectors, and the native physical binder refuses them even when input ownership
is supplied. This constructor does not turn a covector into a checked release.
The existing tensor owner retains arbitrary unnormalized complex contraction;
its complete-file evidence is unchanged. Generic adjoints are not invented.

Actual serialized libraries at depths zero, one and two produce identical
Programs. Tests compare every emitted primitive and operand against the actual
original CircuitOperation leaves, complete complex state including a coherent
borrowed spectator, the empty applicable history and normal circuit.append(op).
The single clean output wire is placed in reversed physical order relative to
the spectator. Source tensor output and original Cirq output agree absolutely.
Existing whole-file controlled-X, measurement histories, tensor contraction,
classical callbacks, diagrams and foreign consumers are reused unchanged.

Measured preparation inventories are one H and X/H, respectively. Original
QEC model costs for all four vector constructors are zero; logical peak width is
one. Effects have no measured physical inventory because conversion is refused.
Primitive counts and model-only estimates are separate. Constructor registration
adds no complete-parent count: Q-0268 and Q-0269 were already accepted.
