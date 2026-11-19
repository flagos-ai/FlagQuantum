# Atomic interface original-test acceptance

This audit covers all four test functions in frozen Qualtran v0.7.0
`qualtran/_infra/bloq_test.py`, Git blob
`fcc5ddd2a2ae3ab03138b39ec9b78c40d8a0cf89`, at revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source file is unchanged.
The original file passes all four cases with PennyLane 0.44.1 and the isolated
Jupyter kernel. Its three warnings are the original notebook marker and two
fixture-class collection warnings. The first sandbox attempt refused local
socket binding; the authorized local-kernel run passed.

| Original case | Native owner and verified behavior |
| --- | --- |
| `test_bloq` | Existing AtomicRegisterFixture, Core Signature and primitive-backed FunctionRecord preserve two one-bit THRU ports, native label spelling and atomic refusal. |
| `test_as_pl_op` | Existing PennyLane export/import retains the CNOT matrix and instruction. The original FromBloq wrapper assertion also passes. Native export uses the native PennyLane CNOT representation. |
| `test_as_composite_bloq` | Existing explicit root wrapping creates one call to the atomic definition. Repeated selection retains that one-call composite. An atomic child cannot be expanded as a composite body. |
| `test_notebook` | All 70 original infrastructure notebook cells execute in the source kernel. Q-0144's native whole-notebook mapping and its direct dependencies are reused and rerun through test_register_component_infrastructure.py. |

The native test uses ordinary `circuit.append(op)` and compares complete complex
states with an entangled spectator for X and CNOT. No work bits or measurements
are introduced. One emitted primitive is counted separately from the original
TestAtom synthetic model of 100 T gates. Source labels lose their Test prefix in
the native fixture; this is an explicit naming adaptation. A Program already
represents a composite, so there is no redundant public as_composite method.

The concentrated regression passes 1,158 cases without skips. The direct
atomic/root group passes 48 cases. Strict typing, Black, Ruff, scope,
architecture and internal IR checks pass for the new test. Dependency policy
records 689 whole-repository findings and zero batch findings. Existing default
suite evidence retains its unrelated example-registry failure. Logs use
`g2-bloq-`. This closes Q-0154. Q-0156's broader source interface remains open,
including symbolic SDK gates that the source can retain and export before
numeric parameter resolution.
