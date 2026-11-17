# State-preparation package source-file audit

Q-0690 covers `qualtran/bloqs/state_preparation/__init__.py` in Qualtran
v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The unchanged
initializer has Git blob `024a57c221ae73f95645dbaa4247f423b037a932`, three
import statements and four class exports.

| Locked export | FlagQuantum owner | Native consumer |
| --- | --- | --- |
| `PrepareUniformSuperposition` | Compiler uniform preparation emitter | `UniformPreparationExample` |
| `StatePreparationAliasSampling` | Algorithms alias preparation emitter | `dense_alias_example()` |
| `SparseStatePreparationAliasSampling` | The same alias emitter with indexed table and sparse register binding | `sparse_alias_example()` |
| `StatePreparationViaRotations` | Algorithms `RotationPreparation.emit` | `RotationPreparationExample` |

The package check reads the full source import AST and confirms each export
is the object from its source module. Four runtime checks instrument the
actual owners and confirm that `circuit.append(op)` passes the concrete
state count, the original dense or sparse table and index binding, or the
phase gradient to the expected emitter. Three isolated notebook kernels
exercise those consumers. The two alias variants share their executable
owner and retain their separate table layouts.

The initializer declares no circuit, workspace or resource estimate.
The four concrete routes inherit the gate, complex amplitude, workspace,
measurement and resource evidence from
[uniform preparation](UNIFORM_PREPARATION_FILE_AUDIT.md),
[alias preparation](ALIAS_PREPARATION_FILE_AUDIT.md) and
[rotation preparation](ROTATION_PREPARATION_FILE_AUDIT.md). Q-0695, Q-0700
and Q-0703 retain their symbolic execution, resource, dtype and port
boundaries. Q-0693 separately tracks the source PREPARE oracle contract.
