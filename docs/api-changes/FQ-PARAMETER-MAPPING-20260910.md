# Parameter mapping key contract

Status: implementation authorized by the user's instruction to proceed with
the proposed Core-to-Compiler-to-Runtime parameter binding typing repair.

## Signature change

The existing `Mapping[str | Parameter, Any]` annotation requires a mapping whose
key type is the union. Python mapping keys are invariant, so it rejects both
`dict[str, Tensor]` and `dict[Parameter, Tensor]` despite their supported runtime
behavior. Replace it with `Mapping[_ParameterKey, Any]`, where `_ParameterKey`
is a private type variable bounded by `str | Parameter`.

Apply this change to `Parameter.bind`, `ParameterExpression.bind`,
`bind_parameter_value`, the internal runtime binding canonicalizer, and the two
device Double-Single expectation and parameter-shift entry points. Optional
bindings remain optional. Names, defaults, return annotations, exports,
serialization, and numerical implementations remain unchanged. Compiler and
artifact callers require no adaptations.

## Compatibility and implementation

String-only, Parameter-only, and mixed mappings are accepted statically.
Unrelated key types are rejected. No user migration is required. There is no
mapping copy or new wrapper. Direct key lookup retains its existing complexity.
Core keeps object-key precedence; the runtime canonicalizer keeps rejecting
duplicate names, including a string key and Parameter key with the same name.

Two precise key casts in `Parameter.bind` follow successful membership checks.
They express the lookup already proven valid by the mapping, without widening
the mapping to `Any`, changing values, or suppressing diagnostics. This is the
only implementation adjustment. No extra runtime key validation is introduced.

## Verification requirements

Run full-package strict mypy before and after; verify the nine binding diagnostics
disappear without checker configuration changes. Add positive static cases for
all three key forms and read-only mappings, and negative cases for integer and
object keys. Verify key precedence, missing names, value identity, gradients,
direct lookup without iteration, and runtime duplicate-name rejection. Run
parameter binding, artifacts, compiler, and Double-Single CPU tests, then the
available CPU smoke/unit tier. No accelerator capacity claim follows from this
single-device typing repair.

## Verified results

Full-package strict mypy with Python 3.12 decreased from 37 errors in 16 files
to 28 errors in nine files, checking 447 source files. All nine removed
diagnostics are the targeted binding errors. No checker settings, ignores,
or API snapshots were changed.

Focused parameter, compiler, artifact, and Double-Single tests: 286 passed.
CPU smoke/unit selection: 1,930 passed, 13 skipped, 1,802 deselected, with one
PyTorch complex-module warning. Static positive cases passed; six invalid-key
calls were rejected. The standalone static fixtures use silent import following
to isolate their assertions; package diagnostics are measured separately with
the unchanged full-package strict command. API, architecture, ownership, and
touched-file language checks passed. No GPU or remote jobs ran.
