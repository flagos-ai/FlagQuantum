# Experimental artifact inspection

Updated: 2026-09-10

`flagquantum.experimental.artifacts` is a read-only preview for inspecting and
canonically reproducing supported ProgramArtifact and compilation-evidence JSON.
It does not construct, compile, execute, deploy, submit, migrate, or repair an
artifact.

```python
from flagquantum.experimental.artifacts import (
    dump_program_artifact,
    load_program_artifact,
)

view = load_program_artifact(received_json)
print(view.version, view.kind, view.producer, view.identity)
canonical_json = dump_program_artifact(view)
```

Compilation evidence uses the corresponding functions:

```python
from flagquantum.experimental.artifacts import (
    dump_compilation_evidence,
    load_compilation_evidence,
)

evidence = load_compilation_evidence(received_evidence_json)
print(evidence.version, evidence.producer, evidence.identity)
canonical_evidence_json = dump_compilation_evidence(evidence)
```

The loaders accept JSON text, not paths, URLs, bytes, streams, or provider handles.
They reject malformed data, duplicate keys, unsupported versions, unknown fields,
identity mismatches, semantic violations, and oversized input. The dumpers accept
only values returned as the matching frozen role view.

The preview deliberately hides concrete v1/v2/v3 classes. Use `view.version` for
inspection and `view.identity` for the verified envelope or bundle identity. For a
legacy v1 artifact, `payload_sha256` is `None` because that field is not present in
its schema.

This namespace is experimental. Stable promotion, artifact construction,
parameter binding, compilation, Runtime verification, Deployment preparation,
provider submission, and QEC/FTOC semantics require separate review and approval.
