# Compiler format ownership

This directory translates stored circuit/component declarations into native IR
and typed Programs. External constructor identifiers are interchange metadata;
they do not define a second runtime or user-facing package. Ecosystem converts
live SDK objects and numerical coefficient payloads before invoking these
owners. Compiler must not execute statevectors, choose backends or import
Algorithms or Simulation to reconstruct a Program.

Use legacy_component_resolution.resolve_component_program with an explicit
factory mapping. Existing legacy_component_library codecs own the wire format.
Individual factory modules validate constructor fields and delegate to shared
register component owners. Exact inverse/control recipes remain scoped to the
factory mapping for one compilation. Unknown same-signature children must be
rejected unless their owner supplies explicit roles and directional Programs.
Do not infer semantics from names, widths, matrices or cost counts.

For packed Hubbard oracles, legacy_hubbard_packed_context.py composes the existing
packed PREPARE/SELECT, Hubbard and block-control dependencies. Its required angles
callback replaces numerical coefficient conversion. Ecosystem's existing packed
entry supplies that callback. Shared signatures and literal SELECT Programs
come from Compiler; no Algorithms object is constructed for parsing.

A short contributor path is tests/unit/test_hubbard_packed_context.py. The test
serializes the actual frozen source example, replaces only the coefficient
converter and verifies the entire resulting Program while source, Algorithms
and numerical execution owners are unavailable. Run it with the locked optional
reference dependencies installed:

```bash
python -m pytest tests/unit/test_hubbard_packed_context.py -q
```

When changing a constructor, preserve metadata validation and source failures,
then compare ordered dependencies and complete bound instructions. Reuse the
corresponding full-complex, measurement-branch, cleanup and resource evidence
only after proving identical Programs or bindings. Report measured primitive
instructions separately from analytical model nodes. Update the relevant
complete-source audit with any unsupported boundary.
