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

The scoped Hubbard selection mapping also supplies shared external-control
constructors and exact directional pairs. Negative, grouped, shaped and packed
controls reuse the existing predicate owners; unknown children remain refused.
See the [constructor audit](../../../docs/development/HUBBARD_EXTERNAL_CONTROL_CONSTRUCTORS.md)
for source codec failures and the nested source control-adder exception.

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

QREF conversion stays in `qref_resources.py` and `_qref_ports.py`. Optional live
SDK metadata enters through Ecosystem's `_qref_source.py`, targeting the frozen
Qualtran 0.7.0 input contract. It converts all source objects before calling
these exporters and is removable independently if that external input contract
is retired. Native Programs and metadata export do not depend on this adapter.
Start with `tests/unit/test_qref_source_capture.py`: it compares complete large
schemas, isolated scalar occurrences and source-free JSON consumers. QREF
resources are declared model data; export emits no physical instructions.

`legacy_su2_factories.controlled_su2_factories()` supplies scoped SU2 controls for
callers that explicitly need them. It tracks actual constructor identity, reuses
CRz/ZPow owners and preserves the source-atomic controlled-Rx boundary. Default
normal SU2 mapping remains unchanged. Run `tests/unit/test_controlled_su2_source.py`
for both polarities, adjoints, complete complex operators, symbolic binding and
SDK-free append. The [SU2 constructor audit](../../../docs/development/RESOLVER_SU2_AUDIT.md)
separates primitive inventories, declared models and source limitations.

`legacy_hamiltonian_gqsp_factories.controlled_hamiltonian_gqsp_factories` accepts
separate ordinary and controlled GQSP contexts, associated with the actual
resolved walk/time/precision. It reuses the existing positive-control protocol
and negative predicate wrapper; both Always PREPARE calls stay uncontrolled.
Missing contexts fail closed. Numerical coefficient and phase derivation belongs
to Simulation, not this constructor. Start with
`tests/unit/test_controlled_hamiltonian_gqsp_source.py` and the
[complete source audit](../../../docs/development/HAMILTONIAN_GQSP_CONSTRUCTOR_AUDIT.md).
