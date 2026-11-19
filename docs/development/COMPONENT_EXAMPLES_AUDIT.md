# Component example metadata acceptance

This batch integrates the complete locked `qualtran/_infra/bloq_example.py`
and `bloq_example_test.py` into the native Testing domain. The reference commit
is `bcb17bb25360ed68ce3bc084645db3e929799389`. The source blobs are
`18abbba75b651863eadf164f73a61ed2670195e3` and
`3075f7c508d6965bc4e52711818ab31c33e2d6ef` respectively. Both original tests run
unchanged, and the native acceptance scenarios independently verify the same
factory and metadata behavior.

## Ownership and full source map

`flagquantum/testing/component_examples.py` owns example metadata used by
conformance tests and documentation. It neither defines IR nor decides how a
component executes. It imports only standard-library types and dataclasses.
The existing Compiler, Simulation and Drawer remain the consumers' owners.
The module has no root-package export.

| Source symbol or behavior | Native implementation |
| --- | --- |
| `BloqExample` | `ComponentExample`, with `factory`, `component_type` and `description` fields |
| `make`, `__call__` | Fresh, lazy invocation of the stored factory |
| Factory equality, excluded hash and repr | Dataclass field with `hash=False`, `repr=False` |
| Generalizer sequence converter | Freeze a supplied sequence as a tuple while retaining callable identity |
| `_name_from_func_name` | Strip every leading underscore from the factory name |
| `_bloq_cls_from_func_annotation` | Require an actual class return annotation; preserve the missing-annotation failure |
| `bloq_example` | `component_example`, directly or with keyword-configured generalizers |
| `_to_tuple` | Capture the documentation example iterable exactly once |
| `BloqDocSpec` | `ComponentDocumentation`, retaining explicit import and graph selection |
| Default import | Infer the containing package using the source parent-module rule |
| Default graph example | First example, or None for an empty iterable |
| Explicit graph None | Suppress graph selection even when examples exist |

`ComponentDocumentation` is keyword-only and immutable, as is the source
specification. An explicit graph example outside the stored example list is
accepted, preserving the source behavior despite the stronger claim in its
prose documentation. Native code does not invent a membership validator.

## Execution and verification

Run `python -m examples.algorithms.component_example_catalog`. The example uses
the metadata-selected factory to instantiate the existing three-bit `WordSwap`,
then calls `circuit.append(operation.make_on(...))`. The full complex state,
including a spectator, matches exactly. The emitted circuit contains nine CX
gates, no measurement and no work wires. These are observed physical primitives;
the metadata module contributes no gates and performs no resource estimation.
Generalizers remain explicit callbacks for the consuming resource query.

The original two tests run unchanged. Native checks additionally cover direct
and decorated factories, callable/list/tuple/empty generalizers, freezing a
mutable source sequence, lazy fresh calls, exact exception identity, missing
and forward-reference annotations, immutable equality/hash/repr, empty and
nonempty documentation defaults, one-pass iterables, explicit imports and graph
suppression. The metadata, declaration and tutorial regression passes 34 cases
without skips. Strict typing covers all three new Python files. Formatting,
lint, scope and architecture checks pass. The full dependency scan retains the
706 existing findings, with none in this batch's paths.

## Deliberate native boundaries

The native metadata accepts any concrete return class. FlagQuantum operations
are independent dataclasses and do not inherit a common Bloq base. Requiring
that foreign base would reintroduce the source package's architecture. The
factory is not executed to infer or validate its result type.

The parent-package import inference can produce an unusable import for a class
that is not exported by its package. This is also a source behavior. The runnable
example supplies its real module import explicitly, and the test records the
inferred form independently. Forward string annotations are not evaluated.

The source decorator accepts arbitrary keyword arguments but ignores them in
its implementation. The native decorator exposes only the supported generalizer
argument and rejects unknown keywords instead of silently discarding them.
Native metadata fields are statically typed; arbitrary invalid field values
admitted by the unvalidated source constructor are outside this acceptance.
No serializer, registry, automatic notebook generator or numerical protocol was
added by this metadata port.
