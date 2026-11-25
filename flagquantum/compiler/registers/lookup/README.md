# Table lookup and selection

`matrix_entry_component.py` owns uniform and QROM-driven entry call graphs.
Numerical table conversion is supplied through the Ecosystem boundary and the
existing Algorithms table owner. QROM owns predicate work; the entry graph
owns its angle-word lifetime. Source controlled Ry is atomic. Run
`tests/unit/test_sparse_matrix_component.py` for all source calls, eight-bit
operators, raw histories and unchanged scalar precision during quantization.

This package owns qROM, QROAM, selected swaps, unary traversal and measured unlookup.

`qroam_bank_unitary_component.qroam_bank_unitary_pair` operates on caller-owned
arbitrary bank words through shared batch XOR and relative-phase routing. Its
inverse reverses routing before XOR; bank storage is never measurement-erased.
This explicit extension differs from the original RIGHT clean-bank isometry
and its matching-image measured erase. Run
`tests/unit/test_qroam_bank_unitary_component.py` for dirty-bank full operators,
multi-axis layouts, source boundaries and every small measurement history.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [qrom_table.py](qrom_table.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_qrom_table.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`clean_lookup_component` binds a scalar power-of-two dataset to an executable
stored component. Forward target and junk ports transfer owned clean banks to
the caller. The measured erase consumes those matching banks and returns its
scratch clean. It reuses `source_qroam_clean` and `qroam_clean_erasure`, whose
primitive counts remain separate from the original model-only adjoint cost.
Run `python -m pytest tests/unit/test_clean_lookup_component.py -q` for source
forward primitives and every raw small lookup/erase measurement history.

`arctangent_declaration.py` owns select/sign/target metadata, binary digit layouts,
self-adjoint identity and fresh symbolic product multiplicities. Concrete
execution remains in `arctan_lookup.py`, using the shared QROM. The original
ArcTan has no gate decomposition. Its tensor-only SDK consumer imports one
matrix leaf through `import_cirq_register_optree` with actual typed placement.
That matrix count, native QROM primitives and symbolic product estimate are
separate accounts. Run `tests/unit/test_trigonometric_foreign_file.py` for complete
complex spectator amplitudes, all small raw histories and clean work.

`qrom_component.qrom_program` stores the source XOR-word dependency graph using
shared bounded unary iteration. Selection words are MSB-first and target arrays
are row-major. Constant rows preserve the original early fold. Predicate work is
private and measurement-erased. Unknown tables remain declarations. Run
`tests/unit/test_qrom_component.py` in the locked reference environment for
recursive source calls, dirty-target complex operators and physical operands.

`majorana_component.majorana_program` stores Pauli Majorana traversal with a
private clean parity accumulator. `selected_pauli_component.selected_pauli_program`
stores bounded row-major Pauli selection under one or two positive controls.
Both reuse the same bounded traversal and exact directional And erasure as
physical emitters. Coordinates are scalar MSB-first bounded words with concrete
extents greater than one. Run `tests/unit/test_majorana_component.py` and
`tests/unit/test_hubbard_select_component.py` for exact recursive source calls,
complex amplitudes, raw histories and private-work checks.

`lookup_binding.py` composes complete integer-table initialization and measured
cleanup using existing recursive lookup and unlookup. Ports are MSB-first,
clean workspace follows source lease order, and classical phase slots follow
caller-declared storage. Run `tests/unit/test_lookup_binding.py` for complete
marked streams, normal append execution and exhaustive small raw branches.
`docs/development/LOOKUP_BINDING_AUDIT.md` records the original six-address-bit
work shortfall, failed-call allocation leak and portable table-size boundary.

`_unary_iteration_session.UnaryIterationPass` composes the accepted physical
cursor with caller-owned operation storage and a shared workspace pool. It owns
reusable context and iterator lifetimes, exact move/cleanup component marks and
classical control scopes. Explicit close returns clean workspace. Start with
`tests/unit/test_unary_iteration_session.py` for full marked source streams,
normal append replacement and all small complex measurement histories. See
`docs/development/UNARY_ITERATION_SESSION_AUDIT.md` for source ownership,
allocation and actual-versus-model resource accounting.
