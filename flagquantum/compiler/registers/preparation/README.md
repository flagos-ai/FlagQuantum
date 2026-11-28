# Register state preparation

This package owns uniform and rank-aware preparation and reflections.

`hubbard_prepare_component.py` owns Hubbard's complete typed preparation and
exact inverse call chains. It receives already converted angles, preserves the
copy-circuit moment barrier and caller-owned temp, and delegates predicate work
to the existing components. Algorithms and original constructor resolution use
this same owner. Run `tests/unit/test_hubbard_prepare_source_constructors.py`
for actual source wire depths, exact angle parameters, full complex histories,
work cleanup and normal append with optional SDKs blocked.

The `thc_uniform_source_component`, `thc_preparation_component` and
`thc_preparation_metadata` owners retain literal THC calls and shared ports.
Source model leaves and invalid cleanup remain explicit physical refusals;
the existing mathematical Algorithms operation has separate execution evidence.
See `docs/development/THC_PREPARATION_SOURCE_AUDIT.md` for the full disposition
and the constructor tests, including the original eight-bit table.

`sparse_chemistry_prepare_component` owns the eighteen sparse source calls and
their explicit directional dependencies. A clean lookup supplies RIGHT banks;
an independently verified bank-unitary dependency instead threads caller-owned
THRU banks. Reflections must retain signal words unless actual private cleanup
is proved. Run `tests/unit/test_sparse_preparation_bank_unitary.py` for arbitrary
bank roundtrips and the controlled/global-phase selection projector checks.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [uniform_preparation.py](uniform_preparation.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_uniform_preparation.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`uniform_preparation_component.uniform_preparation_program` retains complete
uniform source calls and explicit exact adjoints. The shared comparison and
conjunction owners retain their own cleanup. Stored multi-control Hadamards use
source predicate work. Run `tests/unit/test_uniform_preparation_component.py`
for complete source graphs, small complex operators and raw measurement histories.

`block_encoding_component` owns typed SELECT-only and LCU call composition and
their shared signatures. It accepts native oracle Programs, exact inverse pairs
and source control adders; it does not discover external SDK children. RIGHT
preparation junk remains private until unpreparation. Existing Algorithms
signature consumers delegate here. For the shortest replacement scenario, run
`tests/unit/test_legacy_block_encoding_factories.py`: substitute native PREPARE
and SELECT owners, resolve, bind and append without changing the consumer.

`sparse_matrix_component.py` owns packed diffusion and the six-call sparse
matrix root. It receives exact row/column/entry Programs and retains the MSB
flag/location split, source Swap and caller-owned inputs. Child owners retain
private work. Existing Algorithms sparse signature consumers delegate here.
Run `tests/unit/test_sparse_matrix_component.py` for constructor replacement,
all original examples, source-free append, complex operators and small histories.

`hermitian_sparse_matrix_component.py` owns the explicit a/location/b Partition,
six unconditional Always calls and two swaps, with exact child-pair cleanup.
Positive control changes only the swaps. Source return-self adjoints remain
distinct from generic exact call reversal. The existing Algorithms signature
helpers delegate here. See `test_hermitian_sparse_matrix_component.py` for
constructor, recursive call, full complex operator and raw-history evidence.

`linear_combination_component.py` owns packed child-system/ancilla/resource
partitions around exact PREPARE/SELECT/unprepare Programs. Logical junk remains
caller-owned. The Algorithms signature helper delegates here. Only SELECT is
controlled; exact inverses reverse typed calls with supplied child pairs.
`TYPED_LINEAR_COMBINATION_AUDIT.md` distinguishes this accepted root from the
still-open source constructor dependencies. The explicit context and actual
controlled-Phase disposition are documented in
`RESOLVER_LINEAR_COMBINATION_CONTEXT_AUDIT.md`; the default signed fixture does
not execute a registerless scalar phase in its forward chain.


`nu_preparation_component.py` composes six typed source calls and the ordinary
exact logical inverse with directional flags. `potential_preparation_component.py`
composes the two UV calls and their exact outer reversal. Callers supply owned
leaf Programs; missing source bodies remain explicit opaque declarations.
Neither builder evaluates a statevector or selects a backend. Algorithms owns
signatures/models and Ecosystem owns decoded source constructors. Use
`tests/unit/test_momentum_source_declarations.py` as the logical change path and
`tests/unit/test_uniform_preparation_component.py` for executable child cleanup.


Factorization call inverses use explicit imported identities because opaque
forward/inverse leaves may have identical signatures. DF one-body/block and SF
block composition reuse `adjoint_component_program`; specialized source SF
one-body direction keeps its unchanged body. Numerical children stay with their
owners. `tests/unit/test_factorization_source_closure.py` covers complete source
calls and final returns; existing library tests cover selected physical children.

`prepared_reflection_component.literal_prepared_reflection_program` preserves
source RIGHT-junk consume/produce/free lifetimes. It shares zero marking and
phase construction with the existing through profile, without asserting a
unitary bank extension or arbitrary-signal cleanup. Missing preparation leaves
still refuse physical binding. See the THC walk audit and
`tests/unit/test_thc_walk_source_context.py` for exact original connections.

`qubitization_walk_component.doubly_controlled_select_walk_pair` composes the
literal nested-positive SELECT walk using existing packed multi-control owners.
Its inputs are explicit main-entry positive SELECT/inverse and source-declared
self-adjoint reflection Programs, plus their bare data signatures and control
port names. Each child owns its predicate work and erasure; the walk preserves
the caller's outer/inner control order. Run `test_hubbard_double_control_source.py`
for actual Hubbard source graphs and a clean small example's full complex,
all-history and work-zero certificate. Source dirty-Free Hubbard behavior remains
separate in the [Hamiltonian audit](../../../../docs/development/HAMILTONIAN_GQSP_CONSTRUCTOR_AUDIT.md).
