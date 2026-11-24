# Register state preparation

This package owns uniform and rank-aware preparation and reflections.

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
still-open source constructor payload and signed Phase dependencies.
