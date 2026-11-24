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
