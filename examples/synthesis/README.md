# High-level quantum programming examples

Use `synth` to describe computation, quantum data and constraints. Start with
[tutorials](tutorials/README.md), then explore [algorithms](algorithms/README.md)
and [applications](applications/README.md).

The [classified original-case catalog](CATALOG.md) contains all **216** pinned
Qmod paths, each with source, bindings, resource records and native/Circuit
availability. The tutorial, algorithm and application indexes below also retain
the existing 48 teaching examples.

`catalog.json` is a compact machine index. Run
`python tools/synthesis_catalog.py --write` to normalize it and regenerate the
readable Markdown navigation. Compact encoding keeps source retrieval within
GitHub GraphQL text limits without changing case data, bindings or source digests.

Run examples as Python modules from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.tutorials.arithmetic.arithmetic_implementations.main
python -m examples.synthesis.algorithms.grover.grover.main
python -m examples.synthesis.algorithms.grover.grover.comparisons.circuit
```

The arithmetic example prepares two uniform two-bit inputs, composes `energy`,
and returns all 16 `(x, y, x*x+y)` outcomes with probability 1/16. The Grover
example prints named results and actual costs. Its comparison verifies Qmod,
native scalar qfunc and Circuit amplitudes, including workspace and absolute phase.

## Choosing an entry

Each teaching case has a `README.md` and a runnable `main.py`. The fourteen
teaching cases with a `comparisons/` directory use native qfunc in `main.py`;
`comparisons/source.qmod`
and `comparisons/circuit.py` provide the corresponding source and independent
Circuit validation. Comparison runners import the native definitions from
`main.py` so both routes exercise the same native program.

Tutorials retain their existing scope. Entries labelled **Qmod import tutorial**
teach the source importer. Entries labelled **Compiler contributor diagnostic**
use internal IR or planning APIs. The indexes identify these before users open
an example. Native Python and Qmod support boundaries remain explicit in each case.

The separate [Circuit and framework examples](../README.md) cover gate-level
construction, training, simulation and deployment.

## Corpus acceptance

The [216-case catalog](CATALOG.md) classifies the complete pinned snapshot under
these same tutorial, algorithm and application directories. Each catalog case
contains a byte-identical Qmod original and a README with a runnable command.
Six cases additionally contain a separately identified approved numerical revision.

The shared `run_qmod` runner reuses corpus auditing and its recorded classical
bindings and exact execution profiles. Original input is the default; six case
READMEs explicitly select `--revision approved` for successful execution.

```sh
python -m examples.synthesis.run_qmod --verify
python -m examples.synthesis.run_qmod --all --revision approved --output /tmp/qmod-examples.json
python -m tools.synthesis_catalog
```

Native implementations are linked only when the existing three-form comparison
uses the exact original source bytes. Other cases mark the native implementation
as not yet provided. The catalog also links existing tests and validators,
including their Circuit references where found. These links preserve each
validation's own parameters and scope.

Source bytes, licenses and manifests remain independently verifiable through the
[original fixtures](../../tests/fixtures/classiq_qmod) and
[approved-overlay instructions](../../tests/fixtures/classiq_qmod/accepted_revision/README.md).
The example catalog provides the complete runnable source set without requiring
an external corpus checkout. Regenerate catalog documentation after changing
catalog metadata with `python -m tools.synthesis_catalog --write`.

The first native completion batch adds 17 exact-original counterparts: nine
standard gates, Bell/GHZ and real amplitude/probability preparation, unsigned
multiplication and three quadratic-computation tutorials. Together with the six
previous entries, that batch covered 23/216 runnable native/Circuit comparisons.
The native array batch adds Hadamard transformation, Bell-state classical
shadows, entanglement and classical-shadow tomography, bringing the total to
27/216. Array indexing/slices, static loops and writable gate helpers reuse the
existing checked register effects. See the
[native views contract](../../docs/api-changes/FQ-SYNTH-NATIVE-VIEWS-20261107.md).
Run `python -m pytest tests/integration/test_synthesis_native_catalog.py -q`
to execute those comparisons and the preparation, phase and refusal checks.
The shared comparison helper uses `Program.registers` in its declared bit order
and checks every complex amplitude, including zero hidden workspace, without
aligning away global phase. The 216-case execution smoke suite remains a
separate scope. Nested quantum arrays, unrestricted mutable helpers and runtime
classical binding remain outside the native profile.

## Module migration

The former flat modules now have topic-specific paths. Update saved commands and
imports using this table. Historical measurements retain their recorded paths
and source hashes at their recorded revisions.

| Former module | Current module |
| --- | --- |
| `examples.algorithms.synthesis_source_trial` | `examples.synthesis.tutorials.getting_started.source_trial.main` |
| `examples.algorithms.synthesis_trial` | `examples.synthesis.tutorials.getting_started.trial.main` |
| `examples.synthesis_amplification_iterations` | `examples.synthesis.algorithms.amplitude_amplification.amplification_iterations.main` |
| `examples.synthesis_arithmetic_baseline` | `examples.synthesis.tutorials.arithmetic.arithmetic_baseline.main` |
| `examples.synthesis_arithmetic_implementations` | `examples.synthesis.tutorials.arithmetic.arithmetic_implementations.main` |
| `examples.synthesis_autoencoder` | `examples.synthesis.applications.machine_learning.autoencoder.main` |
| `examples.synthesis_boolean_factors` | `examples.synthesis.tutorials.arithmetic.boolean_factors.main` |
| `examples.synthesis_boolean_predicate` | `examples.synthesis.tutorials.arithmetic.boolean_predicate.main` |
| `examples.synthesis_carry_free_add` | `examples.synthesis.tutorials.arithmetic.carry_free_add.main` |
| `examples.synthesis_conditional` | `examples.synthesis.tutorials.arithmetic.conditional.main` |
| `examples.synthesis_const_calls` | `examples.synthesis.tutorials.composition.const_calls.main` |
| `examples.synthesis_controlled_evolution` | `examples.synthesis.algorithms.hamiltonian_simulation.controlled_evolution.main` |
| `examples.synthesis_controlled_predicate` | `examples.synthesis.tutorials.control.controlled_predicate.main` |
| `examples.synthesis_exact_amplification` | `examples.synthesis.algorithms.amplitude_amplification.exact_amplification.main` |
| `examples.synthesis_execution_contract` | `examples.synthesis.tutorials.workspace.execution_contract.main` |
| `examples.synthesis_grover` | `examples.synthesis.algorithms.grover.grover.main` |
| `examples.synthesis_integer_composition` | `examples.synthesis.tutorials.composition.integer_composition.main` |
| `examples.synthesis_interval` | `examples.synthesis.tutorials.arithmetic.interval.main` |
| `examples.synthesis_large_pauli` | `examples.synthesis.algorithms.hamiltonian_simulation.large_pauli.main` |
| `examples.synthesis_local_registers` | `examples.synthesis.tutorials.composition.local_registers.main` |
| `examples.synthesis_native_and_qmod_rz` | `examples.synthesis.tutorials.qmod.native_and_qmod_rz.main` |
| `examples.synthesis_native_predicate_phase` | `examples.synthesis.tutorials.control.native_predicate_phase.main` |
| `examples.synthesis_normalization` | `examples.synthesis.tutorials.arithmetic.normalization.main` |
| `examples.synthesis_phase_oracle` | `examples.synthesis.tutorials.control.phase_oracle.main` |
| `examples.synthesis_power_two_multiply` | `examples.synthesis.tutorials.arithmetic.power_two_multiply.main` |
| `examples.synthesis_predicate` | `examples.synthesis.tutorials.arithmetic.predicate.main` |
| `examples.synthesis_predicate_oracle` | `examples.synthesis.algorithms.grover.predicate_oracle.main` |
| `examples.synthesis_python_powers` | `examples.synthesis.tutorials.arithmetic.python_powers.main` |
| `examples.synthesis_python_program` | `examples.synthesis.tutorials.composition.python_program.main` |
| `examples.synthesis_qmod_bit_predicates` | `examples.synthesis.tutorials.qmod.qmod_bit_predicates.main` |
| `examples.synthesis_qmod_comparisons` | `examples.synthesis.tutorials.qmod.qmod_comparisons.main` |
| `examples.synthesis_qmod_const` | `examples.synthesis.tutorials.qmod.qmod_const.main` |
| `examples.synthesis_qmod_helpers` | `examples.synthesis.tutorials.qmod.qmod_helpers.main` |
| `examples.synthesis_qmod_nested` | `examples.synthesis.tutorials.qmod.qmod_nested.main` |
| `examples.synthesis_qmod_pair_gates` | `examples.synthesis.tutorials.qmod.qmod_pair_gates.main` |
| `examples.synthesis_qmod_phase` | `examples.synthesis.tutorials.qmod.qmod_phase.main` |
| `examples.synthesis_qmod_phase_gate` | `examples.synthesis.tutorials.qmod.qmod_phase_gate.main` |
| `examples.synthesis_qmod_powers` | `examples.synthesis.tutorials.qmod.qmod_powers.main` |
| `examples.synthesis_qmod_within` | `examples.synthesis.tutorials.qmod.qmod_within.main` |
| `examples.synthesis_quantum_callbacks` | `examples.synthesis.algorithms.hidden_shift.quantum_callbacks.main` |
| `examples.synthesis_register_cleanup` | `examples.synthesis.tutorials.composition.register_cleanup.main` |
| `examples.synthesis_register_phase` | `examples.synthesis.tutorials.control.register_phase.main` |
| `examples.synthesis_selection` | `examples.synthesis.tutorials.workspace.selection.main` |
| `examples.synthesis_selection_costs` | `examples.synthesis.tutorials.workspace.selection_costs.main` |
| `examples.synthesis_static_loop` | `examples.synthesis.tutorials.composition.static_loop.main` |
| `examples.synthesis_static_qpe` | `examples.synthesis.algorithms.phase_estimation.static_qpe.main` |
| `examples.synthesis_swap_test` | `examples.synthesis.algorithms.state_preparation.swap_test.main` |
| `examples.synthesis_workspace` | `examples.synthesis.tutorials.workspace.workspace.main` |
