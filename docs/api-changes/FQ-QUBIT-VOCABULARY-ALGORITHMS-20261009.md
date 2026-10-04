# WQ-5: the algorithms slice of the qubit vocabulary migration

One document per approved change to the Stable Core public API. This one records
the fifth slice of
[Quit calling them wires](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md),
[The other two doors](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md),
[the runtime slice](FQ-QUBIT-VOCABULARY-RUNTIME-20261007.md), and
[the runtime executor slice](FQ-QUBIT-VOCABULARY-EXECUTORS-20261008.md) as it
lands.

It proposes no new capability, no export, no default, and no schema. `IR_VERSION`
stays `"1.0"`. It renames names on surfaces the four predecessor documents already
decided and invents no pair of its own. Slice membership is not decided here
either: it is declared by `[[slices]] files` in
`contracts/qubit-vocabulary-contract.toml`, and the gate recomputes each slice's
counts from the frozen baseline on every run.

Like WQ-4, this slice has **no user-facing alias at all**. Its whole surface lives
under `flagquantum.algorithms`, and `hasattr(fq, "algorithms")` is `False`: the
package's `__init__` reaches the algorithms package through no attribute and adds
no algorithm name to `fq.__all__`. A forwarder would therefore have no legal
caller outside the package and would be permanent weight against engineering
decision principle 1 (no permanent compatibility debt). Every rename below is a
hard rename in the same commit.

## Decision and authorization

The authorization is the one the user gave for the program: *"the user-facing
surface of FlagQuantum says `qubit`, never `wire`"*, restated in the
[integral document](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md). WQ-5 invents no new
scope: the contract records this document as `algorithms_authorization` under
`[verification]`, alongside the entries the four predecessor slices left, and the
gate checks that the path exists on every run.

WQ-5 owns **11 files** and is measured at **52 canonical parameters**, **15
ledgered attribute names**, **0 definition names**, and **0 payload-key
exclusions**. The 15 attributes split by declaration kind into 8 `field`, 7
`member`, and 0 `instance`.

The slice is the first with **nothing to keep**. WQ-2 through WQ-4 each had at
least one site whose spelling was a frozen payload key or a serialized field, and
each wrote it down in `[attribute_exclusions]`. WQ-5 has none: the values these
algorithms return are `AmplitudeEstimationResult`, `PcaResult`,
`SingularValueResult`, and `PhaseEstimationSpec`, none of which is written into a
versioned payload or read back by a reader that has to understand an older
package.

## What the slice renames

### 52 parameters

The parameters fall into four groups.

| Group | Before | After |
|---|---|---|
| Register widths (`core.py`, `grover.py`, `amplitude_estimation.py`, `pca.py`, `svd.py`, `qarm.py`) | `n_wires`, `n_counting_wires`, `n_evaluation_wires`, `n_support_wires`, `n_item_wires` | `n_qubits`, `n_counting_qubits`, `n_evaluation_qubits`, `n_support_qubits`, `n_item_qubits` |
| Operator protocols (`primitives/types.py`) | `AmplitudeOperator.{apply_a,apply_a_dagger,apply_mark,apply_plain,apply_zero_reflection}(wires=…)`, `ControlledUnitary.{apply,apply_controlled,apply_power_controlled}(wires=…)`, `StatePreparationOperator.{mark,prepare}(wires=…)` | `qubits=…` |
| Primitive builders (`primitives/oracle.py`, `primitives/phase_estimation.py`, `primitives/qft.py`, `primitives/state_preparation.py`) | `append_bit_oracle`/`append_phase_oracle`/`append_qft`/`append_arbitrary_state`/`arbitrary_state` `(wires=…)`, `bit_oracle`/`phase_oracle`/`marked_states`/`qft`/`uniform_state` `(n_wires=…)`, `append_phase_estimation(counting_wires=…, evaluation_wires=…)` | the `qubit` spellings |
| Hamiltonians and search (`core.py`, `qarm.py`) | `HamiltonianTerm.__init__(wires=…)`, `pauli_term(wires=…)`, `frequent_itemset_operator(n_support_wires=…)`, `run_frequent_itemset(n_counting_wires=…, n_support_wires=…)` | the `qubit` spellings |

`n_a_wires` and `n_b_wires`, the two registers `pca.py` multiplies, are the same
rename without a ledger row of their own: they are private helpers' parameters.

### 15 attribute names

| Group | Before | After |
|---|---|---|
| Result records (`amplitude_estimation.py`, `pca.py`, `svd.py`, `primitives/phase_estimation.py`) | `AmplitudeEstimationResult.{n_counting_wires,n_evaluation_wires}`, `PcaResult.n_counting_wires`, `SingularValueResult.n_counting_wires`, `PhaseEstimationSpec.{n_counting_wires,n_evaluation_wires}` | the `qubit` spellings |
| Operator protocols (`primitives/types.py`) | `AmplitudeOperator.n_wires`, `ControlledUnitary.n_wires`, `StatePreparationOperator.n_wires` | `n_qubits` |
| Hamiltonians (`core.py`) | `Hamiltonian.n_wires`, `HamiltonianTerm.wires`, `HamiltonianTerm.max_wire` | `n_qubits`, `qubits`, `max_qubit` |
| Frequent itemsets (`qarm.py`) | `FrequentItemsetOperator.{n_wires,n_item_wires,n_support_wires}` | `n_qubits`, `n_item_qubits`, `n_support_qubits` |

### 41 private parameters, unledgered

The ledger counts the public function surface, so the 41 `wire`-named parameters
of this slice's private helpers are not rows in it. They move anyway, because a
slice that renames `n_counting_wires` in its public signature and leaves
`_counting_probability(n_counting_wires=…)` alone has renamed a word rather than a
concept. This is why `boundary.measured_private` drops from 372 to 331: the gate
re-measures the private bucket and refuses a stale number.

## The catch-up a rename owes

`flagquantum/algorithms` is sparsely sliced — the contract names 11 of its 18
modules — and three classes of reader had to follow the rename without being part
of the slice.

1. **`flagquantum/algorithms/qubo.py` reads `HamiltonianTerm.wires` and
   `Hamiltonian.n_wires`.** It is a module no slice owns, because everything it
   declares is a local rather than a parameter or an attribute. The five reads
   moved; its own `declared_wires` local, its `_IDENTITY_WIRE` constant, and its
   prose did not, because they are not this slice's names and no gate counts them.
2. **Two files outside the package read the same two declarations.**
   `deployment/cloud.py` (two sites) and `simulation/mps/tebd.py` (one) each read
   `hamiltonian.n_wires` next to a local `n_wires` of their own and next to
   `source_ir.n_wires`, which is a frozen IR payload key. A per-file replacement
   of the spelling would have renamed two names this slice does not own — the
   same over-reach that cost WQ-2 three repairs and WQ-4 six.
3. **The test doubles that stand in for the operator protocols.** A stub that
   defines `n_wires` is asserting conformance to `AmplitudeOperator` or
   `ControlledUnitary`; when the protocol's attribute moves, the stub has to move
   with it or the test measures a protocol that no longer exists. Six test files
   carried such stubs.

Two further spelling traps sat inside the slice's own files and were handled
deliberately rather than mechanically:

* `oracle.py` builds a refusal sentence with `free_word = "wire" if needed == 1
  else "wires"`. A singular/plural helper is not a name, so a whole-token pass
  leaves it alone; it was renamed with the sentence it feeds.
* `wired` appears twice as the English verb (`"These must be wired in the order
  the ladder computes them"`). It is not a spelling of `wire` and is left alone
  everywhere.

## Verification

Each command was run at the commit that carries the rename, in
`/tmp/n19/work/fq-wq5` on branch `feat/qubit-vocabulary-algorithms`.

| # | Command | Result |
|---|---|---|
| 1 | `python -m black --check flagquantum tests tools` | all files unchanged |
| 2 | `python -m ruff check flagquantum tests tools` | `All checks passed!` |
| 3 | `python -m mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum` | `Success: no issues found in 624 source files` |
| 4 | `python tools/check_architecture.py` | `architecture boundaries passed` |
| 5 | `python tools/check_qubit_vocabulary.py` | exit 0; `131 of 341` parameters retired, `83 of 122` attributes retired, `4 of 10` definitions retired; WQ-5 is `52 / 15 / 0` with 0 remaining |
| 6 | `python tools/public_api_snapshot.py` | `public API migration baseline passed`, and neither `docs/public_api_v1.json` nor `contracts/public-api-v0.2-baseline.json` is edited |
| 7 | `python tools/validate_required_checks.py` | still `validated 6 externally configured required checks`: this change adds tests and contract rows, never a CI job |
| 8 | `python tools/check_team_scope.py --require-classified --base d4076b99f` | every changed path is classified: 12 `algorithms`, 1 `remote`, 1 `simulation`, 1 protected, 9 shared |
| 9 | the CI marker lane | no new failure; see the PR body for the exact run |

`tools/check_team_scope.py` reads `flagquantum/deployment/cloud.py` as `remote`
and `flagquantum/simulation/mps/tebd.py` as `simulation`, so the contract declares
WQ-5's owner as `algorithms + remote + simulation` rather than `algorithms`
alone. The alternative — leaving the two catch-up sites to a later slice — would
have left the package failing `mypy --strict` on three files at this commit.

Status: landed. The rename is WQ-5's and no other slice may retire a site this
ledger assigns to WQ-5.
