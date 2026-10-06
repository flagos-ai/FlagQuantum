# The primitives package admits an export on three dimensions, and one export was deleted

**Date:** 2026-10-19
**Slice:** `N1-12`
**Surface:** the admission rule of `flagquantum/algorithms/primitives`, its export list,
and the differentiability of the two amplitude entry points
**Contract:** `contracts/primitives-admission-contract.toml` (new)
**Gate:** `tools/check_primitives_admission_contract.py` (new)
**Conformance:** `tests/unit/test_primitives_admission_contract.py`
**Guide:** [`docs/guides/CIRCUIT_COMPOSITION.md`](../guides/CIRCUIT_COMPOSITION.md), and
the admission paragraph of [`docs/guides/ALGORITHMS.md`](../guides/ALGORITHMS.md)
**Predecessor:** [The algorithms slice of the qubit vocabulary
migration](FQ-QUBIT-VOCABULARY-ALGORITHMS-20261009.md)

## What this slice decides

`flagquantum/algorithms/primitives` states which exports it holds and why. Before this
slice it stated **two grounds** — a scalar distribution semantic and a "the package is
not a general-purpose toolkit" disclaimer — which are not a rule a contributor can apply
to a candidate export, because neither one says what makes an export admissible.

The package now states **three dimensions**, and every one of its 18 exports is recorded
against all three in `contracts/primitives-admission-contract.toml`:

| Dimension | The question it answers | Where the answer is measured |
| --- | --- | --- |
| Consumers | Who calls this, and from where? | derived from the import graph over `flagquantum/`, `tests/`, `examples/`, `benchmarks/` |
| Distribution semantics | What does this claim about where the work happens? | one value of the Core `DistributionSemantics` vocabulary, `single_device_fast_path` unless stated |
| Differentiability | Does a gradient reach through this? | one of three declared shapes, with a reproducing test named where a tensor is accepted |

The measurement also **deleted one export**. `StatePreparationOperator`, a `Protocol`
declared at `flagquantum/algorithms/primitives/types.py:94` on `6a6c080be`, was admitted
on the third dimension's weakest ground — the *expectation* that a third-party state
preparation operator would implement it — and the import graph found no consumer at all.

## Why the consumer dimension is the one that decides

The rule is stated as one sentence in the package, and its shape is the whole of the
decision:

> **The consumer fixes the ground.** An export is admitted on the consumers it has, not
> on the shape of its docstring. A second concrete implementation is evidence; a
> plausible future one is not.

The gate derives the basis rather than reading it, so the contract records a measurement
and not an aspiration. `derive_basis` counts, per export, the modules under
`flagquantum/` that reference it (`framework_consumers`) and the modules outside the
package but inside the tree that do (`package_consumers`), and maps the counts to one of
four bases:

| Basis | Derived when | Exports |
| --- | --- | --- |
| `shared` | two or more framework consumers | 4 |
| `confirmed` | exactly one framework consumer | 3 |
| `grounded_expectation` | no framework consumer, but a consumer inside the package | 2 |
| `public_unit` | no consumer in the framework or the package, but one in `tests/`, `examples/` or `benchmarks/` | 9 |
| `unadmitted` | no consumer anywhere | — refused |

`unadmitted` is not a fourth basis. **No consumer anywhere is a defect, not a base**: the
gate refuses a row that carries it, and `derive_basis` returns it only so the refusal can
name the export. `StatePreparationOperator` is the export that measurement removed, and
the reason is worth stating exactly, because "no consumer" is a claim about a whole tree
rather than about a package:

```console
$ git grep -n StatePreparationOperator 6a6c080be -- flagquantum tests examples benchmarks
6a6c080be:flagquantum/algorithms/primitives/__init__.py:32:from .types import StatePreparationOperator as StatePreparationOperator
6a6c080be:flagquantum/algorithms/primitives/__init__.py:39:    "StatePreparationOperator",
6a6c080be:flagquantum/algorithms/primitives/types.py:94:class StatePreparationOperator(Protocol):
```

Three hits, and all three are the declaration and its re-export. No algorithm module, no
test, no example, and no benchmark named it, so the expectation was the whole of its
support. It is deleted together with its `__all__` entry and its re-export; the package
exports 18 names, down from 19.

The deletion is **not** a Stable Core change. `docs/public_api_v1.json` records 36 stable
exports and none of them belongs to this package, and the string `primitives` does not
occur in `contracts/public-api-v0.2-baseline.json`,
`contracts/public-api-v1-candidate.json`, or `contracts/legacy-root-api-test-debt.json`.
Removing an export from a package that the stable surface does not name is a change to
that package's own contract, which is what this record authorizes.

## What the distribution-semantics dimension replaces

The old paragraph said the package avoids "unwanted complexity". That is a claim about
effort, and it cannot be checked. The dimension that replaced it is the one Core already
owns: `DistributionSemantics`, the seven-literal vocabulary in
`flagquantum/core/contracts.py`. A primitive may not imply a distribution semantic; it
must claim one, from that vocabulary, and the gate requires the value it claims to be a
member of it.

All 18 exports claim `single_device_fast_path`, which is stated as the contract's
default rather than repeated 18 times in prose. That is a measurement and not a
disclaimer: a primitive that became sharded would have to change its row, and the gate
would fail until the row and the code agreed.

## The differentiability dimension, and the one behaviour it changed

An export in this package is public and accepts a tensor at exactly two places, and the
gate requires each of them to declare one of three shapes:

| Shape | Meaning |
| --- | --- |
| `no_tensor_argument` | the export takes no tensor, so the question does not arise |
| `classical_tensor_input` | a tensor is accepted and read as classical data: no gradient flows back |
| `differentiable_tensor_input` | a gradient does flow back, and the row names a test that proves it |

`differentiable_tensor_input` is declared in the contract and unused, because nothing in
the package is differentiable end to end today. Declaring it rather than leaving it out
is deliberate: the third shape is the reason the other two exist, and a contributor
adding a differentiable primitive needs the value to already be in the vocabulary.

`classical_tensor_input` is the shape of both `arbitrary_state` and
`append_arbitrary_state`. Reading the amplitudes as classical data is the existing
behaviour, and it is stated rather than changed:

```console
$ python -c "import torch; from flagquantum.algorithms.primitives import arbitrary_state; \
             a = torch.tensor([0.6, 0.8], dtype=torch.complex128, requires_grad=True); \
             c = arbitrary_state(a); print(a.grad_fn, [i.name for i in c._instructions])"
None ['ry', 'rz']
```

The tensor's `requires_grad` is accepted and ignored, no autograd edge reaches the
circuit, and each angle is emitted as a plain float. That is what the amplitudes do: they
fix the rotation angles outright rather than parameterising them.

**One line changed to make that honest.** `_prepared_amplitudes` computed
`float(data.norm())` on a tensor that may require grad, which makes PyTorch emit
`UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to
unexpected behavior`. The values were correct and the warning was noise, but a warning
that fires on a documented input is a defect in its own right. The vector is now
detached once at entry.

**The change is provably behaviour-neutral**, and that was measured rather than argued.
A probe dumps, for `n` in 1, 2, 3 and for `requires_grad` false and true, every
instruction's name, qubits, parameters, matrix and metadata, for both `arbitrary_state`
and `append_arbitrary_state`:

```console
$ cd /tmp/n19 && rm -f ev_old.json ev_new.json
$ (cd /tmp/n19/fq-main && PYTHONPATH=. python /tmp/n19/probe_sp.py > /tmp/n19/ev_old.json 2>/tmp/n19/ev_old.err)
$ (cd <worktree>       && PYTHONPATH=. python /tmp/n19/probe_sp.py > /tmp/n19/ev_new.json 2>/tmp/n19/ev_new.err)
$ cmp /tmp/n19/ev_old.json /tmp/n19/ev_new.json && echo BITWISE
BITWISE
$ wc -c /tmp/n19/ev_old.json /tmp/n19/ev_new.json
    3982 /tmp/n19/ev_old.json
    3982 /tmp/n19/ev_new.json
    7964 total
$ cat /tmp/n19/ev_old.err
/private/tmp/n19/fq-main/flagquantum/algorithms/primitives/state_preparation.py:152: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)
  norm = float(data.norm())
$ cat /tmp/n19/ev_new.err
$
```

The two dumps are byte-identical, and the only difference between the runs is on stderr:
the old one emits the warning above, the new one emits nothing.

## Why the rule is stated three times and checked once

The rule is written where a contributor meets it — the package docstring, the package
`README.md`, and the admission paragraph of the algorithms guide — and the three
statements are allowed to differ in length, because an entry point and a reference
paragraph are different documents. What is not allowed is for one of them to be silent
about a dimension, because then a contributor reading *that* one is reading a different
rule.

`tools/check_primitives_admission_contract.py` therefore reads the three files the
contract names in its top-level `stated_in` list and requires each of them to write each
dimension's marker:

```toml
stated_in = [
    "flagquantum/algorithms/primitives/__init__.py",
    "flagquantum/algorithms/primitives/README.md",
    "docs/guides/ALGORITHMS.md",
]

[[dimension]]
name = "distribution_semantics"
marker = "distribution semantics"
```

The comparison is folded — case and whitespace — rather than exact, so a statement that
begins a sentence with the marker satisfies it and prose does not have to be contorted to
match a checker. A file that names a dimension it does not state fails by name:

```text
docs/guides/ALGORITHMS.md does not state the differentiability dimension: it never
writes 'differentiability', so the rule a contributor reads there is a different rule
```

## The gate, and what it refuses

`tools/check_primitives_admission_contract.py` reads the contract against the tree and
refuses six kinds of drift. Each is negative-tested in
`tests/unit/test_primitives_admission_contract.py` against a payload the test builds
itself, so the refusal is exercised rather than described:

* the recorded export list differs from the package's `__all__`, so an export cannot be
  added or removed without the contract moving with it;
* a row's recorded basis differs from the basis re-derived from the import graph, so a
  claim cannot survive its consumer moving;
* a row's recorded distribution semantic is not a member of the Core vocabulary;
* a row's recorded consumers are not the ones found, in either direction;
* an export accepted as `differentiable_tensor_input` does not name an existing test, or
  a tensor-accepting export does not say in prose why no gradient flows;
* an export recorded as retired is still on the package's export list.

The gate does not restate the export list; it measures it. Adding a primitive to the
package without adding its row fails with

```text
primitives admission contract surface drifted from the package
```

and the way to clear that failure is to regenerate the measured half, not to edit a
number — the generator lives in the same tool and refuses a row whose declared shape
disagrees with its own note.

## The composition guide

This slice also adds [`docs/guides/CIRCUIT_COMPOSITION.md`](../guides/CIRCUIT_COMPOSITION.md),
which is the user-facing half of the construction-time composition contract that
[`FQ-CIRCUIT-COMPOSITION-20261002.md`](FQ-CIRCUIT-COMPOSITION-20261002.md) froze. It is
written to the rule the plan sets for this slice — it describes composition through the
names this line's work establishes, and it names no port, no adapter, and no
`compose(unitaries=...)`-shaped interface.

Its thirteen code blocks are **executed**, and the output they quote in comment lines is
compared token by token, by `tests/unit/test_circuit_composition_guide.py` — the same
convention `tests/test_documentation_entry_examples.py` applies to `ALGORITHMS.md`,
reused rather than reimplemented so the two guides cannot drift apart. Two of the guide's
quoted values were wrong when first written and were corrected by running them: the
refusal text for an incomplete `qubit_map` is
`ValidationError: Circuit.compose qubit_map must name every local qubit; missing (1,)`,
and the refusal for an unnamed channel ends
`its matrix does not expose a conjugate transpose.`, with the final period that the
comparison requires.

The guide carries the two losses that make the point it is making, and both were measured:

* **A custom operation cannot be re-issued by name.** `fq.Circuit(1).any(0, unitary=gate)`
  composes onto another qubit with its matrix intact, while a loop calling
  `getattr(target, item.name)(*item.wires)` raises
  `TypeError: Circuit.any() missing 1 required keyword-only argument: 'unitary'`.
* **A channel's `is_channel` metadata is what the planner reads.** Dropping it through the
  public `to_ir()`/`from_ir` route rebuilds a circuit whose channel is unnamed, and the
  loss is visible twice: `fq.plan` reports `statevector` and 16 bytes where the composed
  circuit reports `density_matrix` and 32, and `adjoint` refuses it as
  `its matrix does not expose a conjugate transpose` where the composed circuit is refused
  as `it is a noise channel, which has no unitary inverse`. `is_channel` is read at 40
  sites in 24 modules.

## Evidence

```console
$ python tools/check_primitives_admission_contract.py
Primitives admission contract passed: 18 exports, 9 admitted as public units

$ python -m pytest tests/unit/test_primitives_admission_contract.py -q
15 passed

$ python -m pytest tests/unit/test_circuit_composition_guide.py \
      tests/test_documentation_entry_examples.py -q
5 passed, 1 skipped

$ python tools/check_qubit_vocabulary.py
Qubit vocabulary contract passed: 341 of 341 baseline sites retired, 11 kept as deprecated aliases
…
Qubit vocabulary contract passed: 117 of 122 attribute sites retired, 5 kept as deprecated aliases
…
Qubit vocabulary call sites: 0 keywords passed that their callee rejects (invariant: 0)

$ python tools/check_circuit_composition_contract.py
Circuit composition contract passed

$ python tools/check_docs_links.py
Checked 547 markdown files; all local links and anchors resolve.
```

## What this slice does not reach

* **The two refusals the guide quotes that are not in the contract's `[[refusals]]`
  table.** The dynamic-operation and classically-conditioned refusals are properties of
  `Circuit.adjoint` inside the composition family, and they belong to
  `FQ-CIRCUIT-COMPOSITION-20261002.md`'s `[adjoint]` section rather than to this package's
  admission rule. The guide quotes them because a reader meets them at `adjoint()`; this
  slice does not extend the composition contract to cover them.
* **`arbitrary_state`'s behaviour.** Detaching the amplitudes on entry changed no
  instruction, no parameter, no matrix and no metadata, which is proved above. Making the
  amplitudes genuinely differentiable — emitting them as `Parameter` objects that
  `bind_parameters` can move — is a different change with a different contract, and it is
  not claimed here.
* **The other 17 exports' differentiability.** Only two exports accept a tensor, so only
  two have a shape to declare. The other sixteen are `no_tensor_argument` by measurement,
  and the gate re-derives that rather than trusting the row.
* **The algorithms slice's own record.** `FQ-QUBIT-VOCABULARY-ALGORITHMS-20261009.md`
  names `StatePreparationOperator` in two tables (lines 57 and 69) as a protocol whose
  `wires`-named keywords and `n_wires` attribute were migrated. Those lines are a
  historical statement about a protocol that existed when they were written, in the same
  class as the code quotation the Twin-region slice deliberately left in place, and they
  are not rewritten by this slice. The qubit-vocabulary ledgers keep the three
  `StatePreparationOperator` sites as recorded rows — `mark` and `prepare` in
  `[retirement]`, `n_wires` in `[attribute_retirement]` — because a frozen ledger records
  what the surface was, and a site is retired rather than deleted from it.
* **Whether the rule should be stated once and generated.** The three statements are
  checked for agreement on the three markers and not for identical wording. Generating
  all three from one source would make the package docstring, the package README and the
  guide one document, which is a larger change than this slice, and the gate's
  marker check is what makes the smaller one safe.
