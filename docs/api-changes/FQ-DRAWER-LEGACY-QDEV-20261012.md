# The drawer reads a legacy device's spelling without publishing it

Status: approved for implementation.
Area: `flagquantum.drawer`.
Base: the `feat/qubit-vocabulary-attributes` tip at `721ce1a4`, the runtime
executor slice, whose decision record is
[`FQ-QUBIT-VOCABULARY-EXECUTORS-20261008.md`](FQ-QUBIT-VOCABULARY-EXECUTORS-20261008.md),
merged with `main` at `bdb8675f`.
Predecessor: [`FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md`](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md)
and the slice record [`FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md`](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md).

This branch is not the tip of the whole migration: it stops after the runtime
executor slice, so the counts quoted below are the ones this tree reports, not
the ones a tree that also carries WQ-5 through WQ-8 reports. The regression, the
fix, and the repair of the stale test spellings are identical on either tree,
because `to_drawable_circuit` and the Matplotlib renderer stop changing after
WQ-2. It is recorded here rather than only on the later branch because the
`coverage` job — the one lane in `ci.yml` that installs the `viz` extra — is red
on this branch for exactly this reason.

## The regression

The qubit-vocabulary migration renamed the two things a device reports — its
width and the qubits an operation occupies — and left `to_drawable_circuit`
recognising a program as device-shaped only when it carried **both**
`op_history` and `n_qubits`. A third-party object that predates the migration
carries `op_history` and `n_wires`. It therefore fell past that guard, was
returned unchanged, kept its `wires`-keyed operation entries, and the text
renderer read `op.get("qubits", [])` from each of them:

```
$ pytest tests/test_drawer_ir.py
tests/test_drawer_ir.py::test_drawer_keeps_legacy_qdev_compatibility FAILED
E   ValueError: min() iterable argument is empty
flagquantum/drawer/text_drawer.py:196: in _create_layers
    min_qubit = min(qubits)
```

`draw(LegacyDevice())` raised instead of drawing `H`. The same call succeeded
before the migration and succeeds on a tree with the fix.

## The decision

**Acceptance is not publication.** A name this package chose moves; a name a
caller's own object chose is not ours to change. The drawer therefore reads a
legacy spelling at exactly one boundary and immediately re-expresses it in the
vocabulary it publishes:

* `to_drawable_circuit` is the only place that reads `n_wires` or a `wires`-keyed
  operation entry. Whatever it returns carries `n_qubits` and `qubits`, and the
  legacy key is dropped rather than duplicated, so no renderer can read the
  stale spelling even if one tries.
* The width is taken from a reported `n_wires`/`n_qubits` before it is inferred
  from the operation history, so a device that touches one qubit of three still
  renders three rows.
* Three presentation keywords the drawer accepts through `**kwargs` —
  `wire_options`, `show_wire_labels`, `active_wire_notches` — are names this
  package published, so they get a forwarder and a `DeprecationWarning` naming
  both spellings, exactly as a renamed parameter does. Silently ignoring them
  was the alternative, and it is worse than either accepting or refusing them:
  the diagram comes back with the labels the caller asked to hide and nothing
  says why.

Where the two spellings disagree, the canonical one wins.

### Why the width is not simply inferred

`min()` on an empty sequence is the symptom, so "infer the width and stop
reading `n_wires`" is the tempting fix. It is wrong twice over. A three-qubit
device whose history happens to touch qubit 0 first would render as one row, and
a device that reports a width and an empty history — a valid thing to draw —
would render as no rows at all. The reported width is authoritative when it
exists; inference is the fallback for objects that report none.

## What this does not change

* No published name changes. The change *accepts more* than before and *emits*
  exactly what the migration publishes. `fq.Circuit.draw`, `fq.draw`, `draw_text`
  and `draw_mpl` keep their signatures; nothing in `docs/public_api_v1.json`
  moves, and the drawer is not part of the Stable Core surface.
* No census counter moves. `[boundary].measured_private` stays `331` on this
  tree: private code keeps `wire` names by the contract's own `private_note`, and
  this change is not a slice retiring one. `_draw_qubits` therefore keeps its
  pre-qubit parameter name, with a comment saying why.
* No frozen payload key moves. `CircuitIR.n_wires` and
  `Instruction.wires` are read exactly as before, and `to_drawable_circuit`
  still requires `n_wires` on the IR path.
* The contract's reported `[other_surfaces].message_strings` is left alone. It is
  documented there as *reported, never reconciled* — a string-literal scan cannot
  tell a serialized key from a refusal sentence — and the line belongs to the
  slice that owns it. Measured against this change, it does not move at all: the
  census returns the *distinct* literals, and every spelling this change
  introduces already appears elsewhere in the package. The declared `1291` against
  a measured `985` on this tree is drift that predates this branch and stays
  WQ-8's to settle.

## A second finding, from the same corner

`tests/unit/test_drawer_mpl_kernels.py` skipped on the old spelling three times
over: nineteen `show_wire_labels=`, one `wire_options=`, and one
`drawer.n_wires` that no longer existed. The reason no gate objected is worth
recording:

* the vocabulary census scans `flagquantum/` and never `tests/`, so a stale
  spelling in a test is not its business;
* `matplotlib` is a `viz` extra, and `[dev]` does not include it, so
  `pytest.importorskip`/module-level skip removes the whole module from every
  job that installs only `.[dev]`;
* exactly one job installs `viz` — the coverage job, whose install line is the
  only one matching `.[dev,jax,viz,...]`.

So the Matplotlib renderer runs in one job out of the dozen in `ci.yml`, and the
tests that pin its per-gate behaviour are invisible to every other lane. They
were repaired here, and the module now passes with `matplotlib` present:

```
$ pytest tests/unit/test_drawer_mpl_kernels.py      # 15 failed, 10 passed before
25 passed
```

| Stale spelling | Count | Canonical |
|---|---:|---|
| `show_wire_labels=` | 19 | `show_qubit_labels=` |
| `wire_options=` | 1 | `qubit_options=` |
| `drawer.n_wires` | 1 | `drawer.n_qubits` |

The `n_wires` this module passes *to* its own `_LegacyDevice` stub is kept: that
stub exists to be a legacy device, and renaming it would delete the subject of
the test.

## Verification

Every claim above is reproducible on this branch with the command shown.

| # | Claim | Command | Result |
|---|---|---|---|
| 1 | the regression is real and raises before the fix | `pytest tests/test_drawer_ir.py` on `721ce1a4` | `1 failed, 2 passed`, `ValueError: min() iterable argument is empty` |
| 2 | text and mpl both draw a legacy device after the fix | `pytest tests/test_drawer_ir.py` | `11 passed` |
| 3 | the whole mpl renderer suite is green after the stale spellings are repaired | `pytest tests/unit/test_drawer_mpl_kernels.py` | `31 passed` (was `15 failed, 10 passed`) |
| 4 | the three legacy keywords warn and still take effect | `pytest tests/unit/test_drawer_mpl_kernels.py -k legacy_option` | `5 passed, 26 deselected`, each under `pytest.warns(DeprecationWarning)` |
| 5 | a third-party device is still drawn from its own vocabulary | `pytest tests/test_drawer_ir.py -k legacy` | `6 passed, 5 deselected`; `test_drawer_keeps_legacy_qdev_compatibility` is kept byte-identical |
| 6 | the three drawer test modules together | `pytest tests/test_drawer_ir.py tests/unit/test_drawer_mpl_kernels.py tests/unit/test_drawer_input_ownership.py` | `45 passed` |
| 7 | no counter moves and no site is retired | `python tools/check_qubit_vocabulary.py` | exit 0, byte-identical before and after: `131 of 341 baseline sites retired, 11 kept as deprecated aliases`; `83 of 122 attribute sites retired, 3 kept as deprecated aliases`; `4 of 10 definition names retired`; `[boundary].measured_private` still `331` |
| 8 | the drawer is still clean under the project's linters | `ruff check flagquantum tests tools` and `black --check flagquantum tests tools` | `All checks passed!` / `1499 files would be left unchanged` |
| 9 | the change is typed cleanly | `mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum` | `checked 646 source files`; the 22 remaining errors are the `sched_getaffinity` `attr-defined` reports in `benchmarking/socket_local_throughput.py`, identical on this branch and on `main` because the attribute is Linux-only and this checkout is macOS |
| 10 | the change stays inside its ownership | `python tools/check_team_scope.py --require-classified --base 721ce1a4` | 8 changed paths, all classified: the drawer and its tests are `core`, the record and the reference page are shared |
| 11 | the lane that installs `viz` loses this failure and gains none | the `coverage` job's own selection, `pytest -m "(smoke or unit or integration or jax) and not slow and not qiskit and not cudaq and not triton"` | see the run recorded on the pull request; the 16 drawer failures are gone and no new drawer failure appears |

`matplotlib` is a `viz` extra that this checkout had to install for rows 3, 4, 5
and 6; without it the module skips itself, which is the behaviour row 3's
"before" column depends on. The `coverage` job is the only lane in `ci.yml`
whose install line names `viz`, which is why the failure reached CI there and
nowhere else.

## The catch-up this fix owes

The compatibility rule this change restores is recorded where the migration's
rules live, not only here:

* `docs/reference/QUBIT_NAMING_MIGRATION.md` — the "Spellings the census cannot
  see" table now separates *published keywords we rename* from *a caller's own
  spelling we only read*, and the drawer rows move from `open` to `migrated`
  and `accepted, not published`.
* `flagquantum/drawer/mpl_drawer.py` — the `draw_mpl` docstring documents the
  canonical keyword names and names the deprecated ones.

The remaining open row is the capture keyword `wires=`, which belongs to the
hybrid-language owner rather than to this migration.
