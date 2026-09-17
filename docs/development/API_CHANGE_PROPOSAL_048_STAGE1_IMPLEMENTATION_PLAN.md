# QEC Code-Independent Records (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a code-independent QEC records layer — a phase-free `Pauli`, a `StabilizerCode` protocol with a distance-parameterized `RepetitionCode`, and a code-driven memory-circuit builder that derives detector and logical-observable layouts from the code instead of asserting them.

**Architecture:** Three new flat modules beside the frozen `flagquantum/qec/` profile: `pauli.py` (operator records), `codes.py` (code descriptions), `circuit.py` (source generation plus detection/observable layouts). Nothing in the frozen repetition profile changes; the new builder is validated by reproducing the frozen `d = 3` circuit instruction for instruction through the unchanged private hybrid compiler.

**Tech Stack:** Python 3.12 in-container, frozen dataclasses, `typing.Protocol`, pytest (markers `unit` and `integration`), the private `flagquantum.compiler._hybrid` capture/lowering path, `torch`-backed dynamic execution.

**Spec:** `docs/development/API_CHANGE_PROPOSAL_048_QEC_DETECTOR_ERROR_MODEL.md` (Stage 1 section; read the Problem, Layer placement, and Stage 1 subsections before starting).

## Global Constraints

- Work in the worktree `/Users/baai/Documents/liuwei/FlagQuantum-qec` on branch `feat/qec-stage1-code-records`. Never `cd` into the primary clone `/Users/baai/Documents/liuwei/FlagQuantum-upstream`; another session owns that working tree.
- Every command runs in Docker. The dev image has an editable install pointing at a stale `/opt/FlagQuantum` snapshot, so install the worktree before testing:
  `docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && <command>"`
- Do not modify `flagquantum/qec/types.py`, `decoders.py`, `repetition.py`, `noise.py`, or `__init__.py`'s existing exports. The seven names pinned by `contracts/hybrid-compilation-private-v0-candidate.json` keep their signatures.
- Do not modify `flagquantum/noise/`, `flagquantum/compiler/`, `flagquantum/runtime/`, or `architecture.toml`.
- Do not touch `capability-maturity.toml`, the generated docs under `docs/generated/`, or any capability row. Staging a capability promotion is Stage 5 work.
- All source, docstring, and comment text is English (`tools/check_repository_language.py` gates this).
- Gates that must pass on every task that changes `flagquantum/`: `black --check`, `ruff check` (the repository's selected rule set requires `zip(..., strict=...)` in every `zip` call), `mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum`, and `python tools/check_architecture.py`.
- Module line ceiling is 1250 for anything other than `flagquantum/__init__.py`. New modules stay far below it.
- Every task ends with a commit. Conventional-commit prefixes used by this repository: `feat:`, `test:`, `docs:`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `flagquantum/qec/pauli.py` (create) | Phase-free sparse Pauli operators over arbitrary wire indices |
| `flagquantum/qec/codes.py` (create) | `CodeCheck`, the `StabilizerCode` protocol, `RepetitionCode` |
| `flagquantum/qec/circuit.py` (create) | Measurement/detector/observable records, `MemoryCircuit`, `build_memory_circuit` |
| `flagquantum/qec/__init__.py` (modify) | Publish the new names beside the frozen ones |
| `flagquantum/qec/IMPLEMENTATION.md` (modify) | Describe the code-independent layer and its boundary |
| `tests/qec/test_pauli.py` (create) | Unit coverage for `Pauli` |
| `tests/qec/test_codes.py` (create) | Unit coverage for `CodeCheck`, `StabilizerCode`, `RepetitionCode` |
| `tests/qec/test_memory_circuit.py` (create) | Unit coverage for the layouts and the builder's validation |
| `tests/qec/test_memory_circuit_execution.py` (create) | Integration coverage: `d = 3` equality, `d = 5/7` execution, injected-error detector signature |

---

### Task 1: Phase-free Pauli operators

**Files:**
- Create: `flagquantum/qec/pauli.py`
- Test: `tests/qec/test_pauli.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Pauli`, a frozen ordered dataclass with fields `x_wires: tuple[int, ...]` and `z_wires: tuple[int, ...]`; a classmethod `from_label(label: str, wire: int) -> Pauli`; a classmethod `from_text(text: str) -> Pauli`; properties `support -> tuple[int, ...]`, `weight -> int`, `is_identity -> bool`; methods `commutes_with(other: Pauli) -> bool`, `__mul__(other: Pauli) -> Pauli`, `to_text() -> str`. A wire present in both collections carries `Y`.

The symplectic convention is the standard one and later tasks depend on it: `P = product of X_w^{a_w} Z_w^{b_w}`, where `a` is `x_wires` and `b` is `z_wires`. So `Z_w` is `Pauli(z_wires=(w,))` and `X_w` is `Pauli(x_wires=(w,))`.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_pauli.py`:

```python
"""Unit coverage for phase-free Pauli operators."""

from __future__ import annotations

import pytest

from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


def test_identity_has_no_support_and_renders_as_i() -> None:
    identity = Pauli()

    assert identity.support == ()
    assert identity.weight == 0
    assert identity.is_identity
    assert identity.to_text() == "I"


@pytest.mark.parametrize(
    ("label", "expected"),
    (
        ("i", Pauli()),
        ("x", Pauli(x_wires=(2,))),
        ("Y", Pauli(x_wires=(2,), z_wires=(2,))),
        ("z", Pauli(z_wires=(2,))),
    ),
)
def test_from_label_maps_every_pauli_letter(label: str, expected: Pauli) -> None:
    assert Pauli.from_label(label, 2) == expected


def test_from_label_rejects_an_unknown_letter() -> None:
    with pytest.raises(ValueError, match="label"):
        Pauli.from_label("q", 0)


def test_from_label_rejects_a_negative_wire() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Pauli.from_label("x", -1)


def test_support_unions_both_components_in_wire_order() -> None:
    pauli = Pauli(x_wires=(1, 3), z_wires=(2,))

    assert pauli.support == (1, 2, 3)
    assert pauli.weight == 3


def test_weight_counts_a_y_wire_once() -> None:
    assert Pauli(x_wires=(0,), z_wires=(0,)).weight == 1


@pytest.mark.parametrize(
    "pauli",
    (
        Pauli(),
        Pauli(x_wires=(0,)),
        Pauli(z_wires=(2,)),
        Pauli(x_wires=(0,), z_wires=(0,)),
        Pauli(x_wires=(0, 4), z_wires=(2,)),
    ),
)
def test_text_round_trips(pauli: Pauli) -> None:
    assert Pauli.from_text(pauli.to_text()) == pauli


@pytest.mark.parametrize("text", ("", "   ", "Q0", "X", "X0**Z1", "I0"))
def test_from_text_rejects_malformed_text(text: str) -> None:
    with pytest.raises(ValueError):
        Pauli.from_text(text)


@pytest.mark.parametrize(
    "wires",
    ({"x_wires": (2, 0)}, {"x_wires": (1, 1)}, {"z_wires": (3, 1)}),
)
def test_unordered_or_repeated_wires_are_rejected(wires: dict) -> None:
    with pytest.raises(ValueError, match="increasing"):
        Pauli(**wires)


def test_negative_wires_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Pauli(z_wires=(-1,))


def test_non_integer_wires_are_rejected() -> None:
    with pytest.raises(TypeError, match="integer"):
        Pauli(x_wires=(1.5,))


def test_booleans_are_not_wire_indices() -> None:
    with pytest.raises(TypeError, match="integer"):
        Pauli(x_wires=(True,))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (Pauli(x_wires=(0,)), Pauli(z_wires=(0,)), False),
        (Pauli(x_wires=(0,)), Pauli(z_wires=(1,)), True),
        (Pauli(x_wires=(0, 1)), Pauli(z_wires=(0, 1)), True),
        (Pauli(x_wires=(0,), z_wires=(0,)), Pauli(z_wires=(0,)), False),
        (Pauli(), Pauli(x_wires=(5,), z_wires=(5,)), True),
    ),
)
def test_commutes_with_uses_the_symplectic_product(
    left: Pauli, right: Pauli, expected: bool
) -> None:
    assert left.commutes_with(right) is expected
    assert right.commutes_with(left) is expected


def test_composition_cancels_a_repeated_letter() -> None:
    assert Pauli(x_wires=(0,)) * Pauli(x_wires=(0,)) == Pauli()


def test_composition_of_x_and_z_on_one_wire_is_y() -> None:
    assert Pauli(x_wires=(0,)) * Pauli(z_wires=(0,)) == Pauli(
        x_wires=(0,), z_wires=(0,)
    )


def test_composition_is_commutative_and_associative() -> None:
    left = Pauli(x_wires=(0, 2))
    middle = Pauli(z_wires=(2, 3))
    right = Pauli(x_wires=(3,))

    assert left * middle == middle * left
    assert (left * middle) * right == left * (middle * right)


def test_equal_paulis_compare_equal_and_hash_equal() -> None:
    assert Pauli(x_wires=(1,)) == Pauli(x_wires=(1,))
    assert len({Pauli(x_wires=(1,)), Pauli(x_wires=(1,))}) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_pauli.py -q"
```

Expected: collection error, `ModuleNotFoundError: No module named 'flagquantum.qec.pauli'`.

- [ ] **Step 3: Write the implementation**

Create `flagquantum/qec/pauli.py`:

```python
"""Phase-free Pauli operators over arbitrary wire indices.

The symplectic convention is the standard one: a wire listed in ``x_wires``
carries an ``X`` factor, a wire listed in ``z_wires`` carries a ``Z`` factor,
and a wire listed in both carries ``Y``. Global phase is not tracked, so an
operator is described only up to a factor of ``+/-1`` or ``+/-i``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

_LABELS = ("i", "x", "y", "z")


def _wire_tuple(wires: Iterable[int], *, owner: str) -> tuple[int, ...]:
    """Validate one strictly increasing, duplicate-free wire collection."""

    values = tuple(wires)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError(f"{owner} must contain integer wire indices")
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{owner} must contain non-negative wire indices")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{owner} must be strictly increasing without duplicates")
    return normalized


@dataclass(frozen=True, order=True)
class Pauli:
    """A phase-free Pauli operator over arbitrary wire indices.

    A wire listed in both ``x_wires`` and ``z_wires`` carries ``Y``. Ordering is
    lexicographic on ``(x_wires, z_wires)``, so operators sort deterministically.
    """

    x_wires: tuple[int, ...] = ()
    z_wires: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x_wires", _wire_tuple(self.x_wires, owner="Pauli X wires")
        )
        object.__setattr__(
            self, "z_wires", _wire_tuple(self.z_wires, owner="Pauli Z wires")
        )

    @classmethod
    def from_label(cls, label: str, wire: int) -> Pauli:
        """Build the single-wire operator named by ``label`` (``I``, ``X``, ``Y``, ``Z``)."""

        normalized = label.lower()
        if normalized not in _LABELS:
            raise ValueError("Pauli label must be one of I, X, Y, or Z")
        if wire < 0:
            raise ValueError("Pauli wire must be non-negative")
        if normalized == "i":
            return cls()
        if normalized == "x":
            return cls(x_wires=(wire,))
        if normalized == "z":
            return cls(z_wires=(wire,))
        return cls(x_wires=(wire,), z_wires=(wire,))

    @classmethod
    def from_text(cls, text: str) -> Pauli:
        """Parse the form produced by :meth:`to_text`."""

        stripped = text.strip()
        if not stripped:
            raise ValueError("Pauli text must not be empty")
        if stripped == "I":
            return cls()
        x_wires: list[int] = []
        z_wires: list[int] = []
        for term in stripped.split("*"):
            if not term:
                raise ValueError("Pauli text must not contain an empty term")
            label = term[0].lower()
            if label not in _LABELS or label == "i":
                raise ValueError("Pauli text terms must start with X, Y, or Z")
            digits = term[1:]
            if not digits.isdigit():
                raise ValueError("Pauli text terms must end with a wire index")
            wire = int(digits)
            if label in ("x", "y"):
                x_wires.append(wire)
            if label in ("z", "y"):
                z_wires.append(wire)
        return cls(x_wires=tuple(x_wires), z_wires=tuple(z_wires))

    @property
    def support(self) -> tuple[int, ...]:
        """Wire indices carrying a non-identity factor, in ascending order."""

        return tuple(sorted(set(self.x_wires) | set(self.z_wires)))

    @property
    def weight(self) -> int:
        """Number of wires carrying a non-identity factor."""

        return len(self.support)

    @property
    def is_identity(self) -> bool:
        """Whether the operator acts as the identity on every wire."""

        return not self.x_wires and not self.z_wires

    def commutes_with(self, other: Pauli) -> bool:
        """Whether the two operators commute, ignoring global phase."""

        x_wires = set(self.x_wires)
        z_wires = set(self.z_wires)
        overlap = sum(1 for wire in other.x_wires if wire in z_wires)
        overlap += sum(1 for wire in other.z_wires if wire in x_wires)
        return overlap % 2 == 0

    def __mul__(self, other: Pauli) -> Pauli:
        """Compose two operators, dropping the phase of the product."""

        return Pauli(
            x_wires=tuple(sorted(set(self.x_wires) ^ set(other.x_wires))),
            z_wires=tuple(sorted(set(self.z_wires) ^ set(other.z_wires))),
        )

    def to_text(self) -> str:
        """Render a deterministic text form such as ``X0*Z2`` or ``I``."""

        x_wires = set(self.x_wires)
        z_wires = set(self.z_wires)
        terms = []
        for wire in self.support:
            if wire in x_wires and wire in z_wires:
                terms.append(f"Y{wire}")
            elif wire in x_wires:
                terms.append(f"X{wire}")
            else:
                terms.append(f"Z{wire}")
        return "*".join(terms) if terms else "I"


__all__ = ("Pauli",)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_pauli.py -q && black --check flagquantum/qec/pauli.py tests/qec/test_pauli.py && ruff check flagquantum/qec/pauli.py tests/qec/test_pauli.py && mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum/qec/pauli.py"
```

Expected: all tests pass, then each gate prints a success line.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/pauli.py tests/qec/test_pauli.py
git commit -m "feat: add a phase-free Pauli operator for QEC records"
```

---

### Task 2: Code descriptions

**Files:**
- Create: `flagquantum/qec/codes.py`
- Test: `tests/qec/test_codes.py`

**Interfaces:**
- Consumes: `Pauli` from `flagquantum/qec/pauli.py` (Task 1).
- Produces: `CodeCheck`, a frozen dataclass with fields `index: int`, `stabilizer: Pauli`, `ancilla_wire: int`, `cnot_wires: tuple[tuple[int, int], ...]` where each pair is `(control, target)`. `StabilizerCode`, a `runtime_checkable` `Protocol` with read-only properties `distance -> int`, `num_data_qubits -> int`, `num_ancilla_qubits -> int`, `data_wires -> tuple[int, ...]`, `ancilla_wires -> tuple[int, ...]`, `checks -> tuple[CodeCheck, ...]`, `stabilizers -> tuple[Pauli, ...]`, `logical_observables -> tuple[Pauli, ...]`. `RepetitionCode`, a frozen dataclass with field `distance: int = 3`.

`RepetitionCode` wires: data `0..distance-1`, ancillas `distance..2*distance-2`. Check `c` measures `Z_c Z_{c+1}` with `CNOT(c, ancilla)` then `CNOT(c+1, ancilla)`, so the code detects bit flips. `logical_observables` is a single `Z` operator on every data wire, which makes its readout representative the parity of the terminal data measurements and matches the stim repetition-memory convention that Stage 2 must interoperate with.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_codes.py`:

```python
"""Unit coverage for code descriptions and the repetition-code reference."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from flagquantum.qec.codes import CodeCheck, RepetitionCode, StabilizerCode
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


def test_default_distance_matches_the_reference_profile() -> None:
    assert RepetitionCode().distance == 3


def test_distance_three_wire_layout() -> None:
    code = RepetitionCode(3)

    assert code.num_data_qubits == 3
    assert code.num_ancilla_qubits == 2
    assert code.data_wires == (0, 1, 2)
    assert code.ancilla_wires == (3, 4)


def test_distance_three_checks_are_adjacent_z_pairs() -> None:
    code = RepetitionCode(3)

    assert len(code.checks) == 2
    for index, check in enumerate(code.checks):
        assert check.index == index
        assert check.stabilizer == Pauli(z_wires=(index, index + 1))
        assert check.ancilla_wire == 3 + index
        assert check.cnot_wires == ((index, 3 + index), (index + 1, 3 + index))


def test_stabilizers_are_the_declared_check_operators() -> None:
    code = RepetitionCode(5)

    assert code.stabilizers == tuple(check.stabilizer for check in code.checks)
    assert code.stabilizers == (
        Pauli(z_wires=(0, 1)),
        Pauli(z_wires=(1, 2)),
        Pauli(z_wires=(2, 3)),
        Pauli(z_wires=(3, 4)),
    )


def test_logical_observable_is_z_type_and_commutes_with_every_stabilizer() -> None:
    code = RepetitionCode(5)

    (observable,) = code.logical_observables
    assert observable == Pauli(z_wires=(0, 1, 2, 3, 4))
    assert not observable.x_wires
    assert all(observable.commutes_with(item) for item in code.stabilizers)


def test_scaled_layouts_keep_checks_and_ancillas_contiguous() -> None:
    code = RepetitionCode(7)

    assert code.num_ancilla_qubits == 6
    assert code.ancilla_wires == (7, 8, 9, 10, 11, 12)
    assert len(code.checks) == 6
    assert code.checks[-1].cnot_wires == ((5, 12), (6, 12))


def test_distance_two_is_the_smallest_supported_code() -> None:
    code = RepetitionCode(2)

    assert code.num_ancilla_qubits == 1
    assert code.ancilla_wires == (2,)
    assert len(code.checks) == 1
    assert code.checks[0].stabilizer == Pauli(z_wires=(0, 1))


@pytest.mark.parametrize("distance", (1, 0, -3))
def test_distances_below_two_are_rejected(distance: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        RepetitionCode(distance)


@pytest.mark.parametrize("distance", (2.5, "3", None, True))
def test_non_integer_distances_are_rejected(distance: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        RepetitionCode(distance)  # type: ignore[arg-type]


def test_repetition_code_satisfies_the_stabilizer_code_protocol() -> None:
    assert isinstance(RepetitionCode(3), StabilizerCode)


def test_repetition_codes_are_frozen_and_hashable() -> None:
    assert RepetitionCode(3) == RepetitionCode(3)
    assert len({RepetitionCode(3), RepetitionCode(3)}) == 1
    with pytest.raises(FrozenInstanceError):
        RepetitionCode(3).distance = 5  # type: ignore[misc]


def test_check_rejects_a_cnot_that_does_not_target_the_ancilla() -> None:
    with pytest.raises(ValueError, match="target"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 1), (1, 2)),
        )


def test_check_rejects_a_cnot_that_controls_the_ancilla() -> None:
    with pytest.raises(ValueError, match="control"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((2, 2),),
        )


def test_check_rejects_an_empty_cnot_schedule() -> None:
    with pytest.raises(ValueError, match="at least one CNOT"):
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=(),
        )


def test_check_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError, match="index"):
        CodeCheck(
            index=-1,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2)),
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_codes.py -q"
```

Expected: collection error, `ModuleNotFoundError: No module named 'flagquantum.qec.codes'`.

- [ ] **Step 3: Write the implementation**

Create `flagquantum/qec/codes.py`:

```python
"""Code-independent stabilizer-code descriptions.

A code declares its wire layout, its checks, and its logical observables. The
frozen repetition profile keeps its own records; this module describes a code as
a value so that circuit generation, detector layout, and decoding can be derived
from it rather than pinned to one instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

from .pauli import Pauli


@dataclass(frozen=True)
class CodeCheck:
    """One stabilizer check with its ancilla and its data-to-ancilla CNOTs.

    Each entry of ``cnot_wires`` is a ``(control, target)`` pair. The control is
    a data wire and the target is the check's ancilla.
    """

    index: int
    stabilizer: Pauli
    ancilla_wire: int
    cnot_wires: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("check index must be non-negative")
        if not isinstance(self.stabilizer, Pauli):
            raise TypeError("check stabilizer must be a Pauli operator")
        if self.ancilla_wire < 0:
            raise ValueError("check ancilla wire must be non-negative")
        if not self.cnot_wires:
            raise ValueError("check must declare at least one CNOT")
        for control, target in self.cnot_wires:
            if control < 0 or target < 0:
                raise ValueError("check CNOT wires must be non-negative")
        if any(control == self.ancilla_wire for control, _ in self.cnot_wires):
            raise ValueError("check CNOTs must control data wires, not the ancilla")
        if any(target != self.ancilla_wire for _, target in self.cnot_wires):
            raise ValueError("check CNOTs must target the declared ancilla")


@runtime_checkable
class StabilizerCode(Protocol):
    """A code that declares its wire layout, checks, and logical observables."""

    @property
    def distance(self) -> int: ...

    @property
    def num_data_qubits(self) -> int: ...

    @property
    def num_ancilla_qubits(self) -> int: ...

    @property
    def data_wires(self) -> tuple[int, ...]: ...

    @property
    def ancilla_wires(self) -> tuple[int, ...]: ...

    @property
    def checks(self) -> tuple[CodeCheck, ...]: ...

    @property
    def stabilizers(self) -> tuple[Pauli, ...]: ...

    @property
    def logical_observables(self) -> tuple[Pauli, ...]: ...


@dataclass(frozen=True)
class RepetitionCode:
    """The bit-flip repetition code with ``distance`` data qubits.

    Data qubits occupy wires ``0..distance-1`` and check ancillas occupy wires
    ``distance..2*distance-2``. Check ``c`` measures ``Z_c Z_{c+1}``, so the code
    detects bit flips. The single declared logical observable is ``Z`` on every
    data wire, which makes its readout representative the parity of the terminal
    data measurements.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("repetition-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("repetition-code distance must be at least two")

    @property
    def num_data_qubits(self) -> int:
        return self.distance

    @property
    def num_ancilla_qubits(self) -> int:
        return self.distance - 1

    @property
    def data_wires(self) -> tuple[int, ...]:
        return tuple(range(self.distance))

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return tuple(range(self.distance, 2 * self.distance - 1))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return tuple(
            CodeCheck(
                index=index,
                stabilizer=Pauli(z_wires=(index, index + 1)),
                ancilla_wire=self.distance + index,
                cnot_wires=(
                    (index, self.distance + index),
                    (index + 1, self.distance + index),
                ),
            )
            for index in range(self.distance - 1)
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=self.data_wires),)


__all__ = ("CodeCheck", "RepetitionCode", "StabilizerCode")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_codes.py -q && black --check flagquantum/qec/codes.py tests/qec/test_codes.py && ruff check flagquantum/qec/codes.py tests/qec/test_codes.py && mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum/qec"
```

Expected: all tests pass, then each gate prints a success line. If the `control` / `target` rejection tests are the wrong way round relative to the implementation's message text, fix the message text in `codes.py`, not the test expectation.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/codes.py tests/qec/test_codes.py
git commit -m "feat: describe a code as a value with the StabilizerCode protocol"
```

---

### Task 3: Detector, observable, and memory-circuit records

**Files:**
- Create: `flagquantum/qec/circuit.py`
- Test: `tests/qec/test_memory_circuit.py`

**Interfaces:**
- Consumes: `Pauli` (Task 1); `StabilizerCode`, `RepetitionCode`, `CodeCheck` (Task 2).
- Produces:
  - `MeasurementRef(round_index: int | None, wire: int)` — `round_index is None` denotes the terminal data readout.
  - `Detector(index: int, parity: tuple[MeasurementRef, ...])`.
  - `DetectorLayout(detectors: tuple[Detector, ...])` with `__len__`.
  - `LogicalObservable(index: int, pauli: Pauli, measurement_parity: tuple[MeasurementRef, ...])`.
  - `ObservableLayout(observables: tuple[LogicalObservable, ...])` with `__len__`.
  - `MemoryCircuit(code: StabilizerCode, rounds: int, source: str, detectors: DetectorLayout, observables: ObservableLayout)`.
  - `build_memory_circuit(code: StabilizerCode, *, rounds: int) -> MemoryCircuit`.

Detector grammar for a `rounds`-round distance-`d` memory experiment, `(d - 1) * (rounds + 1)` detectors, dense and round-major:

- Detector `r * (d - 1) + c` for `r` in `[0, rounds)`: parity `{S(r, c)}` when `r == 0`, and `{S(r, c), S(r - 1, c)}` otherwise, where `S(r, c)` is `MeasurementRef(r, ancilla_wire_of_check_c)`.
- Detector `rounds * (d - 1) + c`: parity `{S(rounds - 1, c), M(c), M(c + 1)}`, where `M(w)` is `MeasurementRef(None, w)` and `c, c + 1` are the check's stabilizer support.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_memory_circuit.py`:

```python
"""Unit coverage for detector and observable layouts and the circuit builder."""

from __future__ import annotations

import pytest

from flagquantum.qec.circuit import (
    Detector,
    DetectorLayout,
    LogicalObservable,
    MeasurementRef,
    MemoryCircuit,
    ObservableLayout,
    build_memory_circuit,
)
from flagquantum.qec.codes import CodeCheck, RepetitionCode
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


class _TwoQubitParityCode:
    """A second code, to prove the builder is not pinned to the repetition code."""

    distance = 2
    num_data_qubits = 2
    num_ancilla_qubits = 1
    data_wires = (0, 1)
    ancilla_wires = (2,)
    checks = (
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_wires=(0, 1)),
            ancilla_wire=2,
            cnot_wires=((0, 2), (1, 2)),
        ),
    )
    stabilizers = (Pauli(z_wires=(0, 1)),)
    logical_observables = (Pauli(z_wires=(0, 1)),)


def test_detector_count_matches_the_stated_formula() -> None:
    for distance, rounds in ((3, 3), (2, 1), (5, 5), (7, 7)):
        built = build_memory_circuit(RepetitionCode(distance), rounds=rounds)
        assert len(built.detectors) == (distance - 1) * (rounds + 1)


def test_detector_indices_are_dense_and_ordered() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    assert tuple(item.index for item in built.detectors.detectors) == tuple(range(8))


def test_round_zero_detectors_reference_only_that_round() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    first, second = built.detectors.detectors[0], built.detectors.detectors[1]
    assert first.parity == (MeasurementRef(0, 3),)
    assert second.parity == (MeasurementRef(0, 4),)


def test_later_round_detectors_compare_with_the_previous_round() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    third = built.detectors.detectors[2]
    assert set(third.parity) == {MeasurementRef(1, 3), MeasurementRef(0, 3)}


def test_final_boundary_detectors_join_the_last_round_to_the_data_readout() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    final_left, final_right = built.detectors.detectors[6], built.detectors.detectors[7]
    assert set(final_left.parity) == {
        MeasurementRef(2, 3),
        MeasurementRef(None, 0),
        MeasurementRef(None, 1),
    }
    assert set(final_right.parity) == {
        MeasurementRef(2, 4),
        MeasurementRef(None, 1),
        MeasurementRef(None, 2),
    }


def test_observable_layout_declares_the_terminal_data_parity() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    assert len(built.observables) == 1
    (observable,) = built.observables.observables
    assert observable.index == 0
    assert observable.pauli == Pauli(z_wires=(0, 1, 2))
    assert observable.measurement_parity == (
        MeasurementRef(None, 0),
        MeasurementRef(None, 1),
        MeasurementRef(None, 2),
    )


def test_source_declares_every_check_and_returns_the_last_measurement() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    assert "qp.CNOT(wires=[0, 3])" in built.source
    assert "qp.CNOT(wires=[1, 4])" in built.source
    assert "qp.measure(wires=4)" in built.source
    assert "qp.reset(wires=4)" in built.source
    assert built.source.rstrip().endswith("return last")


def test_builder_accepts_a_second_code_without_repetition_assumptions() -> None:
    built = build_memory_circuit(_TwoQubitParityCode(), rounds=1)

    assert built.code.distance == 2
    assert len(built.detectors) == 2
    assert "qp.CNOT(wires=[0, 2])" in built.source
    assert "qp.CNOT(wires=[1, 2])" in built.source


def test_builder_rounds_the_round_count_onto_the_record() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=4)

    assert built.rounds == 4
    assert isinstance(built, MemoryCircuit)


def test_builder_rejects_a_non_code_object() -> None:
    with pytest.raises(TypeError, match="StabilizerCode"):
        build_memory_circuit(object(), rounds=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("rounds", (0, -1))
def test_builder_rejects_a_non_positive_round_count(rounds: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_memory_circuit(RepetitionCode(3), rounds=rounds)


def test_builder_rejects_a_non_integer_round_count() -> None:
    with pytest.raises(TypeError, match="integer"):
        build_memory_circuit(RepetitionCode(3), rounds=2.5)  # type: ignore[arg-type]


def test_memory_circuit_rejects_a_mismatched_detector_count() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    with pytest.raises(ValueError, match="detector count"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source=built.source,
            detectors=DetectorLayout(built.detectors.detectors[:-1]),
            observables=built.observables,
        )


def test_memory_circuit_rejects_empty_source() -> None:
    built = build_memory_circuit(RepetitionCode(3), rounds=3)

    with pytest.raises(ValueError, match="source"):
        MemoryCircuit(
            code=built.code,
            rounds=built.rounds,
            source="   ",
            detectors=built.detectors,
            observables=built.observables,
        )


def test_measurement_reference_accepts_a_terminal_readout() -> None:
    assert MeasurementRef(None, 0).round_index is None


@pytest.mark.parametrize("round_index", (-1,))
def test_measurement_reference_rejects_a_negative_round(round_index: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MeasurementRef(round_index, 0)


def test_measurement_reference_rejects_a_negative_wire() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MeasurementRef(0, -1)


def test_measurement_reference_rejects_a_non_integer_wire() -> None:
    with pytest.raises(TypeError, match="integer"):
        MeasurementRef(0, 1.5)  # type: ignore[arg-type]


def test_detector_rejects_an_empty_parity() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        Detector(index=0, parity=())


def test_detector_rejects_a_repeated_measurement() -> None:
    reference = MeasurementRef(0, 3)

    with pytest.raises(ValueError, match="repeat"):
        Detector(index=0, parity=(reference, reference))


def test_detector_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError, match="index"):
        Detector(index=-1, parity=(MeasurementRef(0, 3),))


def test_detector_layout_requires_a_dense_ordering() -> None:
    with pytest.raises(ValueError, match="dense"):
        DetectorLayout(
            (Detector(index=1, parity=(MeasurementRef(0, 3),)),),
        )


def test_detector_layout_requires_at_least_one_detector() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorLayout(())


def test_logical_observable_rejects_an_identity_operator() -> None:
    with pytest.raises(ValueError, match="identity"):
        LogicalObservable(index=0, pauli=Pauli(), measurement_parity=())


def test_logical_observable_rejects_an_x_type_operator() -> None:
    with pytest.raises(ValueError, match="Z-type"):
        LogicalObservable(
            index=0,
            pauli=Pauli(x_wires=(0,)),
            measurement_parity=(MeasurementRef(None, 0),),
        )


def test_observable_layout_requires_a_dense_ordering() -> None:
    with pytest.raises(ValueError, match="dense"):
        ObservableLayout(
            (
                LogicalObservable(
                    index=2,
                    pauli=Pauli(z_wires=(0,)),
                    measurement_parity=(MeasurementRef(None, 0),),
                ),
            )
        )


def test_observable_layout_requires_at_least_one_observable() -> None:
    with pytest.raises(ValueError, match="at least one observable"):
        ObservableLayout(())
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_memory_circuit.py -q"
```

Expected: collection error, `ModuleNotFoundError: No module named 'flagquantum.qec.circuit'`.

- [ ] **Step 3: Write the implementation**

Create `flagquantum/qec/circuit.py`:

```python
"""Code-driven memory-circuit source with detection and observable layouts.

The source is a bounded hybrid-compiler program that runs the configured number
of syndrome-extraction rounds and returns the final check measurement. Detector
and observable identity is derived from the code, so a caller never asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .codes import StabilizerCode
from .pauli import Pauli

_FUNCTION_NAME = "memory_experiment"


@dataclass(frozen=True)
class MeasurementRef:
    """One measurement location, by round and wire.

    ``round_index`` is ``None`` for the terminal data readout that follows the
    final syndrome round. The record is not ordered, because an optional round
    index has no total order.
    """

    round_index: int | None
    wire: int

    def __post_init__(self) -> None:
        if self.round_index is not None:
            if isinstance(self.round_index, bool) or not isinstance(
                self.round_index, Integral
            ):
                raise TypeError("measurement round index must be an integer or None")
            if self.round_index < 0:
                raise ValueError("measurement round index must be non-negative")
        if isinstance(self.wire, bool) or not isinstance(self.wire, Integral):
            raise TypeError("measurement wire must be an integer")
        if self.wire < 0:
            raise ValueError("measurement wire must be non-negative")


@dataclass(frozen=True)
class Detector:
    """One measurement parity that is deterministic in the noiseless circuit."""

    index: int
    parity: tuple[MeasurementRef, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("detector index must be non-negative")
        if not self.parity:
            raise ValueError("detector must reference at least one measurement")
        for reference in self.parity:
            if not isinstance(reference, MeasurementRef):
                raise TypeError(
                    "detector parity entries must be MeasurementRef records"
                )
        if len(set(self.parity)) != len(self.parity):
            raise ValueError("a detector cannot repeat a measurement reference")


@dataclass(frozen=True)
class DetectorLayout:
    """Dense, ordered detectors for one configured memory experiment."""

    detectors: tuple[Detector, ...]

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValueError("detector layout requires at least one detector")
        expected = tuple(range(len(self.detectors)))
        if tuple(item.index for item in self.detectors) != expected:
            raise ValueError("detector layout must be dense and ordered from zero")

    def __len__(self) -> int:
        return len(self.detectors)


@dataclass(frozen=True)
class LogicalObservable:
    """One declared logical operator and the readout parity that realizes it."""

    index: int
    pauli: Pauli
    measurement_parity: tuple[MeasurementRef, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("observable index must be non-negative")
        if not isinstance(self.pauli, Pauli):
            raise TypeError("observable operator must be a Pauli operator")
        if self.pauli.is_identity:
            raise ValueError("observable operator must not be the identity")
        if self.pauli.x_wires:
            raise ValueError("observable readout supports Z-type operators only")


@dataclass(frozen=True)
class ObservableLayout:
    """Dense, ordered logical observables for one configured memory experiment."""

    observables: tuple[LogicalObservable, ...]

    def __post_init__(self) -> None:
        if not self.observables:
            raise ValueError("observable layout requires at least one observable")
        expected = tuple(range(len(self.observables)))
        if tuple(item.index for item in self.observables) != expected:
            raise ValueError("observable layout must be dense and ordered from zero")

    def __len__(self) -> int:
        return len(self.observables)


@dataclass(frozen=True)
class MemoryCircuit:
    """Code-driven memory-experiment source with its detection layouts."""

    code: StabilizerCode
    rounds: int
    source: str
    detectors: DetectorLayout
    observables: ObservableLayout

    def __post_init__(self) -> None:
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral):
            raise TypeError("memory rounds must be an integer")
        if self.rounds <= 0:
            raise ValueError("memory rounds must be positive")
        if not self.source.strip():
            raise ValueError("memory circuit source must not be empty")
        expected = (self.code.distance - 1) * (self.rounds + 1)
        if len(self.detectors) != expected:
            raise ValueError(
                "memory circuit detector count must match "
                "(distance - 1) * (rounds + 1)"
            )


def _check_source(code: StabilizerCode) -> str:
    lines = [
        f"def {_FUNCTION_NAME}(rounds):",
        "    last = False",
        "    for round_index in range(rounds):",
    ]
    for check in code.checks:
        for control, target in check.cnot_wires:
            lines.append(f"        qp.CNOT(wires=[{control}, {target}])")
        lines.append(f"        last = qp.measure(wires={check.ancilla_wire})")
        lines.append(f"        qp.reset(wires={check.ancilla_wire})")
    lines.append("    return last")
    return "\n".join(lines) + "\n"


def _detector_layout(code: StabilizerCode, *, rounds: int) -> DetectorLayout:
    checks = code.checks
    detectors: list[Detector] = []
    for round_index in range(rounds):
        for check in checks:
            parity = [MeasurementRef(round_index, check.ancilla_wire)]
            if round_index > 0:
                parity.append(MeasurementRef(round_index - 1, check.ancilla_wire))
            detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    for check in checks:
        parity = [MeasurementRef(rounds - 1, check.ancilla_wire)]
        parity.extend(MeasurementRef(None, wire) for wire in check.stabilizer.support)
        detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    return DetectorLayout(tuple(detectors))


def _observable_layout(code: StabilizerCode) -> ObservableLayout:
    return ObservableLayout(
        tuple(
            LogicalObservable(
                index=index,
                pauli=pauli,
                measurement_parity=tuple(
                    MeasurementRef(None, wire) for wire in pauli.support
                ),
            )
            for index, pauli in enumerate(code.logical_observables)
        )
    )


def build_memory_circuit(code: StabilizerCode, *, rounds: int) -> MemoryCircuit:
    """Build a code's memory-experiment source and its detection layouts.

    Detector ``r * (distance - 1) + c`` compares check ``c`` in round ``r`` with
    the same check in round ``r - 1``, where round ``-1`` is the known all-zero
    prior state. The last ``distance - 1`` detectors compare the final syndrome
    round with the terminal data readout. The source itself is a bounded hybrid
    compiler program; this function neither lowers nor executes it.
    """

    if not isinstance(code, StabilizerCode):
        raise TypeError("code must implement the StabilizerCode protocol")
    if isinstance(rounds, bool) or not isinstance(rounds, Integral):
        raise TypeError("rounds must be an integer")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    configured_rounds = int(rounds)
    return MemoryCircuit(
        code=code,
        rounds=configured_rounds,
        source=_check_source(code),
        detectors=_detector_layout(code, rounds=configured_rounds),
        observables=_observable_layout(code),
    )


__all__ = (
    "Detector",
    "DetectorLayout",
    "LogicalObservable",
    "MeasurementRef",
    "MemoryCircuit",
    "ObservableLayout",
    "build_memory_circuit",
)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_memory_circuit.py -q && black --check flagquantum/qec/circuit.py tests/qec/test_memory_circuit.py && ruff check flagquantum/qec/circuit.py tests/qec/test_memory_circuit.py && mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum/qec"
```

Expected: all tests pass, then each gate prints a success line.

- [ ] **Step 5: Commit**

```bash
git add flagquantum/qec/circuit.py tests/qec/test_memory_circuit.py
git commit -m "feat: derive detector and observable layouts from the code"
```

---

### Task 4: Execution acceptance — `d = 3` equality, scaling, and layout semantics

**Files:**
- Create: `tests/qec/test_memory_circuit_execution.py`

**Interfaces:**
- Consumes: `RepetitionCode` (Task 2); `MeasurementRef`, `MemoryCircuit`, `build_memory_circuit` (Task 3); the frozen `_memory_source` and `ErrorSchedule` from `flagquantum.qec.repetition` and `flagquantum.qec.types`; `INDEX`, `capture_source`, `lower_dynamic_program` from `flagquantum.compiler._hybrid`; `execute_hybrid_dynamic_session` from `flagquantum.runtime.dynamic.hybrid_session`.
- Produces: nothing consumed by later tasks. This task is Stage 1's acceptance gate.

Three claims are pinned here, and they are the ones a reviewer should check first:

1. The general builder's `d = 3` circuit lowers to the same instruction sequence as the frozen profile's. This is what makes the new layer a refactor of the same circuit rather than a second, different one.
2. `d = 5` and `d = 7` lower and execute through the unchanged compiler and runtime.
3. The declared detectors are the *right* parities. Noiseless determinism alone cannot show this, because every parity in a deterministic noiseless circuit is constant. The discriminating test injects one known `X` error and asserts the exact fired-detector pattern and the observable parity.

The expected signature for an `X` error on data wire 0 at round 0, with `distance = 3` and `rounds = 3`: the error flips `S(0, 0)` and the terminal measurement `M(0)`, and persists in every later round. So `D(0, 0) = S(0, 0)` flips; `D(r, 0) = S(r, 0) XOR S(r-1, 0)` stays zero for `r >= 1`; `D_final(0) = S(2, 0) XOR M(0) XOR M(1)` receives two flips and stays zero. Exactly detector index 0 fires, and the observable parity is 1.

- [ ] **Step 1: Write the failing test**

Create `tests/qec/test_memory_circuit_execution.py`:

```python
"""Integration coverage for the code-driven memory circuit."""

from __future__ import annotations

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.qec.circuit import MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.repetition import _memory_source
from flagquantum.qec.types import ErrorSchedule
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration


def _lower(source: str, *, checks: int, rounds: int):
    program = capture_source(source, (INDEX,))
    return lower_dynamic_program(
        program, (rounds,), max_dynamic_measurements=rounds * checks
    )


def _run(
    memory: MemoryCircuit, *, shots: int
) -> tuple[list[list[int]], list[list[int]]]:
    lowered = _lower(
        memory.source, checks=len(memory.code.checks), rounds=memory.rounds
    )
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=shots, seed=0)
    return execution.classical_bits.tolist(), execution.samples.tolist()


def _with_injected_x(source: str, *, round_index: int, wire: int) -> str:
    anchor = "    for round_index in range(rounds):\n"
    assert source.count(anchor) == 1
    injection = (
        f"        if round_index == {round_index}:\n"
        f"            qp.X(wires={wire})\n"
    )
    return source.replace(anchor, anchor + injection)


def _detector_bits(
    memory: MemoryCircuit, classical: list[int], sample: list[int]
) -> tuple[int, ...]:
    code = memory.code
    checks = len(code.checks)
    ancilla_index = {check.ancilla_wire: check.index for check in code.checks}

    def value(reference) -> int:
        if reference.round_index is None:
            return int(sample[reference.wire])
        offset = reference.round_index * checks + ancilla_index[reference.wire]
        return int(classical[offset])

    return tuple(
        sum(value(reference) for reference in detector.parity) % 2
        for detector in memory.detectors.detectors
    )


def _observable_bits(memory: MemoryCircuit, sample: list[int]) -> tuple[int, ...]:
    return tuple(
        sum(int(sample[reference.wire]) for reference in observable.measurement_parity)
        % 2
        for observable in memory.observables.observables
    )


def test_distance_three_matches_the_frozen_circuit_exactly() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)

    general = _lower(memory.source, checks=2, rounds=3).circuit
    frozen = _lower(
        _memory_source(ErrorSchedule(), compiled_feedback=False), checks=2, rounds=3
    ).circuit

    assert general.n_wires == frozen.n_wires == 5
    assert general.instructions == frozen.instructions


@pytest.mark.parametrize("distance", (5, 7))
def test_scaled_distances_lower_and_execute_unchanged(distance: int) -> None:
    rounds = distance
    memory = build_memory_circuit(RepetitionCode(distance), rounds=rounds)

    lowered = _lower(memory.source, checks=distance - 1, rounds=rounds)
    assert lowered.circuit.n_wires == 2 * distance - 1

    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=8, seed=0)
    assert execution.samples.shape == (8, 2 * distance - 1)
    assert execution.classical_bits.shape == (8, rounds * (distance - 1))


@pytest.mark.parametrize("distance", (3, 5))
def test_noiseless_run_fires_no_detector_and_no_observable(distance: int) -> None:
    memory = build_memory_circuit(RepetitionCode(distance), rounds=3)
    classical_rows, sample_rows = _run(memory, shots=4)
    expected = (0,) * (distance - 1) * 4

    for classical, sample in zip(classical_rows, sample_rows, strict=True):
        assert _detector_bits(memory, classical, sample) == expected
        assert _observable_bits(memory, sample) == (0,)


def test_one_injected_error_fires_exactly_the_round_zero_boundary_detector() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    injected = _with_injected_x(memory.source, round_index=0, wire=0)
    lowered = _lower(injected, checks=2, rounds=3)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (1, 0, 0, 0, 0, 0, 0, 0)
        assert _observable_bits(memory, sample) == (1,)


def test_injected_error_on_the_middle_wire_fires_two_detectors() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    injected = _with_injected_x(memory.source, round_index=0, wire=1)
    lowered = _lower(injected, checks=2, rounds=3)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=2, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (1, 1, 0, 0, 0, 0, 0, 0)
        assert _observable_bits(memory, sample) == (1,)
```

Before running, re-derive the middle-wire expectation rather than trusting it: an `X` on data wire 1 flips `Z_0 Z_1` and `Z_1 Z_2`, so `S(r, 0)` and `S(r, 1)` are both 1 in every round. Then `D(0, 0) = 1` and `D(0, 1) = 1`; later rounds cancel; each final-boundary detector receives two flips and stays zero. The observable parity is 1 because the error flips the terminal measurement of wire 1. If the derivation disagrees with the assertion, fix the assertion to the derivation and say so in the commit message.

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_memory_circuit_execution.py -q"
```

Expected: collection error, `ModuleNotFoundError: No module named 'flagquantum.qec.circuit'` if Tasks 1–3 are not yet done; otherwise failures inside `_detector_bits` or the equality assertion.

- [ ] **Step 3: Make the acceptance test pass**

No production code is expected to change. If the instruction-equality assertion fails, the cause is in `flagquantum/qec/circuit.py`'s source generation, not in the frozen profile: the frozen `_memory_source` is the reference. Note that the captured function name does not affect lowered instructions but does change `program_identity`, so compare `instructions`, never `program_identity`.

If the detector-signature assertions fail, re-derive the expected pattern from the circuit by hand, print the actual pattern with the measurement rows, and fix `_detector_layout` or the recorded convention in `docs/development/API_CHANGE_PROPOSAL_048_QEC_DETECTOR_ERROR_MODEL.md`. Do not weaken the assertion to "some detector fires".

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec -q && black --check tests/qec/test_memory_circuit_execution.py && ruff check tests/qec/test_memory_circuit_execution.py"
```

Expected: the whole `tests/qec` directory passes, including the 82 pre-existing tests.

Every assertion in this file, every detector count, and both injected-error signatures were verified against real execution before this plan was written. If an assertion fails, the change under test has diverged from the plan — investigate rather than adjusting the expectation.

- [ ] **Step 5: Commit**

```bash
git add tests/qec/test_memory_circuit_execution.py
git commit -m "test: pin the code-driven circuit against the frozen d=3 profile"
```

---

### Task 5: Publish the layer and describe its boundary

**Files:**
- Modify: `flagquantum/qec/__init__.py`
- Modify: `flagquantum/qec/IMPLEMENTATION.md`
- Test: `tests/qec/test_codes.py` (append one import test) and `tests/qec/test_memory_circuit.py` (append one import test)

**Interfaces:**
- Consumes: every name produced by Tasks 1–3.
- Produces: `flagquantum.qec.Pauli`, `.CodeCheck`, `.StabilizerCode`, `.RepetitionCode`, `.Detector`, `.DetectorLayout`, `.LogicalObservable`, `.MeasurementRef`, `.MemoryCircuit`, `.ObservableLayout`, `.build_memory_circuit`.

- [ ] **Step 1: Write the failing test**

Append to `tests/qec/test_memory_circuit.py`:

```python
def test_public_namespace_publishes_the_code_independent_layer() -> None:
    import flagquantum.qec as qec

    expected = (
        "CodeCheck",
        "Detector",
        "DetectorLayout",
        "LogicalObservable",
        "MeasurementRef",
        "MemoryCircuit",
        "ObservableLayout",
        "Pauli",
        "RepetitionCode",
        "StabilizerCode",
        "build_memory_circuit",
    )
    missing = [name for name in expected if not hasattr(qec, name)]
    assert not missing, f"flagquantum.qec is missing {missing}"
    for name in expected:
        assert name in qec.__all__, f"{name} is not in flagquantum.qec.__all__"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec/test_memory_circuit.py::test_public_namespace_publishes_the_code_independent_layer -q"
```

Expected: FAIL, `flagquantum.qec is missing ['CodeCheck', 'Detector', ...]`.

- [ ] **Step 3: Publish the names**

Edit `flagquantum/qec/__init__.py`. Add these imports after the existing `from .decoders import (...)` block and before `from .noise import ...`:

```python
from .circuit import (
    Detector,
    DetectorLayout,
    LogicalObservable,
    MeasurementRef,
    MemoryCircuit,
    ObservableLayout,
    build_memory_circuit,
)
from .codes import CodeCheck, RepetitionCode, StabilizerCode
```

Add these imports after the existing `from .noise import ...` line:

```python
from .pauli import Pauli
```

Extend `__all__` so it contains exactly the existing entries plus the eleven new names, keeping the existing relative order and inserting the new names in place alphabetically:

```
"CodeCheck",
"Detector",
"DetectorLayout",
"LogicalObservable",
"MeasurementRef",
"MemoryCircuit",
"ObservableLayout",
"Pauli",
"RepetitionCode",
"StabilizerCode",
"build_memory_circuit",
```

Do not remove, rename, or reorder any existing entry. Do not add the domain to the root `flagquantum` namespace.

- [ ] **Step 4: Describe the layer in `IMPLEMENTATION.md`**

Append a section to `flagquantum/qec/IMPLEMENTATION.md`:

```markdown
## Code-independent records

`pauli.py`, `codes.py`, and `circuit.py` add a code-independent layer beside the
frozen repetition profile. A `Pauli` is a phase-free operator over arbitrary wire
indices; a `StabilizerCode` is a value that declares its distance, wire layout,
checks, stabilizers, and logical observables; `build_memory_circuit` turns a code
and a round count into circuit source plus a detector layout and an observable
layout.

Detector semantics are fixed. A detector is a measurement parity that is
deterministic in the noiseless circuit. A `rounds`-round distance-`d` memory
experiment declares `(d - 1) * (rounds + 1)` detectors: one per check per round
comparing that round against its predecessor, where the first round is compared
against the known all-zero prior state, plus one per check comparing the final
syndrome round against the terminal data readout. Logical failure is the parity
of a declared logical observable.

This layer is additive. The frozen repetition types, the two decoder protocols,
three reference decoders, and both existing workflows keep their current
signatures and behavior, and the seven names pinned by
`contracts/hybrid-compilation-private-v0-candidate.json` are unchanged. The
`RepetitionCode` at `distance=3` reproduces the frozen circuit instruction for
instruction, which is asserted by
`tests/qec/test_memory_circuit_execution.py`.

The frozen profile defines logical failure as the majority of its three
frame-corrected data bits. That is a profile-specific decision rule rather than a
linear logical observable, so general-layer failure counts are not claimed to
equal frozen-profile failure counts at `distance=3`. This layer does not change
the frozen profile's arithmetic.

The layer declares codes and detectors only. It does not build a detector error
model, decode, sample evidence, or make any threshold, logical-suppression,
real-time, or fault-tolerance claim.
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python -m pytest tests/qec -q && black --check flagquantum/qec tests/qec && ruff check flagquantum/qec tests/qec && mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum"
```

Expected: all tests pass and each gate prints a success line.

- [ ] **Step 6: Commit**

```bash
git add flagquantum/qec/__init__.py flagquantum/qec/IMPLEMENTATION.md tests/qec/test_memory_circuit.py
git commit -m "feat: publish the code-independent QEC records layer"
```

---

### Task 6: Repository gates

**Files:**
- Modify: only if a gate reports a real defect in this change. Do not weaken a gate, relax a threshold, or edit `contracts/coverage-policy.toml` to make a failure disappear.

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: nothing. This task is the merge gate.

- [ ] **Step 1: Run the structural gates**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:local bash -lc "python -m pip install -e . --no-deps -q && python tools/check_architecture.py && python tools/check_repository_hygiene.py && python tools/check_repository_language.py && python -m tools.docs_source_of_truth --check && python -m tools.operator_manifest --check && python -m tools.correctness_certification --check && python tools/check_legacy_root_api_usage.py"
```

Expected: every command exits 0 with no error list. `docs_source_of_truth` and `public_api_snapshot` are the two most likely to react to the new exports; if either reports a required regeneration, read `tools/docs_source_of_truth.py`'s own instructions and regenerate rather than hand-editing a generated file.

- [ ] **Step 2: Run the coverage channel and record the real numbers**

```bash
docker run --rm -v $PWD:/w -v /Users/baai/Documents/liuwei/FlagQuantum-upstream/.git:/Users/baai/Documents/liuwei/FlagQuantum-upstream/.git -w /w flagquantum-dev:viz bash -lc "python -m pip install -e . --no-deps -q && python -m pytest -m 'smoke or unit or integration or jax' --cov=flagquantum --cov-report=xml -q && python tools/check_coverage.py"
```

Expected: the global floor is 75 and `packages.qec` has a floor of 82. Report the measured global rate and the measured `qec` rate verbatim in the task summary. If `packages.qec` falls below 82, add tests to the new modules rather than lowering the floor. This run is long; if it does not finish within the command timeout, report that it did not finish and give the per-file numbers from `python -m pytest tests/qec --cov=flagquantum.qec --cov-report=term-missing -q` instead, clearly labeled as a partial measurement.

- [ ] **Step 3: Confirm the frozen surface is untouched**

```bash
git diff --stat origin/main -- flagquantum/qec/types.py flagquantum/qec/decoders.py flagquantum/qec/repetition.py flagquantum/qec/noise.py flagquantum/noise flagquantum/compiler flagquantum/runtime contracts capability-maturity.toml docs/generated
```

Expected: no output. Any output means the change touched a frozen or out-of-scope file; revert that part before proceeding.

- [ ] **Step 4: Commit any gate-driven fixes**

```bash
git add -A
git commit -m "chore: satisfy the repository gates for the QEC records layer"
```

Skip this commit when Step 1 and Step 2 produced no changes.

---

## Self-Review

**Spec coverage** (against the Stage 1 subsection and Acceptance list of `docs/development/API_CHANGE_PROPOSAL_048_QEC_DETECTOR_ERROR_MODEL.md`):

| Spec requirement | Task |
| --- | --- |
| `pauli.py`: sparse `Pauli` with support, weight, `commutes_with`, composition, deterministic text | Task 1 |
| `codes.py`: `StabilizerCode` protocol with distance, counts, wires, checks, stabilizers, logical observables | Task 2 |
| `codes.py`: `RepetitionCode(distance)` | Task 2 |
| `circuit.py`: `build_memory_circuit` returning source, `DetectorLayout`, `ObservableLayout` | Task 3 |
| `DetectorLayout` is a dense ordered tuple of detectors with round/check identity and composing parities | Task 3 |
| `ObservableLayout` is a dense ordered tuple of observables, each a `Pauli` plus a readout parity | Task 3 |
| Detector count is `(d - 1) * (rounds + 1)`, round zero compared against the all-zero prior, plus a final boundary | Task 3 (count, structure) and Task 4 (semantics) |
| `RepetitionCode(distance=3)` reproduces the frozen circuit instruction for instruction | Task 4 |
| `distance=5` and `distance=7` build, lower, and execute with no compiler or runtime change | Task 4 |
| Failure defined by a declared logical observable, with the frozen majority convention documented as non-equivalent | Task 2 (Z-type logical observable), Task 4 (observable parity), Task 5 (`IMPLEMENTATION.md`) |
| Everything additive; the seven pinned names unchanged | Task 5 Step 3, Task 6 Step 3 |
| No threshold, logical-suppression, real-time, or hardware claim | Task 5 Step 4 boundary paragraph; no task introduces one |

Deliberately out of scope for Stage 1, owned by later stages: `dem.py`, decoder and matching modules, `PhenomenologicalNoise`, `statistics.py`, the `pymatching` extra, the capability row, and the executable example. Stages 2–5 each get their own plan once the preceding stage lands, because the DEM's construction details depend on the detector conventions this stage fixes in code rather than in prose.

**Placeholder scan:** no `TBD`, `TODO`, or "handle edge cases" steps. Every code step carries complete file content. The one instruction that defers to judgment — Task 4 Step 3 — states the exact cause to investigate and forbids weakening the assertion.

**Type consistency:** `Pauli(x_wires, z_wires)`, `CodeCheck(index, stabilizer, ancilla_wire, cnot_wires)`, `MeasurementRef(round_index, wire)`, `Detector(index, parity)`, `DetectorLayout(detectors)`, `LogicalObservable(index, pauli, measurement_parity)`, `ObservableLayout(observables)`, `MemoryCircuit(code, rounds, source, detectors, observables)`, and `build_memory_circuit(code, *, rounds)` are used with the same names, field order, and types in every task that touches them. `StabilizerCode`, `RepetitionCode`, `Pauli`, `Detector`, `DetectorLayout`, `LogicalObservable`, `MeasurementRef`, `MemoryCircuit`, `ObservableLayout`, `build_memory_circuit`, and `CodeCheck` are the exact spellings published in Task 5.
