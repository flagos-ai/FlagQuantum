"""Contract fake before integrating guarded classical parity consumers."""

from itertools import product

import pytest

pytestmark = pytest.mark.unit


def _reference_assignment(
    values: list[int],
    sources: tuple[int, ...],
    target: int,
    constant: bool,
    conditions: tuple[tuple[int, int], ...],
) -> list[int]:
    """Independent whole-record fake with old-read and inactive-preserve semantics."""
    old = values.copy()
    result = values.copy()
    if all(old[slot] == expected for slot, expected in conditions):
        result[target] = (sum(old[source] for source in sources) + int(constant)) % 2
    return result


def test_contract_fake_exhausts_alias_and_inactive_truth_tables() -> None:
    for bits, target, constant in product(
        product((0, 1), repeat=3), range(3), (False, True)
    ):
        for sources in ((), (0,), (1, 2), (0, 1, 0), (target, target, 2)):
            conditions = ((0, 1), (1, 0))
            result = _reference_assignment(
                list(bits), sources, target, constant, conditions
            )
            assert result[target] == (
                (sum(bits[source] for source in sources) + constant) % 2
                if bits[:2] == (1, 0)
                else bits[target]
            )
            assert all(
                result[slot] == bits[slot] for slot in range(3) if slot != target
            )


@pytest.mark.ftqc_reference
def test_locked_native_guard_snapshots_match_independent_classical_fake() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"} and hasattr(km, "Simulator")
    text = "\n".join(
        (
            "PUSH_CONDITION if b0",
            "PUSH_CONDITION if b1",
            "BIT_INVERT b0 if b2",
            "BIT_INVERT b3",
            "POP_CONDITION",
            "BIT_INVERT b3",
            "POP_CONDITION",
        )
    )
    for bits in product((0, 1), repeat=4):
        sim = km.Simulator(batch_size=1, seed=31)
        for slot, value in enumerate(bits):
            sim.write_across_shots(km.b(slot), value)
        sim.do(km.Circuit(text))
        expected = list(bits)
        outer, inner = bits[0], bits[1]
        expected = (
            _reference_assignment(expected, (0, 2), 0, False, ())
            if outer and inner
            else expected
        )
        expected = (
            _reference_assignment(expected, (3,), 3, True, ())
            if outer and inner
            else expected
        )
        expected = (
            _reference_assignment(expected, (3,), 3, True, ()) if outer else expected
        )
        assert [
            sim.read_across_shots(km.b(slot), out=int) for slot in range(4)
        ] == expected
