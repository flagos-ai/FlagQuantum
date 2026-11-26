"""Proposal 129 extends only structured Pauli SELECT target admission."""

import pytest

from flagquantum.core.ir import CircuitIR, Instruction, IRValidationError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("words", (("X", "-"), ("--", ""), ()))
def test_malformed_empty_target_words_stay_invalid(words: tuple[str, ...]) -> None:
    with pytest.raises(IRValidationError):
        Instruction(
            "select_pauli_lcu",
            (0,),
            {
                "control_width": 0,
                "selection_width": 1,
                "target_width": 0,
                "select_unitaries": words,
                "control_val": None,
            },
        )


def test_zero_wire_and_majorana_rules_remain_rejected() -> None:
    with pytest.raises(IRValidationError):
        Instruction(
            "select_pauli_lcu",
            (),
            {
                "control_width": 0,
                "selection_width": 0,
                "target_width": 0,
                "select_unitaries": ("-",),
                "control_val": None,
            },
        )
    with pytest.raises(IRValidationError):
        Instruction(
            "selected_majorana_fermion",
            (0,),
            {
                "control_width": 0,
                "selection_width": 1,
                "target_width": 0,
                "target_gate": "X",
            },
        )


@pytest.mark.parametrize("wires", ((0,), (0, 0)))
def test_empty_target_wrong_cardinality_and_duplicate_wires_are_refused(
    wires: tuple[int, ...],
) -> None:
    with pytest.raises(IRValidationError):
        Instruction(
            "select_pauli_lcu",
            wires,
            {
                "control_width": 1,
                "selection_width": 1,
                "target_width": 0,
                "select_unitaries": ("", "-"),
                "control_val": 1,
            },
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("target_width", -1),
        ("target_width", False),
        ("selection_width", True),
        ("control_val", 2),
    ),
)
def test_other_invalid_metadata_is_not_admitted(field: str, value: int | bool) -> None:
    params = {
        "control_width": 0,
        "selection_width": 1,
        "target_width": 0,
        "select_unitaries": ("", "-"),
        "control_val": None,
    }
    params[field] = value
    with pytest.raises(IRValidationError):
        Instruction("select_pauli_lcu", (0,), params)


@pytest.mark.parametrize("control", (None, 0, 1))
def test_empty_target_instruction_and_existing_json_profile(
    control: int | None,
) -> None:
    width = 1 + int(control is not None)
    operation = Instruction(
        "select_pauli_lcu",
        tuple(range(width)),
        {
            "control_width": int(control is not None),
            "selection_width": 1,
            "target_width": 0,
            "select_unitaries": ("", "-"),
            "control_val": control,
        },
    )
    ir = CircuitIR(width, (operation,))
    assert CircuitIR.from_json(ir.to_json()).instructions == (operation,)
