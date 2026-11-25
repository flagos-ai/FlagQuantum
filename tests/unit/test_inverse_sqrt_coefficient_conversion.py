"""Complete supplied-coefficient conversion uses the unchanged source codec."""

from itertools import product
from typing import Any

import numpy as np
import pytest

from flagquantum.ecosystem.inverse_sqrt_coefficients import (
    inverse_sqrt_coefficient_qrom_data,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "selection,target",
    ((2, 0), (2, 1), (3, 8), (4, 32), (3, 63), (2, 64), (2, 128), (0, 8), (1, 8)),
)
@pytest.mark.parametrize(
    "coefficients",
    (
        (
            (
                0.99994132489119882162,
                0.49609891915903542303,
                0.33261112772430493331,
                0.14876762006038398086,
            ),
            (
                0.81648515205385221995,
                0.27136515484240234115,
                0.12756148214815175348,
                0.044753028579153842218,
            ),
        ),
        ((1.0, -0.5, 1.25, 0.0), (-1.0, 0.25, 2.0, 0.75)),
        ((0.5, 0.25), (0.125,)),
        ((), ()),
        ((0.5,) * 5, (0.25,) * 5),
    ),
)
def test_original_arbitrary_coefficients_and_failure_boundaries(
    selection: int,
    target: int,
    coefficients: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        build_qrom_data_for_poly_fit,
    )

    arrays = tuple(np.array(bank) for bank in coefficients)
    try:
        original = build_qrom_data_for_poly_fit(selection, target, arrays)
    except (IndexError, ValueError, OverflowError, TypeError) as error:
        with pytest.raises(type(error)):
            inverse_sqrt_coefficient_qrom_data(selection, target, coefficients)
    else:
        assert inverse_sqrt_coefficient_qrom_data(
            selection, target, coefficients
        ) == tuple(tuple(int(word) for word in row) for row in original)


def test_generic_helper_reuses_default_tables_without_replacing_bounded_owner() -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        get_inverse_square_root_poly_coeffs,
    )

    from flagquantum.algorithms.primitives.grid_inverse_sqrt import (
        _HIGH,
        _LOW,
        grid_inverse_sqrt_qrom_data,
    )

    low, high = get_inverse_square_root_poly_coeffs()
    assert tuple(low) == _LOW and tuple(high) == _HIGH
    coefficients = (tuple(float(x) for x in low), tuple(float(x) for x in high))
    for n, p in ((2, 4), (3, 32), (6, 15), (7, 12), (8, 30), (5, 55)):
        assert inverse_sqrt_coefficient_qrom_data(
            n, p, coefficients
        ) == grid_inverse_sqrt_qrom_data(n, p)


def test_conversion_does_not_import_reference_and_missing_codec_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    import flagquantum.ecosystem.fixed_point as codec

    original = builtins.__import__

    def no_reference(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_reference)
    data = inverse_sqrt_coefficient_qrom_data(2, 2, ((0.5,), (0.25,)))
    assert data == ((0, 2, 1, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))

    def missing(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(codec, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match="fxpmath"):
        inverse_sqrt_coefficient_qrom_data(2, 2, ((0.5,), (0.25,)))


def test_supplied_coefficient_table_append_matches_source_gates_all_complex_branches() -> (
    None
):
    import torch
    from qualtran.bloqs.data_loading import QROM

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.lookup.qrom_component import qrom_program
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    data = inverse_sqrt_coefficient_qrom_data(
        2, 1, ((0.75, 0.5, 0.25, 0.0), (0.5, 0.25, 0.75, 0.5))
    )
    table = QromTable.from_data(*data, target_bitsizes=(1,) * 4)
    ports = ((0, 1), (2,), (3,), (4,), (5,))
    bound = bind_register_program(
        qrom_program(table, reserve_workspace=True), ports, clean=(7,)
    )
    reference = QROM(tuple(np.array(row) for row in data), (2,), (1,) * 4, ((),) * 4)
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            reference,
            dict(
                zip(
                    ("selection", "target0_", "target1_", "target2_", "target3_"),
                    ports,
                    strict=True,
                )
            ),
            (7,),
            MeasurementSlots(bound.circuit.instructions),
            cirq_style=True,
        ),
    )
    raw = torch.arange(1, 129, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::2] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(64):
        selection, targets = word >> 4, word & 15
        mask = sum(data[i][selection] << (3 - i) for i in range(4))
        destination = selection << 4 | (targets ^ mask)
        for spectator in (0, 1):
            expected[((destination << 1) | spectator) << 1] = amplitudes[
                (word << 1) | spectator
            ]
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(8, bound.circuit.instructions)
    )
    measurements = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=measurements):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, expected / 2 ** (measurements / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch[1::2]) == 0
