"""Locked dense rotation factories, state grids, and phase-gradient kickback."""

import ast
from cmath import exp
from hashlib import sha1
from importlib.metadata import version
from math import pi, sqrt, tau
from pathlib import Path
from typing import Any

import pytest
import torch
from sympy import simplify

from examples.algorithms.rotation_preparation_catalog import (
    ORIGINAL_COEFFICIENTS,
    RotationPreparationExample,
)
from flagquantum.algorithms.primitives.symbolic_rotation_resource_model import (
    source_rotation_preparation_resource_formula,
)
from flagquantum.algorithms.primitives.symbolic_rotation_resource_model import (
    source_rotation_preparation_signature as rotation_signature,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.phase.phase_gradient_lookup import (
    phase_gradient_lookup_ir,
    phase_gradient_lookup_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_phase_gradient_lookup import source_lookup_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_rotation_preparation import (
    execute_rotation_sparse,
    source_rotation_preparation_gates,
)
from tests.unit.test_rotation_resource_model import assert_source_cost

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def locked_rotation() -> tuple[Any, ast.Module]:
    import kickmix
    from qualtran.bloqs.state_preparation import (
        state_preparation_via_rotation as source,
    )

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(source.__file__).parent
    for name, digest in (
        (
            "state_preparation_via_rotation.ipynb",
            "ee713c6e8b5f077f72c2aa056fdf91deddf42d47",
        ),
        (
            "state_preparation_via_rotation.py",
            "2466a11c3303e23c066602dce535750e650f83bf",
        ),
        (
            "state_preparation_via_rotation_test.py",
            "6b7a770579188189c032c55a9c6bf5636314df13",
        ),
        (
            "state_preparation_via_rotation_tutorial.ipynb",
            "834025f31ea5b74b47a4113712fd722fd71a03af",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return source, ast.parse(
        (folder / "state_preparation_via_rotation_test.py").read_bytes()
    )


@pytest.mark.parametrize("factory", ("concrete", "shape", "gradient"))
def test_original_rotation_factory(
    locked_rotation: tuple[Any, ast.Module], factory: str
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    src, _ = locked_rotation
    operation = {
        "concrete": src._state_prep_via_rotation,
        "shape": src._state_prep_via_rotation_symb,
        "gradient": src._state_prep_via_rotation_symb_phasegrad,
    }[factory]()
    signature = rotation_signature(operation.n_coeff, operation.phase_bitsize)
    assert [(r.name, r.bitsize, r.side.name) for r in signature] == [
        (r.name, r.bitsize, r.side.name) for r in operation.signature
    ]
    direct = operation.build_call_graph(None)
    rx_count = sum(
        count for child, count in direct.items() if type(child).__name__ == "Rx"
    )
    assert rx_count == 2 * operation.state_bitsize
    x_count = sum(
        count for child, count in direct.items() if type(child).__name__ == "XGate"
    )
    assert x_count == 2
    prga = [
        (child, count)
        for child, count in direct.items()
        if isinstance(child, src.PRGAViaPhaseGradient)
    ]
    if factory == "concrete":
        example = RotationPreparationExample()
        assert example.coefficients == tuple(operation.state_coefficients)
        assert_source_cost(example.resource_model(), operation)
        actual = {
            child: count
            for (_, child), count in example.call_graph().direct_calls.items()
        }
        expected = {"Rx(+π/2)": 3, "Rx(-π/2)": 3, "X": 2}
        expected.update(
            {
                f"PRGA({child.selection_bitsize} selection, {child.phase_bitsize} phase, "
                f"{child.control_bitsize} controls; {child.rom_values})": count
                for child, count in prga
            }
        )
        assert actual == expected
        verify_rotation(src, ORIGINAL_COEFFICIENTS, 2, reverse=True)
    else:
        assert len(prga) == 1 and prga[0][1] == 2
        formula = source_rotation_preparation_resource_formula(
            operation.n_coeff, operation.phase_bitsize
        )
        gates = get_cost_value(operation, QECGatesCost())
        for key in ("and_bloq", "clifford", "measurement", "toffoli", "rotation"):
            assert simplify(getattr(formula, key) - getattr(gates, key)) == 0
        assert (
            simplify(formula.logical_qubits - get_cost_value(operation, QubitCount()))
            == 0
        )
        assert (
            simplify(
                4 * (formula.and_bloq + formula.toffoli)
                - (16 * operation.n_coeff + 8 * operation.phase_bitsize - 32)
            )
            == 0
        )


@pytest.mark.parametrize(
    "name,controls,bases,reverse",
    [
        ("test_exact_state_prep_via_rotation_", 0, (0,), False),
        ("test_state_prep_via_rotation_adjoint", 0, (0,), True),
        ("test_approximate_state_prep_via_rotation", 0, (0,), False),
        (
            "test_controlled_state_preparation_via_rotation_do_not_prepare",
            1,
            (0,),
            False,
        ),
        ("test_state_preparation_via_rotation_superposition_ctrl", 1, (0, 1), False),
        ("test_state_preparation_via_rotation_multi_qubit_ctrl", 2, (0, 3), False),
    ],
)
def test_original_rotation_state_grid(
    locked_rotation: tuple[Any, ast.Module],
    name: str,
    controls: int,
    bases: tuple[int, ...],
    reverse: bool,
) -> None:
    src, tests = locked_rotation
    test = next(
        node
        for node in tests.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    fixtures = next(
        ast.literal_eval(mark.args[1])
        for mark in test.decorator_list
        if isinstance(mark, ast.Call)
        and isinstance(mark.func, ast.Attribute)
        and mark.func.attr == "parametrize"
    )
    for phase, coefficients in fixtures:
        verify_rotation(src, coefficients, phase, controls, bases, reverse)


def verify_rotation(
    src: Any,
    coefficients: tuple[complex, ...],
    phase: int,
    controls: int = 0,
    bases: tuple[int, ...] = (0,),
    reverse: bool = False,
) -> None:
    op = RotationPreparationExample(coefficients, phase, controls)
    data_width = op.gradient[-1] + 1
    gradient = tuple(
        exp(-1j * tau * i / (1 << phase)) / sqrt(1 << phase) for i in range(1 << phase)
    )
    initial = torch.zeros(1 << data_width, dtype=torch.complex128)
    for address in bases:
        start = address << (op.target_width + phase)
        initial[start : start + len(gradient)] = torch.tensor(
            gradient, dtype=torch.complex128
        ) / sqrt(len(bases))
    check_rotation_direction(src, op, initial, reverse)


def check_rotation_direction(
    src: Any, op: RotationPreparationExample, initial: torch.Tensor, reverse: bool
) -> None:
    from dataclasses import replace

    data_width = op.gradient[-1] + 1
    expected = initial
    for direction in (False, True) if reverse else (False,):
        example = replace(op, uncompute=direction)
        source = src.StatePreparationViaRotations(
            example.coefficients, example.phase_width, example.controls, direction
        )
        work, slots = example.preparation.requirements(
            example.controls, uncompute=direction
        )
        circuit = Circuit(example.qubit_count)
        circuit.append(example)
        ports = {
            "target_state": tuple(
                range(example.controls, example.controls + example.target_width)
            ),
            "phase_gradient": example.gradient,
        }
        if example.controls:
            ports["prepare_control"] = tuple(range(example.controls))
        ir = circuit.to_ir()
        expected_gates = source_rotation_preparation_gates(
            source,
            ports,
            tuple(range(data_width, data_width + work)),
            tuple(range(slots)),
        )
        assert_source(ir.instructions, expected_gates)
        assert_source_cost(example.resource_model(), source)
        start = {
            i << work: complex(value)
            for i, value in enumerate(expected)
            if abs(value) > 1e-14
        }
        expected = (
            torch.tensor(source.tensor_contract(), dtype=torch.complex128) @ expected
        )
        desired = {
            i << work: complex(value)
            for i, value in enumerate(expected)
            if abs(value) > 1e-14
        }
        for seed in (0, 1, 27):
            actual = execute_rotation_sparse(ir.instructions, ir.n_wires, start, seed)
            assert (
                max(
                    abs(actual.get(i, 0) - desired.get(i, 0))
                    for i in actual.keys() | desired.keys()
                )
                < 3e-12
            )
    if reverse:
        torch.testing.assert_close(expected, initial, atol=3e-12, rtol=0)


def test_original_prga_kickback(locked_rotation: tuple[Any, ast.Module]) -> None:
    src, _ = locked_rotation
    phase, words = 3, (6, 0, 5, 2)
    source = src.PRGAViaPhaseGradient(2, phase, words, 1)
    work, slots = phase_gradient_lookup_requirements(1, 2, phase, words)
    control, selection, gradient = (0,), (1, 2), (3, 4, 5)
    scratch = tuple(range(6, 6 + work))
    bits = tuple(range(slots))
    gates = phase_gradient_lookup_ir(control, selection, gradient, words, scratch, bits)
    assert_source(
        gates,
        source_lookup_gates(
            source,
            {"control": control, "selection": selection, "phase_gradient": gradient},
            scratch,
            bits,
        ),
    )
    initial = {
        ((address << phase) | g) << work: exp(-1j * tau * g / 8) / 8
        for address in range(8)
        for g in range(8)
    }
    expected = {
        index: value
        * (
            exp(1j * pi * words[(index >> (phase + work)) % 4] / 4)
            if index >> (5 + work)
            else 1
        )
        for index, value in initial.items()
    }
    for seed in (0, 1, 27):
        actual = execute_sparse_register(CircuitIR(6 + work, gates), initial, seed=seed)
        assert (
            max(
                abs(actual.get(i, 0) - expected.get(i, 0))
                for i in actual.keys() | expected.keys()
            )
            < 3e-12
        )


@pytest.mark.parametrize(
    "case", ["one_bit", "uncompute", "shape", "phase", "singleton"]
)
def test_rotation_and_prga_typed_direct_declarations(
    locked_rotation: tuple[Any, ast.Module], case: str
) -> None:
    from qualtran.symbolics import HasLength
    from sympy import Symbol

    from flagquantum.algorithms.primitives.symbolic_rotation_resource_model import (
        source_rotation_preparation_calls,
    )
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
        RotationCall,
    )
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from flagquantum.compiler.registers.phase.phase_gradient_lookup import (
        PhaseGradientLookupCall,
    )
    from flagquantum.core.ir.program.signatures import Signature

    source_module, _ = locked_rotation
    n, p = (Symbol(name, positive=True, integer=True) for name in ("n", "p"))
    coefficients = {
        "one_bit": (1, 0),
        "uncompute": ORIGINAL_COEFFICIENTS,
        "shape": None,
        "phase": (1, 0),
        "singleton": (1,),
    }[case]
    count = n if coefficients is None else len(coefficients)
    phase = p if case in ("shape", "phase") else 2
    controls = 1
    source = source_module.StatePreparationViaRotations(
        HasLength(count) if coefficients is None else coefficients,
        phase,
        controls,
        uncompute=case == "uncompute",
    )

    def assert_ports(native: Signature, source_signature: Any) -> None:
        assert [
            (reg.name, reg.bitsize, reg.side.name, reg.dtype.encoding) for reg in native
        ] == [
            (
                reg.name,
                reg.bitsize,
                reg.side.name,
                {"QBit": "bit", "QAny": "opaque", "QFxp": "fixed_point"}[
                    type(reg.dtype).__name__
                ],
            )
            for reg in source_signature
        ]

    assert_ports(rotation_signature(count, phase, controls), source.signature)
    native = source_rotation_preparation_calls(
        count, phase, controls, coefficients=coefficients, uncompute=case == "uncompute"
    )
    assert len(native) == len(source.build_call_graph(None))
    for child, multiplicity in source.build_call_graph(None).items():
        kind = type(child).__name__
        if kind == "Rx":
            assert native[RotationCall("rx", child.angle, child.eps)] == multiplicity
        elif kind == "XGate":
            assert native[CliffordCall("x", 1)] == multiplicity
        else:
            assert kind == "PRGAViaPhaseGradient"
            qrom = child.qrom_bloq
            table = QromTable(
                qrom.data_shape,
                qrom.selection_bitsizes,
                qrom.target_bitsizes,
                qrom.target_shapes,
                qrom.num_controls,
                tuple(tuple(map(int, data.flat)) for data in qrom.data)
                if qrom.has_data()
                else None,
            )
            lookup = PhaseGradientLookupCall(table)
            assert native[lookup] == multiplicity
            assert_ports(lookup.signature, child.signature)
            assert_ports(lookup.addition_signature, child.add_into_phase_grad.signature)
            for reg, upstream_reg in zip(
                lookup.addition_signature,
                child.add_into_phase_grad.signature,
                strict=True,
            ):
                assert (
                    dict(reg.dtype.parameters)["fractional_width"]
                    == upstream_reg.dtype.num_frac
                )
            assert child.build_call_graph(None) == {
                qrom: 2,
                child.add_into_phase_grad: 1,
            }
            assert lookup.direct_calls() == {
                table: 2,
                ("AddIntoPhaseGrad", phase, phase): 1,
            }
