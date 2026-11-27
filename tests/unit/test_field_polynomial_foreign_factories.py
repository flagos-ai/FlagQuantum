"""Actual polynomial factory graphs, packed views and native execution."""

import builtins
from collections import Counter
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
    field_polynomial_component_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.register_source_graph import source_line_qubit_index
from tests.unit.test_binary_polynomial import execute_sparse
from tests.unit.test_field_multiplication import assert_gate_streams

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("constant", (False, True))
@pytest.mark.parametrize("reversed_binding", (False, True))
def test_actual_polynomial_factories_original_directions_sparse_complex_and_resources(
    constant: bool, reversed_binding: bool
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.gf_poly_arithmetic import gf2_poly_add, gf2_poly_add_k
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = (
        gf2_poly_add_k._gf2_poly_4_8_add_k.make()
        if constant
        else gf2_poly_add._gf2_poly_4_8_add.make()
    )
    emitter = import_register_emitter(source)
    logical = 15 if constant else 30
    wires = tuple(range(logical))
    if reversed_binding:
        wires = wires[::-1]
    ports = partition_register(wires, emitter.layout)
    assert emitter.requirements() == (0, 0)
    assert get_cost_value(source, QubitCount()) == logical
    n = logical + 1
    # Every packed source word appears with both spectator values and two
    # arbitrary target words. This sparse subspace avoids a 31-qubit allocation.
    labels = torch.arange((1 << 15) * (2 if constant else 4))
    value = labels >> (1 if constant else 2)
    spectator = labels & 1
    target = ((value * 19 + 7) ^ ((labels >> 1 & 1) * 32767)) & 32767
    mask = sum(v << (3 * (4 - i)) for i, v in enumerate((1, 2, 3, 4, 5)))

    def pack(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        result = spectator.clone()
        values = {"f_x": left, "g_x": right}
        for name, word in ports.items():
            for bit, wire in enumerate(word):
                result |= ((values[name] >> (14 - bit)) & 1) << (n - wire - 1)
        return result

    initial = pack(value, target)
    expected = pack(value ^ mask, target) if constant else pack(value, value ^ target)
    amplitudes = torch.complex(labels.double() + 1, labels.remainder(31).double() - 15)
    amplitudes /= amplitudes.norm()
    for adjoint, direction in enumerate((source, source.adjoint())):
        raw, returned = (
            direction.decompose_bloq()
            .flatten()
            .to_cirq_circuit_and_quregs(
                qubit_manager=None,
                **{
                    name: np.array([cirq.LineQubit(w) for w in word])
                    for name, word in ports.items()
                },
            )
        )
        assert {
            name: tuple(source_line_qubit_index(q) for q in word.flat)
            for name, word in returned.items()
        } == ports
        independent = []
        for op in raw.all_operations():
            assert op.gate == (cirq.X if constant else cirq.CNOT)
            independent.append(
                (
                    "x" if constant else "cx",
                    tuple(source_line_qubit_index(q) for q in op.qubits),
                )
            )
        gates = emitter(ports, (), (), bool(adjoint))
        assert_gate_streams([(g.name, g.wires) for g in gates], independent)
        assert emitter.binding_swaps[adjoint] == 0
        model = get_cost_value(direction, QECGatesCost())
        assert model.clifford == (7 if constant else 15)
        assert model.total_t_count() == 0 and model.measurement == 0
        assert Counter(g.name for g in gates) == {
            "x" if constant else "cx": model.clifford
        }
        artifact = CircuitIR.from_json(CircuitIR(n, gates).to_json())
        actual = execute_sparse(artifact.instructions, initial, n)
        assert torch.equal(actual, expected)
        torch.testing.assert_close(
            amplitudes[actual.argsort()], amplitudes[expected.argsort()], atol=0, rtol=0
        )
        assert torch.equal(execute_sparse(artifact.instructions, actual, n), initial)
    # Independently check the packed arithmetic against the original value path.
    import galois

    dtype = source.qgf_poly
    for left, right in ((0, 32767), (1234, 23456), (32767, 1)):
        f = galois.Poly(
            dtype.qgf.gf_type([(left >> (3 * i)) & 7 for i in reversed(range(5))])
        )
        args = {"f_x": f}
        if not constant:
            args["g_x"] = galois.Poly(
                dtype.qgf.gf_type([(right >> (3 * i)) & 7 for i in reversed(range(5))])
            )
        result = source.on_classical_vals(**args)
        output = result["f_x" if constant else "g_x"]
        assert isinstance(output, galois.Poly)
        packed = sum(
            int(v) << (3 * (4 - i))
            for i, v in enumerate(dtype.to_gf_coefficients(output))
        )
        assert packed == (left ^ mask if constant else left ^ right)


@pytest.mark.parametrize("constant", (False, True))
def test_native_polynomial_append_after_sdk_removal(
    constant: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran._infra.data_types import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic import gf2_poly_add, gf2_poly_add_k

    # The constant case uses the exact 15-bit factory. Register addition uses
    # the same complete source file at degree zero for a dense all-pairs check.
    source = (
        gf2_poly_add_k._gf2_poly_4_8_add_k.make()
        if constant
        else gf2_poly_add.GF2PolyAdd(QGFPoly(0, QGF(2, 3)))
    )
    emitter = import_register_emitter(source)
    logical = sum(width for _, width in emitter.layout)
    ports = partition_register(tuple(range(logical)), emitter.layout)
    forward, backward = (emitter(ports, (), (), a) for a in (False, True))
    n = logical + 1
    labels = torch.arange(1 << n)
    initial = torch.complex(labels.double() + 1, labels.remainder(17).double() - 8)
    initial /= initial.norm()
    destinations = execute_sparse(forward, labels, n)
    expected = torch.zeros_like(initial)
    expected[destinations] = initial
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError(f"Captured polynomial operation imported {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        artifact = CircuitIR.from_json(CircuitIR(n, forward).to_json())
        circuit = Circuit(n, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(n, artifact.instructions)
        )
        torch.testing.assert_close(
            circuit.state().flatten(), expected, atol=2e-12, rtol=0
        )
        circuit.append(FieldOperation(n, backward))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, atol=2e-12, rtol=0
        )


@pytest.mark.parametrize("join", (False, True))
@pytest.mark.parametrize("reversed_binding", (False, True))
def test_actual_coefficient_view_factory_typed_aliases_and_complex_append(
    join: bool, reversed_binding: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.gf_poly_arithmetic import gf_poly_split_and_join as original
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = (original._gf_poly_join if join else original._gf_poly_split).make()
    field = FiniteFieldEncoding(2, 3)
    dtype = FiniteFieldPolynomialEncoding(4, field).register_type()
    native = field_polynomial_component_program(
        FieldPolynomialLayout(4, 3),
        field.register_type(),
        reverse=join,
        packed_dtype=dtype,
    )
    source_regs = tuple(source.signature)
    for owned, foreign in zip(
        native.functions[0].registers or (), source_regs, strict=True
    ):
        assert (owned.name, owned.bitsize, owned.shape, owned.side.name) == (
            foreign.name,
            foreign.bitsize,
            foreign.shape,
            foreign.side.name,
        )
    wires = tuple(range(15))
    if reversed_binding:
        wires = wires[::-1]
    array = np.array([cirq.LineQubit(w) for w in wires]).reshape(
        (5, 3) if join else (15,)
    )
    operation, returned = source.as_cirq_op(cirq.SimpleQubitManager(), array)
    assert operation is None
    assert returned["reg"].shape == ((15,) if join else (5, 3))
    expected = tuple(source_line_qubit_index(q) for q in returned["reg"].flat)
    inputs = (
        tuple(tuple(wires[i : i + 3]) for i in range(0, 15, 3)) if join else (wires,)
    )
    bound = bind_register_program(native, inputs)
    assert tuple(w for word in bound.outputs for w in word) == expected == wires
    assert (
        not bound.circuit.instructions
        and not bound.events
        and not bound.classical_outputs
    )
    assert get_cost_value(source, QubitCount()) == 15
    assert get_cost_value(source, QECGatesCost()).total_t_count() == 0
    labels = torch.arange(1 << 16, dtype=torch.float64)
    state = torch.complex(labels + 1, labels.remainder(13) - 6)
    state /= state.norm()
    # The captured native Program contains only owned metadata and aliases.
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError(f"Polynomial view imported {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        actual = Circuit(16, inputs=state, dtype=state.dtype).append(
            FieldOperation(16, bound.circuit.instructions)
        )
        torch.testing.assert_close(actual.state().flatten(), state, atol=2e-12, rtol=0)


def test_complete_polynomial_factory_inventory_original_codec_and_symbolic_boundaries() -> (
    None
):
    from qualtran._infra.bloq import DecomposeTypeError
    from qualtran._infra.bloq_example import BloqExample
    from qualtran.bloqs.gf_poly_arithmetic import (
        gf2_poly_add,
        gf2_poly_add_k,
        gf_poly_split_and_join,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.serialization.bloq import bloqs_to_proto

    modules = (gf2_poly_add, gf2_poly_add_k, gf_poly_split_and_join)
    inventory = {
        name: example
        for module in modules
        for name, example in vars(module).items()
        if isinstance(example, BloqExample)
    }
    assert set(inventory) == {
        "_gf2_poly_4_8_add",
        "_gf2_poly_add_symbolic",
        "_gf2_poly_4_8_add_k",
        "_gf2_poly_add_k_symbolic",
        "_gf_poly_split",
        "_gf_poly_join",
    }
    for name, example in inventory.items():
        source = example.make()
        with pytest.raises(TypeError, match="QGF"):
            bloqs_to_proto(source, max_depth=0)
        if name.endswith("symbolic"):
            with pytest.raises(DecomposeTypeError, match="symbolic"):
                source.decompose_bloq()
            with pytest.raises(ValueError, match="concrete positive shapes"):
                import_register_emitter(source)
            width = source.qgf_poly.qgf.bitsize
            model = get_cost_value(source, QECGatesCost())
            assert model.clifford == (source.qgf_poly.degree + 1) * width
            assert model.total_t_count() == 0
        elif name in {"_gf_poly_split", "_gf_poly_join"}:
            with pytest.raises(DecomposeTypeError, match="atomic"):
                source.decompose_bloq()
            with pytest.raises(ValueError, match="requires through ports"):
                import_register_emitter(source)
