"""Reuse complete source control witnesses for typed binary fields and polynomials."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip(
        "NumPy is needed for typed field control fixtures", allow_module_level=True
    )

from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.classical_register_codec import (
    decode_register_bits,
    encode_register_value,
)
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_metadata import resolve_finite_field_encoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from flagquantum.simulation.finite_field_polynomial_metadata import (
    resolve_finite_field_polynomial_encoding,
)
from tests.unit import test_component_control_extension as extension
from tests.unit import test_component_structured_control as structured
from tests.unit.test_register_bindings import SourceContract
from tests.unit.test_register_control_predicate import ControlCase

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES: tuple[ControlCase, ...] = tuple(
    ((kind, 1, shape, (value,), "unsigned"),)
    for kind in ("QGF", "QGFPolyBit")
    for shape in ((), (1,))
    for value in (0, 1)
)
WIDE_CASES: tuple[ControlCase, ...] = (
    (("QGF", 2, (), (1,), "unsigned"),),
    (("QGF", 1, (2,), (0, 1), "unsigned"),),
    (("QGFPolyBit", 2, (), (2,), "unsigned"),),
    (("QGFPoly", 4, (), (6,), "unsigned"),),
    (("QGFPoly", 2, (1, 2), (0, 3), "unsigned"),),
    (("QGF", 1, (), (0,), "unsigned"), ("QGFPolyBit", 1, (), (1,), "unsigned")),
)


def source_dtype(register: Register) -> Any:
    from qualtran import QGF, QGFPoly

    if register.dtype.encoding == "finite_field":
        field = resolve_finite_field_encoding(register.dtype)
        return QGF(2, int(field.degree), field.gf_type.irreducible_poly)
    polynomial = resolve_finite_field_polynomial_encoding(register.dtype)
    field = polynomial.field
    return QGFPoly(
        int(polynomial.degree),
        QGF(2, int(field.degree), field.gf_type.irreducible_poly),
    )


def controls(case: ControlCase, names: tuple[str, ...]) -> tuple[Register, ...]:
    result = []
    for name, (kind, width, shape, _, _) in zip(names, case, strict=True):
        coefficient_width = width if kind == "QGF" else 1 if kind == "QGFPolyBit" else 2
        field = FiniteFieldEncoding(2, coefficient_width)
        dtype = (
            field.register_type()
            if kind == "QGF"
            else FiniteFieldPolynomialEncoding(
                width // coefficient_width - 1, field
            ).register_type()
        )
        result.append(Register(name, dtype, shape))
    return tuple(result)


def fixture(
    case: ControlCase,
) -> tuple[Any, tuple[ControlRegister, ...], tuple[tuple[int, ...], ...]]:
    from qualtran import CtrlSpec

    ports = controls(case, tuple(f"ctrl{i}" for i in range(len(case))))
    types, activations, patterns, groups = [], [], [], []
    cursor = 0
    for port, (_, width, shape, words, _) in zip(ports, case, strict=True):
        scalar = Register(port.name, port.dtype, side=port.side)
        values = np.empty(shape, dtype=object)
        for index, word in zip(port.all_idxs(), words, strict=True):
            bits = [(word >> i) & 1 for i in reversed(range(width))]
            value = decode_register_bits(scalar, np.asarray(bits))
            assert encode_register_value(scalar, value) == bits
            values[index] = value
        types.append(source_dtype(port))
        activations.append(values)
        patterns.append(ControlRegister(width, words))
        groups.append(tuple(range(cursor, cursor + width * len(words))))
        cursor += width * len(words)
    spec = CtrlSpec(tuple(types), tuple(activations))
    assert (
        tuple(bit for pattern in patterns for bit in pattern.bits())
        == tuple(spec.to_cirq_cv().expand())[0]
    )
    return spec, tuple(patterns), tuple(groups)


@pytest.fixture(autouse=True)
def typed_reference_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extend only the local metadata adapter, keeping frozen source untouched."""
    original: Any = SourceContract.register

    def source_register(register: Register) -> Any:
        from qualtran import Register as ForeignRegister
        from qualtran import Side

        if register.dtype.encoding in {"finite_field", "finite_field_polynomial"}:
            return ForeignRegister(
                register.name,
                source_dtype(register),
                register.shape,
                Side[register.side.name],
            )
        return original(register)

    monkeypatch.setattr(SourceContract, "register", staticmethod(source_register))
    for module in (structured, extension):
        monkeypatch.setattr(module, "control_fixture", fixture)
        monkeypatch.setattr(module, "native_controls", controls)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("expanded", [False, True])
def test_complete_source_wrapper_calls_branches_and_resources(
    case: ControlCase, expanded: bool
) -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.component_control import (
        control_component_via_predicate,
    )
    from flagquantum.compiler.registers.component_expansion import (
        expand_component_calls,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program import verify_program_record
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_control import add_h, add_phase, fixture_program
    from tests.unit.test_component_graph import _native_trace, _source, _source_trace
    from tests.unit.test_register_subtraction import assert_source

    record, positive = fixture_program(1)
    spec, patterns, groups = fixture(case)
    source = ControlledViaAnd(positive.subbloq, spec)
    ports = controls(case, tuple(source.ctrl_reg_names))
    module = control_component_via_predicate(
        record, {"h": add_h, "p": add_phase}, ports, patterns
    )
    assert verify_program_record(module) == ()
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(source.signature)
    # Native typed ports require explicit zero-gate Split/Join views. Compare
    # the intervening source calls after resolving these two wire aliases.
    trace: Any = _native_trace(module)
    assert dict(trace[0][1]).keys() == {"reg"}
    assert dict(trace[-1][1]).keys() == {"reg"}

    def resolve(value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], str):
            producer, name, index = value
            if producer == "0":
                return ("entry", ports[0].name, (0,) if ports[0].shape else ())
            if producer != "entry":
                return (str(int(producer) - 1), name, index)
        return (
            tuple(resolve(item) for item in value)
            if isinstance(value, tuple)
            else value
        )

    comparable = [
        (str(i), resolve(inputs), resolve(outputs))
        for i, (_, inputs, outputs) in enumerate(trace[1:-1])
    ]
    assert structured.normalize_trace(comparable) == structured.normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    operations = module.functions[0].blocks[0].operations
    terminals = [
        instance.bloq
        for _, (instance, _, _) in zip(
            operations[1:-1], source.decompose_bloq().iter_bloqsoqs(), strict=True
        )
    ]
    assert Counter(terminals) == source.bloq_counts()
    if case[0][0] == "QGF":
        # Original CtrlSpec stores unhashable galois scalars. Preserve its
        # root-cost failure; the original decomposed children still have costs.
        with pytest.raises(TypeError, match="unhashable"):
            get_cost_value(source, QECGatesCost())
        models = [get_cost_value(child, QECGatesCost()) for child in terminals]
        assert sum(model.measurement for model in models) == 1
        assert sum(model.and_bloq for model in models) == 1
    else:
        model = get_cost_value(source, QECGatesCost())
        assert model.measurement == model.and_bloq == 1
    if expanded:
        module, _ = expand_component_calls(module)
    bound = bind_register_program(module, ((0,), (1,), (2,)), clean=(3,))
    assert bound.outputs == ((0,), (1,), (2,))
    gates = bound.circuit.instructions
    measurements = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    reference = structured.original_gates(
        source,
        {source.ctrl_reg_names[0]: groups[0], "q": (1,), "spectator": (2,)},
        (3,),
        measurements,
    )
    assert_source(gates, reference)
    assert Counter(g.name for g in gates) == Counter(g.name for g in reference)
    state = torch.zeros(16, dtype=torch.complex128)
    state[::2] = torch.tensor(
        [complex(i + 1, i - 2) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    for history in product((0, 1), repeat=1):
        actual = forced_branch(state, gates, history)
        torch.testing.assert_close(
            actual, forced_branch(state, reference, history), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("case", CASES)
def test_existing_append_and_callback_replacement(case: ControlCase) -> None:
    structured.test_existing_append_protocol_and_specialization_replacement(case)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("expanded", [False, True])
def test_exact_adjoint_all_history_pairs(case: ControlCase, expanded: bool) -> None:
    structured.test_exact_adjoint_body_restores_all_forward_inverse_histories(
        case, expanded
    )


@pytest.mark.parametrize("case", WIDE_CASES)
def test_frozen_wide_field_predicate_refusal(case: ControlCase) -> None:
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    from flagquantum.compiler.registers.component_control import (
        control_component_via_predicate,
    )
    from flagquantum.compiler.registers.control.component_predicate import (
        control_predicate_program,
    )
    from tests.unit.test_component_control import add_h, add_phase, fixture_program

    spec, patterns, _ = fixture(case)
    with pytest.raises(NotImplementedError, match="only supports integer types"):
        CtrlSpecAnd(spec)
    ports = controls(case, tuple(f"ctrl{i}" for i in range(len(case))))
    with pytest.raises(ValueError, match="only a single-bit"):
        control_predicate_program(ports, patterns)
    record, positive = fixture_program(1)
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    with pytest.raises(NotImplementedError, match="only supports integer types"):
        ControlledViaAnd(positive.subbloq, spec).decompose_bloq()
    with pytest.raises(ValueError, match="only a single-bit"):
        control_component_via_predicate(
            record, {"h": add_h, "p": add_phase}, ports, patterns
        )
