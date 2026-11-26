"""Original RSA constructor graphs retain terminal failure and reused arithmetic."""

import sys
from importlib import import_module

import pytest
import sympy
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
    modular_multiplication_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_definition_match import (
    _named_definitions,
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._rsa_component_library import rsa_component_factories
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_RSA = "qualtran.bloqs.cryptography.rsa.rsa_phase_estimate"


@pytest.mark.parametrize("profile", (0, 1, 2, 3))
def test_original_rsa_root_connections_constants_and_factory_replacement(
    profile: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module(_RSA)
    source = (
        module.RSAPhaseEstimate(2, 3, 2)
        if profile == 0
        else (
            module.RSAPhaseEstimate(4, 15, 2)
            if profile == 1
            else (
                module._rsa_pe_small.make()
                if profile == 2
                else module.RSAPhaseEstimate(4, sympy.Symbol("p"), sympy.Symbol("g"))
            )
        )
    )
    expected = _named_inputs(_source_trace(SourceCalls, source.decompose_bloq()))
    children = [
        instance.bloq
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(instance.bloq).__name__ == "CModMulK"
    ]
    baseline = None
    checked_children: set[object] = set()
    for depth in (0, 1):
        library = _library(source, depth)
        encoded = serialize_component_library(library)
        assert (
            serialize_component_library(deserialize_component_library(encoded))
            == encoded
        )
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            program = resolve_component_program(library, 0, rsa_component_factories())
        assert verify_program_record(program) == ()
        root = program.functions[0]
        assert root.registers == () and root.blocks[0].return_values == ()
        assert len(root.blocks[0].operations) == 4 * source.n + 3
        assert _named_inputs(_native_trace(program)) == expected
        assert root.blocks[0].operations[-1].callee is not None
        free = next(
            f
            for f in program.functions
            if f.name == root.blocks[0].operations[-1].callee
        )
        assert free.blocks[0].operations[0].workspace_dirty
        operations = root.blocks[0].operations[2 * source.n + 1 : 4 * source.n + 1]
        for call, child in zip(operations, children, strict=True):
            if child in checked_children:
                continue
            direct = resolve_component_program(
                _library(child, 0), 0, modular_multiplication_factories()
            )
            assert call.callee is not None and matches_component_definition(
                program, call.callee, direct
            )
            checked_children.add(child)
        for i, child in enumerate(children):
            for j, other in enumerate(children):
                assert (operations[i].callee == operations[j].callee) == (
                    child == other
                )
        with pytest.raises(
            ValueError, match="physical lowering requires component bodies"
        ):
            bind_register_program(program, (), clean=tuple(range(100)))
        if baseline is None:
            baseline = program
        else:
            assert baseline == program


@pytest.mark.parametrize("width", (0, 1, 2))
def test_original_terminal_shallow_declaration_and_exact_deeper_failure(
    width: int,
) -> None:
    source_ir = import_module("qualtran")
    source = import_module("qualtran.bloqs.cryptography._factoring_shims").MeasureQFT(
        width
    )
    library = _library(source, 0)
    program = resolve_component_program(library, 0, rsa_component_factories())
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    assert verify_program_record(program) == ()
    message = "does not accept Soquets" if width == 0 else "not used"
    with pytest.raises(source_ir.BloqError, match=message):
        source.decompose_bloq()
    for depth in (1, 2):
        with pytest.raises(source_ir.BloqError, match=message):
            _library(source, depth)


def test_symbolic_source_metadata_and_concrete_deeper_refusal() -> None:
    source_ir = import_module("qualtran")
    source = import_module(_RSA)._rsa_pe.make()
    for depth in (0, 1, 2):
        program = resolve_component_program(
            _library(source, depth), 0, rsa_component_factories()
        )
        assert program.functions[0].registers == ()
        assert not program.functions[0].blocks
    with pytest.raises(source_ir.DecomposeTypeError, match="symbolic"):
        source.decompose_bloq()
    concrete = import_module(_RSA).RSAPhaseEstimate(2, 3, 2)
    with pytest.raises(source_ir.BloqError, match="not used"):
        _library(concrete, 2)


def test_selected_multiplier_source_primitives_complex_subspace_and_all_histories() -> (
    None
):
    from tests.unit.measurement_certificate import certify_erasure_histories
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_default_linear_combination_oracles import bind
    from tests.unit.test_modular_multiplication_component import source_stream
    from tests.unit.test_register_subtraction import assert_source

    source = import_module(_RSA).RSAPhaseEstimate(2, 3, 2)
    program = resolve_component_program(
        _library(source, 0), 0, rsa_component_factories()
    )
    name = program.functions[0].blocks[0].operations[5].callee
    assert name is not None
    record = _rename_program(ProgramRecord(_named_definitions(program, name)), "main")
    bound, logical = bind(record)
    [child] = [
        i.bloq
        for i, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(i.bloq).__name__ == "CModMulK" and i.bloq.k == 2
    ]
    assert_source(
        bound.circuit.instructions,
        source_stream(child, bound, {"ctrl": (0,), "x": (1, 2)}),
    )
    events = certify_erasure_histories(
        bound.circuit.instructions,
        n_wires=bound.circuit.n_wires,
        zero_wires=tuple(range(logical, bound.circuit.n_wires)),
        check_clean_output=False,
    )
    superposition = torch.tensor(
        [1 + 2j, -2 + 0.5j, 3 - 1j, -1 - 0.2j, 2 + 3j, -0.5 + 1j],
        dtype=torch.complex128,
    )
    superposition /= torch.linalg.vector_norm(superposition)
    for coefficients in (
        *torch.unbind(torch.eye(6, dtype=torch.complex128)),
        superposition,
    ):
        initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        expected = torch.zeros_like(initial)
        for control in range(2):
            for word in range(3):
                index = 3 * control + word
                initial[
                    ((control << 2) | word) << (bound.circuit.n_wires - logical)
                ] = coefficients[index]
                values = child.on_classical_vals(ctrl=control, x=word)
                output = 0
                for wires, value in zip(
                    bound.outputs, (values["ctrl"], values["x"]), strict=True
                ):
                    for bit, wire in enumerate(reversed(wires)):
                        output |= ((int(value) >> bit) & 1) << (
                            bound.circuit.n_wires - 1 - wire
                        )
                expected[output] = coefficients[index]
        actual = forced_branch(
            initial, bound.circuit.instructions, (0,) * events
        ) * 2 ** (events / 2)
        torch.testing.assert_close(actual, expected, atol=8e-12, rtol=0)
