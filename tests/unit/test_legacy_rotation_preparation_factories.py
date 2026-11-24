"""Source rotation constructors retain exact calls, complex action and cleanup."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_rotation_preparation_factories import (
    rotation_preparation_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.phase.scaled_phase_component import (
    phase_add_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def pinned_rotation() -> None:
    import kickmix
    from qualtran.bloqs.state_preparation import state_preparation_via_rotation

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        (
            "state_preparation_via_rotation.py",
            "2466a11c3303e23c066602dce535750e650f83bf",
        ),
        (
            "state_preparation_via_rotation_test.py",
            "6b7a770579188189c032c55a9c6bf5636314df13",
        ),
    ):
        raw = (Path(state_preparation_via_rotation.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def _terminals(record: ProgramRecord, width: int) -> set[str]:
    result = atomic_leaves(record)
    templates = tuple(
        phase_add_component_program(width, width, sign=sign) for sign in (1, -1)
    )
    for function in record.functions[1:]:
        if tuple(reg.name for reg in function.registers or ()) != ("x", "phase_grad"):
            continue
        if any(
            matches_component_definition(
                record,
                function.name,
                template,
            )
            for template in templates
        ):
            result.add(function.name)
    return result


@pytest.mark.parametrize("inverse", (False, True))
def test_original_concrete_example_and_all_prga_children(inverse: bool) -> None:
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        _state_prep_via_rotation,
    )

    original = _state_prep_via_rotation.make()
    for forward in (
        original,
        *original.prga_prepare_amplitude,
        original.prga_prepare_phases,
    ):
        source = forward.adjoint() if inverse else forward
        record = resolve_component_program(
            _library(source, 0), 0, rotation_preparation_factories()
        )
        assert verify_program_record(record) == ()
        deeper = resolve_component_program(
            _library(source, 1), 0, rotation_preparation_factories()
        )
        assert verify_program_record(deeper) == ()
        assert matches_component_definition(record, "main", deeper)
        compare_layers(record, source, _terminals(record, 2))


@pytest.mark.parametrize("controls", (0, 1, 2))
@pytest.mark.parametrize("uncompute", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_full_small_complex_operator_exact_adjoint_and_all_history_cleanup(
    controls: int, uncompute: bool, inverse: bool
) -> None:
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        StatePreparationViaRotations,
    )

    forward = StatePreparationViaRotations((0.6, 0.8j), 2, controls, uncompute)
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, rotation_preparation_factories()
    )
    compare_layers(record, source, _terminals(record, 2))
    cursor = 0
    inputs = []
    for reg in record.functions[0].registers or ():
        width = int(reg.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(cursor, cursor + 12))
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, cursor), expected, atol=3e-11, rtol=0
    )
    assert bound.outputs == tuple(inputs)
    assert bound.circuit.from_json(bound.circuit.to_json()) == bound.circuit


def test_symbolic_phase_width_now_resolves_and_source_import_is_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        _state_prep_via_rotation,
        _state_prep_via_rotation_symb_phasegrad,
    )

    symbolic = _library(_state_prep_via_rotation_symb_phasegrad.make(), 0)
    concrete = _library(_state_prep_via_rotation.make(), 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        declared = resolve_component_program(
            symbolic, 0, rotation_preparation_factories()
        )
        executable = resolve_component_program(
            concrete, 0, rotation_preparation_factories()
        )
    assert verify_program_record(declared) == verify_program_record(executable) == ()
    assert not declared.functions[0].blocks
    assert executable.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(declared, ())


@pytest.mark.parametrize("inverse", (False, True))
def test_eight_bit_table_complex_kickback_all_histories_and_entangled_spectator(
    inverse: bool,
) -> None:
    from cmath import exp
    from math import tau

    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        PRGAViaPhaseGradient,
    )

    from tests.unit.measurement_certificate import certify_erasure_histories
    from tests.unit.test_rotation_preparation import execute_rotation_sparse

    words = (8, 51, 153, 231)
    forward = PRGAViaPhaseGradient(2, 8, words, 1)
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, rotation_preparation_factories()
    )
    compare_layers(record, source, _terminals(record, 8))
    bound = bind_register_program(
        record, ((0,), (1, 2), tuple(range(3, 11))), clean=tuple(range(12, 80))
    )
    n = bound.circuit.n_wires
    events = certify_erasure_histories(
        bound.circuit.instructions, n_wires=n, zero_wires=tuple(range(12, n))
    )
    assert events == sum(gate.name == "measure" for gate in bound.circuit.instructions)
    norm = (
        256
        * sum(
            abs(complex(address + 1, control - 2)) ** 2
            for address in range(4)
            for control in (0, 1)
        )
    ) ** 0.5
    initial = {}
    expected = {}
    for control in (0, 1):
        for address in range(4):
            coefficient = complex(address + 1, control - 2) / norm
            phase = (
                exp((-1 if inverse else 1) * 1j * tau * words[address] / 256)
                if control
                else 1
            )
            for gradient in range(256):
                index = (((control << 10) | (address << 8) | gradient) << (n - 11)) | (
                    ((control + address) % 2) << (n - 12)
                )
                initial[index] = coefficient * exp(-1j * tau * gradient / 256)
                expected[index] = initial[index] * phase
    for seed in (3, 17):
        actual = execute_rotation_sparse(bound.circuit.instructions, n, initial, seed)
        assert (
            max(
                abs(actual.get(i, 0j) - expected.get(i, 0j))
                for i in actual.keys() | expected.keys()
            )
            < 5e-11
        )
