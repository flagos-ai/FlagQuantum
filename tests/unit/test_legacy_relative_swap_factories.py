"""Relative exchange constructors preserve source phases and inverse calls."""

from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_relative_swap_factories import (
    relative_swap_factories,
)
from flagquantum.compiler.registers.bits.fanout_component import (
    fanout_component_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def pinned_relative_swap() -> None:
    import kickmix
    from qualtran.bloqs.swap_network import cswap_approx

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for filename, digest in (
        ("cswap_approx.py", "362798e240461e6fa6269af8820e9436e5645b4a"),
        ("cswap_approx_test.py", "665e1cb122308cb86d7494728ae85d540275806a"),
    ):
        raw = (Path(cswap_approx.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize(
    "example", ("_approx_cswap_symb", "_approx_cswap_small", "_approx_cswap_large")
)
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_original_factories_and_exact_adjoint(
    example: str, inverse: bool
) -> None:
    from qualtran.bloqs.swap_network import cswap_approx

    forward = getattr(cswap_approx, example).make()
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, relative_swap_factories()
    )
    assert verify_program_record(record) == ()
    deeper = resolve_component_program(
        _library(source, 1), 0, relative_swap_factories()
    )
    assert matches_component_definition(record, "main", deeper)
    if example == "_approx_cswap_symb":
        assert not record.functions[0].blocks
    else:
        assert record.functions[0].blocks
        compare_layers(record, source, atomic_leaves(record))


@pytest.mark.parametrize("width", (1, 2, 3, 8))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_primitive_dependencies_full_phase_and_clean_append(
    width: int, inverse: bool
) -> None:
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    source = CSwapApprox(width).adjoint() if inverse else CSwapApprox(width)
    record = resolve_component_program(
        _library(source, 0), 0, relative_swap_factories()
    )
    compare_layers(record, source, atomic_leaves(record))
    inputs = ((0,), tuple(range(1, width + 1)), tuple(range(width + 1, 2 * width + 1)))
    bound = bind_register_program(record, inputs)
    original = source_relative_swap(0, inputs[1], inputs[2])
    if inverse:
        # The independent source call comparison above verifies generic
        # Adjoint structure; these measurement-free gate inverses check its
        # primitive action without losing the source S-dagger phases.
        inverses = {
            "cx": "cx",
            "h": "h",
            "s": "sdg",
            "sdg": "s",
            "t": "tdg",
            "tdg": "t",
        }
        original = tuple(
            replace(gate, name=inverses[gate.name]) for gate in reversed(original)
        )
    assert_source(bound.circuit.instructions, original)
    assert not bound.events and bound.outputs == inputs
    logical = 2 * width + 1
    if width <= 3:
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical),
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=3e-11,
            rtol=0,
        )
    else:
        # One additional occupied spectator remains entangled with both words.
        index = torch.arange(1 << (logical + 1), dtype=torch.float64)
        state = torch.complex(index + 1, (index % 11 - 4) * index)
        state /= state.norm()
        torch.testing.assert_close(
            forced_branch(state, bound.circuit.instructions, ()),
            forced_branch(state, original, ()),
            atol=3e-11,
            rtol=0,
        )
    appended = (
        Circuit(logical)
        .append(_BoundBlockOperation(bound.circuit.instructions, logical))
        .to_ir()
    )
    assert appended.instructions == bound.circuit.instructions
    assert appended.from_json(appended.to_json()) == appended


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
def test_exact_fanout_inverse_source_order(width: int) -> None:
    from qualtran.bloqs.mcmt.multi_target_cnot import MultiTargetCNOT

    source = MultiTargetCNOT(width).adjoint()
    record = fanout_component_program(width, adjoint=True)
    compare_layers(record, source, atomic_leaves(record))
    inputs = ((width,), tuple(reversed(range(width))))
    bound = bind_register_program(record, inputs)
    original = reference_arithmetic_gates(
        source,
        dict(zip(("control", "targets"), inputs, strict=True)),
        (),
        {},
    )
    assert_source(bound.circuit.instructions, original)
    assert not bound.events and bound.outputs == inputs


def test_concrete_and_symbolic_resolution_does_not_import_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import sympy
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    for width in (2, sympy.Symbol("n")):
        for inverse in (False, True):
            forward = CSwapApprox(width)
            library = _library(forward.adjoint() if inverse else forward, 0)
            with monkeypatch.context() as patch:
                patch.setitem(sys.modules, "qualtran", None)
                record = resolve_component_program(
                    library, 0, relative_swap_factories()
                )
            assert verify_program_record(record) == ()
            assert bool(record.functions[0].blocks) == (width == 2)
