"""Verify literal THC selection constructors, call wiring and phase dependency."""

import sys
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_signature,
    thc_rotations_signature,
)
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_exchange_factories import exchange_factories
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_multi_control_factories import (
    multi_control_factories,
)
from flagquantum.compiler.formats.legacy_thc_selection_factories import (
    thc_selection_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import MeasurementSlots, atomic_leaves
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.chemistry.thc import select_bloq

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(select_bloq.__file__).parent
    for path, digest in (
        (folder / "select_bloq.py", "a3b825950fc09a429f829637ed14d7c73e922bb6"),
        (folder / "select_bloq_test.py", "f1b2b7a84dc7d831216470214ca02f6b07e02349"),
        (folder.parent / "black_boxes.py", "f9670a2ebaf2fe0973323cf7b532c88142970902"),
    ):
        raw = path.read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return select_bloq


def factories() -> dict[str, ComponentFactory]:
    return {
        **bookkeeping_factories(),
        **fixed_gate_factories(),
        **exchange_factories(),
        **multi_control_factories(),
        **thc_selection_factories(),
    }


@pytest.mark.parametrize("control", (None, 0, 1))
def test_literal_select_exact_inverse_constructor(
    source_module: Any, control: int | None
) -> None:
    source = source_module.SelectTHC(2, 4, 8, 8, control_val=control)
    record = resolve_component_program(_library(source.adjoint(), 0), 0, factories())
    assert verify_program_record(record) == ()
    compare_layers(record, source.adjoint(), atomic_leaves(record))


@pytest.mark.parametrize(
    "values", [(2, 4, 8, 1), (10, 8, 12, 10), (1, 2, 2, 1), (2, 3, 1, 1)]
)
@pytest.mark.parametrize("control", [None, 0, 1])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_complete_select_constructor_and_stored_graph(
    source_module: Any,
    values: tuple[int, int, int, int],
    control: int | None,
    depth: int,
) -> None:
    source = source_module.SelectTHC(*values, kr1=2, kr2=3, control_val=control)
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    program = resolve_component_program(library, 0, factories())
    assert verify_program_record(program) == ()
    assert len(program.functions[0].blocks[0].operations) == 15
    pairs = compare_layers(program, source, atomic_leaves(program))
    rotations = [
        (name, node)
        for name, node in pairs.items()
        if isinstance(node, source_module.THCRotations)
    ]
    assert len(rotations) == 4
    for name, node in rotations:
        assert f"thc_rotation_{int(node.two_body_only)}_{int(node.is_adjoint)}" in name
        assert (
            node.num_mu,
            node.num_spin_orb,
            node.num_bits_theta,
            node.kr1,
            node.kr2,
        ) == (*values[:3], 2, 3)
    if control is not None:
        root = program.functions[0].blocks[0]
        assert root.return_values is not None
        assert root.return_values[0] == root.arguments[0]
    root = program.functions[0].blocks[0]
    ports = program.functions[0].registers or ()
    assert root.return_values is not None
    for i, port in enumerate(ports):
        if port.name in {"sigma", "rot"}:
            assert root.return_values[i] == root.arguments[i]
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(program, ())


def test_actual_original_factory_and_doc(source_module: Any) -> None:
    source = source_module._thc_sel.make()
    assert (
        source.num_mu,
        source.num_spin_orb,
        source.num_bits_theta,
        source.keep_bitsize,
    ) == (10, 8, 12, 10)
    assert source_module._THC_SELECT.examples == (source_module._thc_sel,)
    program = resolve_component_program(_library(source, 0), 0, factories())
    compare_layers(program, source, atomic_leaves(program))


@pytest.mark.parametrize("two,adjoint", tuple(product((False, True), repeat=2)))
def test_rotation_declaration_and_exact_source_inverse_metadata(
    source_module: Any, two: bool, adjoint: bool
) -> None:
    from qualtran import DecomposeNotImplementedError

    source = source_module.THCRotations(2, 4, 8, 2, 3, two, adjoint)
    library = _library(source, 1)
    program = resolve_component_program(library, 0, factories())
    assert not program.functions[0].blocks
    assert program.functions[0].registers == library.entries[0].declaration.registers
    inverse = source.adjoint()
    assert inverse.is_adjoint is not adjoint and inverse.two_body_only is two
    assert str(source) == "In_mu-R" + ("†" if adjoint else "")
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize("values", [(0, 2, 1, 0), (2, 0, 0, 0)])
def test_original_empty_word_failure_retains_logical_declaration(
    source_module: Any, values: tuple[int, int, int, int]
) -> None:
    from qualtran import BloqError

    source = source_module.SelectTHC(*values)
    program = resolve_component_program(_library(source, 0), 0, factories())
    assert not program.functions[0].blocks
    with pytest.raises(BloqError, match="CSwap does not accept"):
        source.decompose_bloq()


def test_native_numerical_validation_is_unchanged() -> None:
    with pytest.raises(ValueError, match="paired spin"):
        select_thc_signature(2, 3, 1)
    with pytest.raises(ValueError, match="two angle bits"):
        thc_rotations_signature(2, 4, 1)


@pytest.mark.parametrize("profile,depth", tuple(product(range(3), (0, 1, 2))))
def test_symbolic_constructor_graphs_and_source_failure(
    source_module: Any, profile: int, depth: int
) -> None:
    from sympy import symbols

    b, n, k = symbols("b n k")
    values = ((2, 4, b, 1), (2, n, b, k), (2, 4, b + 2, k))[profile]
    source = source_module.SelectTHC(*values)
    if profile == 1 and depth == 2:
        with pytest.raises(ValueError, match="Split cannot have a symbolic"):
            _library(source, depth)
        return
    program = resolve_component_program(_library(source, depth), 0, factories())
    assert verify_program_record(program) == ()
    compare_layers(program, source, atomic_leaves(program))
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize("value", [0, 1])
def test_actual_original_control_dispatch(source_module: Any, value: int) -> None:
    from qualtran import CtrlSpec

    source = source_module.SelectTHC(2, 4, 8, 1)
    selected, _ = source.get_ctrl_system(CtrlSpec(cvs=value))
    assert selected.control_val == value
    program = resolve_component_program(_library(selected, 0), 0, factories())
    compare_layers(program, selected, atomic_leaves(program))


def test_invalid_constructor_metadata_does_not_build_a_body() -> None:
    from flagquantum.errors import SerializationError

    root = thc_selection_factories()[
        "qualtran.bloqs.chemistry.thc.select_bloq.SelectTHC"
    ]
    values = {"num_mu": 2, "num_spin_orb": 4, "num_bits_theta": 8, "keep_bitsize": 1}
    with pytest.raises(SerializationError, match="signature"):
        root("main", (), values)
    with pytest.raises(SerializationError, match="control"):
        root("main", None, {**values, "control_val": 2})
    with pytest.raises(SerializationError, match="arguments"):
        root("main", None, {**values, "unknown": 0})


def test_native_constructor_binding_without_optional_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in ("qualtran", "cirq", "quimb", "kickmix", "qiskit"):
        monkeypatch.setitem(sys.modules, module, None)
    root = thc_selection_factories()[
        "qualtran.bloqs.chemistry.thc.select_bloq.SelectTHC"
    ]
    program = root(
        "main",
        None,
        {
            "num_mu": 2,
            "num_spin_orb": 4,
            "num_bits_theta": 8,
            "keep_bitsize": 1,
        },
    )
    from flagquantum.core.ir.program import ProgramRecord

    assert isinstance(program, ProgramRecord) and verify_program_record(program) == ()
    assert len(program.functions[0].blocks[0].operations) == 15
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("values", [(1,), (1, 0), (0, 1, 1)])
def test_direct_controlled_z_word_primitive_stream_and_all_histories(
    source_module: Any, width: int, values: tuple[int, ...]
) -> None:
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs

    source = ApplyControlledZs(values, width)
    program = resolve_component_program(_library(source, 0), 0, factories())
    compare_layers(program, source, atomic_leaves(program))
    n = len(values)
    logical = n + width
    inputs = (*((i,) for i in range(n)), tuple(range(n, logical)))
    clean = tuple(range(logical, logical + n - 1))
    bound = bind_register_program(program, inputs, clean=clean)
    gates = bound.circuit.instructions
    assert_source(
        gates,
        reference_arithmetic_gates(
            source,
            {"ctrls": tuple(range(n)), "system": tuple(range(n, logical))},
            clean,
            MeasurementSlots(gates),
            keep_pauli_leaves=True,
        ),
    )
    count = Counter(g.name for g in gates)["measure"]
    for word in range(1 << logical):
        initial = torch.zeros(1 << (logical + n - 1), dtype=torch.complex128)
        offset = word << (n - 1)
        initial[offset] = 1
        active = all(
            (word >> (logical - i - 1)) & 1 == value for i, value in enumerate(values)
        )
        expected = initial * (-1 if active and ((word >> (width - 1)) & 1) else 1)
        for history in product((0, 1), repeat=count):
            torch.testing.assert_close(
                forced_branch(initial, gates, history),
                expected / 2 ** (count / 2),
                rtol=0,
                atol=2e-12,
            )


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_controlled_z_word_normal_append_and_coherent_spectators(
    source_module: Any, strategy: str
) -> None:
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs

    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    source = ApplyControlledZs((1, 0, 1), 2)
    program = resolve_component_program(_library(source, 0), 0, factories())
    bound = bind_register_program(program, ((0,), (1,), (2,), (3, 4)), clean=(5, 6))
    values = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(values, values.flip(0))
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::4] = amplitudes
    expected = initial.clone()
    for word in range(32):
        if word >> 2 == 5 and (word & 2):
            expected[word << 2] *= -1
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(7, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=32, seed=42, strategy=strategy)
    assert result.statistics["measurement_count"] == 64
    for state in result.final_states:
        torch.testing.assert_close(state.reshape(-1), expected, rtol=0, atol=2e-12)
