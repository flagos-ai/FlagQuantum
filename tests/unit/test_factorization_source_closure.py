"""Complete factorization constructors retain literal source adjoints and models."""

from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import sympy

from flagquantum.algorithms.primitives.factorization_source_formula import (
    source_double_factorization_one_body_formula,
    source_single_factorization_block_formula,
    source_single_factorization_one_body_formula,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.component_graph import (
    _elements,
    iter_component_calls,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._double_factorization_library import (
    double_factorization_factories,
)
from flagquantum.ecosystem._single_factorization_library import (
    single_factorization_factories,
)
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_momentum_source_declarations import frozen_source as frozen_source
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
B = sympy.Symbol("b", positive=True, integer=True)


@pytest.fixture(scope="module", autouse=True)
def frozen_factorization(frozen_source: None) -> None:
    for family, name, blob in (
        ("df", "double_factorization", "e83b385240a5e98b8ab939f6e7eef0fff1547e5c"),
        ("sf", "single_factorization", "5e1104df8ac20e98c1e263a83214c7d1ec72a51f"),
    ):
        module = import_module(f"qualtran.bloqs.chemistry.{family}.{name}")
        assert module.__file__ is not None
        raw = Path(module.__file__).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def original(family: str, block: bool, field: str | None = None) -> Any:
    from attrs import evolve

    module = import_module(
        f"qualtran.bloqs.chemistry.{family}."
        + ("double" if family == "df" else "single")
        + "_factorization"
    )
    result = getattr(
        module, f"_{family}_" + ("block_encoding" if block else "one_body")
    ).make()
    return result if field is None else evolve(result, **{field: B})


def assert_returns(record: ProgramRecord, graph: Any) -> None:
    from qualtran import LeftDangle, RightDangle

    expected = {
        (edge.right.reg.name, edge.right.idx): (
            "entry" if edge.left.binst == LeftDangle else str(edge.left.binst.i),
            edge.left.reg.name,
            edge.left.idx,
        )
        for edge in graph.connections
        if edge.right.binst == RightDangle
    }
    root = record.functions[0]
    assert root.registers is not None
    ports = [(reg.name, idx) for reg in root.registers for idx in reg.all_idxs()]
    body = root.blocks[0]
    producers = {
        value: ("entry", port, idx)
        for value, (port, idx) in zip(body.arguments, ports, strict=True)
    }
    for position, _, outputs in iter_component_calls(record):
        for binding in _elements(outputs):
            producers[binding.value] = (
                str(position),
                binding.register.name,
                binding.index,
            )
    assert body.return_values is not None
    assert {
        port: producers[value]
        for port, value in zip(ports, body.return_values, strict=True)
    } == expected


@pytest.mark.parametrize("family", ("sf", "df"))
@pytest.mark.parametrize("block", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_root_directions_match_source_calls_and_boundary_returns(
    family: str, block: bool, inverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = original(family, block)
    if inverse:
        source = source.adjoint()
    graph = source.decompose_bloq()
    expected = _named_inputs(_source_trace(SourceCalls, graph))
    owners = (
        single_factorization_factories
        if family == "sf"
        else double_factorization_factories
    )
    for depth in (0, 1):
        library = _library(source, depth)
        import sys

        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            result = resolve_component_program(library, 0, owners())
            assert verify_program_record(result) == ()
            assert _named_inputs(_native_trace(result)) == expected
            with pytest.raises(
                ValueError, match="physical lowering requires component bodies"
            ):
                bind_register_program(result, (), clean=tuple(range(128)))
        assert_returns(result, graph)


@pytest.mark.parametrize(
    "family,block,field",
    (
        ("sf", False, "num_bits_state_prep"),
        ("sf", False, "num_bits_rot_aa"),
        ("sf", True, "num_bits_state_prep"),
        ("sf", True, "num_bits_rot_aa_inner"),
        ("sf", True, "num_bits_rot_aa_outer"),
        ("df", False, "num_bits_state_prep"),
        ("df", False, "num_bits_rot"),
        ("df", False, "num_bits_rot_aa"),
        ("df", True, "num_bits_state_prep"),
        ("df", True, "num_bits_rot"),
        ("df", True, "num_bits_rot_aa_inner"),
        ("df", True, "num_bits_rot_aa_outer"),
    ),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_source_success_symbolic_precision_is_not_rejected_as_physical_allocation(
    family: str, block: bool, field: str, inverse: bool
) -> None:
    source = original(family, block, field)
    if inverse:
        source = source.adjoint()
    owners = (
        single_factorization_factories
        if family == "sf"
        else double_factorization_factories
    )
    result = resolve_component_program(_library(source, 0), 0, owners())
    graph = source.decompose_bloq()
    assert verify_program_record(result) == ()
    assert _named_inputs(_native_trace(result)) == _named_inputs(
        _source_trace(SourceCalls, graph)
    )
    assert_returns(result, graph)


@pytest.mark.parametrize("alias,rotation", ((12, 8), (B, 8), (12, B), (B + 1, B)))
@pytest.mark.parametrize("blocks", (((1, 1), (1, 1)), ((4, 8), (2, 4))))
def test_single_body_complete_symbolic_categories_and_peak(
    alias: Any, rotation: Any, blocks: tuple[tuple[int, int], tuple[int, int]]
) -> None:
    from qualtran.bloqs.chemistry.sf.single_factorization import (
        SingleFactorizationOneBody,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    forward, inverse = blocks
    source = SingleFactorizationOneBody(
        50,
        10,
        alias,
        rotation,
        kp1=forward[0],
        kp2=forward[1],
        kp1_inv=inverse[0],
        kp2_inv=inverse[1],
    )
    result = source_single_factorization_one_body_formula(
        50, 10, alias, rotation, forward_blocks=forward, inverse_blocks=inverse
    )
    expected = get_cost_value(source, QECGatesCost()).asdict()
    for key in expected:
        assert sympy.simplify(getattr(result, key) - expected[key]) == 0
    assert (
        sympy.simplify(result.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )


@pytest.mark.parametrize("family", ("sf", "df"))
def test_remaining_successful_symbolic_rotation_models(family: str) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = original(
        family,
        family == "sf",
        "num_bits_rot_aa_inner" if family == "sf" else "num_bits_rot_aa",
    )
    if family == "df":
        result = source_double_factorization_one_body_formula(
            source.num_spin_orb,
            source.num_eig,
            source.num_bits_state_prep,
            source.num_bits_rot,
            B,
        )
    else:
        result = source_single_factorization_block_formula(
            source.num_aux,
            source.num_spin_orb,
            source.num_bits_state_prep,
            B,
            forward_blocks=(source.kp1, source.kp2),
            inverse_blocks=(source.kp1_inv, source.kp2_inv),
        )
    expected = get_cost_value(source, QECGatesCost()).asdict()
    for key in expected:
        assert sympy.simplify(getattr(result, key) - expected[key]) == 0
    assert (
        sympy.simplify(result.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )


@pytest.mark.parametrize(
    "family,block,field",
    (
        ("sf", False, "num_aux"),
        ("sf", True, "num_aux"),
        ("df", False, "num_bits_rot_aa"),
        ("df", True, "num_bits_rot_aa_inner"),
        ("df", True, "num_aux"),
    ),
)
def test_original_failed_zero_port_body_remains_a_native_header(
    family: str, block: bool, field: str
) -> None:
    from attrs import evolve
    from qualtran import BloqError

    source = evolve(original(family, block), **{field: 0})
    with pytest.raises(BloqError, match="does not accept Soquets"):
        source.decompose_bloq()
    owners = (
        single_factorization_factories
        if family == "sf"
        else double_factorization_factories
    )
    result = resolve_component_program(_library(source, 0), 0, owners())
    assert isinstance(result, ProgramRecord)
    assert verify_program_record(result) == ()
    assert not result.functions[0].blocks
