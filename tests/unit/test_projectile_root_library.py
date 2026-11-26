"""Literal projectile graph views retain source controls, tails and owned work."""

from importlib import import_module
from itertools import product

import pytest
import torch

from flagquantum.algorithms.first_quantization import (
    ControlledMultiplexedCSwap3D,
    MultiplexedCSwap3D,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.layout.momentum_exchange_component import (
    momentum_exchange_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._projectile_component_library import (
    projectile_component_factories,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_unary_component_calls import source_slot_names

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


def test_original_empty_control_array_executes_uncontrolled_prefix_with_live_tails() -> (
    None
):
    source = import_module(
        _BASE + "projectile.select_and_prepare"
    ).ControlledMultiplexedCSwap3D(1, 2, 2, ())
    program = resolve_component_program(
        _library(source, 0), 0, projectile_component_factories()
    )
    costs = import_module("qualtran.resource_counting")
    with pytest.raises(ValueError, match="Cannot apply_along_axis"):
        costs.get_cost_value(source, costs.QECGatesCost())
    assert _named_inputs(_native_trace(program)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    ports = ((0,), *((q,) for q in range(1, 7)), (7, 8), (9, 10), (11, 12))
    bound = bind_register_program(program, ports)
    reference = MultiplexedCSwap3D.make_on(
        selection=(0,),
        targets=(((1,), (2,), (3,)), ((4,), (5,), (6,))),
        junk=((7,), (9,), (11,)),
    )
    assert_source_dependencies(bound.circuit.instructions, reference.emit())
    assert not bound.events
    assert not any(set(op.wires) & {8, 10, 12} for op in bound.circuit.instructions)
    indices = torch.arange(1 << 13, dtype=torch.int64)
    permutation = indices.clone()
    for row in range(2):
        active = ((indices >> 12) & 1) == row
        for axis in range(3):
            a, b = 1 << (12 - (1 + row * 3 + axis)), 1 << (12 - (7 + axis * 2))
            different = ((indices & a) != 0) ^ ((indices & b) != 0)
            permutation[active & different] ^= a | b
    state = torch.complex(indices.double() + 1, 2 - 3 * indices.double())
    state /= state.norm()
    expected = torch.zeros_like(state)
    expected[permutation] = state
    torch.testing.assert_close(
        forced_branch(state, bound.circuit.instructions, ()),
        expected,
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("select", (False, True))
@pytest.mark.parametrize("example", (False, True))
def test_complete_root_metadata_graph_and_opaque_execution_boundary(
    select: bool, example: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    source = import_module(_BASE + "projectile.select_and_prepare")
    if example:
        original = getattr(
            source, "_sel_first_quant" if select else "_prep_first_quant"
        ).make()
    else:
        original = getattr(
            source,
            (
                "SelectFirstQuantizationWithProj"
                if select
                else "PrepareFirstQuantizationWithProj"
            ),
        )(2, 3, 2, 3, 3, m_param=4, num_bits_nuc_pos=8, num_bits_t=2, num_bits_rot_aa=3)
    expected = _named_inputs(_source_trace(SourceCalls, original.decompose_bloq()))
    baseline = None
    for depth in (0, 1, 2):
        library = _library(original, depth)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            program = resolve_component_program(
                library, 0, projectile_component_factories()
            )
            assert verify_program_record(program) == ()
            assert (
                program.functions[0].registers
                == library.entries[0].declaration.registers
            )
            assert _named_inputs(_native_trace(program)) == expected
            if not select:
                root = program.functions[0]
                offset = 0
                for register in root.registers or ():
                    if register.name == "plus_t":
                        body = root.blocks[0]
                        assert body.return_values is not None
                        assert body.return_values[offset] == body.arguments[offset]
                        break
                    offset += sum(1 for _ in register.all_idxs())
            with pytest.raises(
                ValueError, match="physical lowering requires component bodies"
            ):
                bind_register_program(program, (), clean=tuple(range(100)))
        if baseline is None:
            baseline = program
        else:
            assert baseline == program


@pytest.mark.parametrize("cvs", ((), (0,), (1,), (0, 1), (True,)))
def test_source_control_labels_do_not_change_the_omitted_child_argument(
    cvs: tuple[int, ...],
) -> None:
    source = import_module(
        _BASE + "projectile.select_and_prepare"
    ).ControlledMultiplexedCSwap3D(1, 2, 2, cvs)
    program = resolve_component_program(
        _library(source, 0), 0, projectile_component_factories()
    )
    positive = import_module(
        _BASE + "projectile.select_and_prepare"
    ).ControlledMultiplexedCSwap3D(1, 2, 2, (1,) * len(cvs))
    assert program == resolve_component_program(
        _library(positive, 0), 0, projectile_component_factories()
    )


@pytest.mark.parametrize("direction", (False, True))
def test_term_and_select_leaf_identifiers_keep_original_direction_and_signatures(
    direction: bool,
) -> None:
    modules = (
        import_module(_BASE + "projectile.select_and_prepare"),
        import_module(_BASE + "select_and_prepare"),
        import_module(_BASE + "projectile.select_t"),
        import_module(_BASE + "projectile.select_uv"),
    )
    sources = (
        modules[0].PrepareTUVSuperpositions(3, 2, 3, is_adjoint=direction),
        modules[1].UniformSuperpostionIJFirstQuantization(2, 3),
        modules[2].SelectTFirstQuantizationWithProj(3, 2),
        modules[3].SelectUVFirstQuantizationWithProj(2, 3, 2, 3, 8),
    )
    for source in sources:
        library = _library(source, 0)
        record = resolve_component_program(library, 0, projectile_component_factories())
        assert record.functions[0].registers == library.entries[0].declaration.registers
        assert not record.functions[0].blocks


@pytest.mark.parametrize(
    "eta,p,n,cvs",
    (
        (2, 1, 1, None),
        (3, 1, 1, None),
        (2, 1, 2, (0,)),
        (2, 1, 2, (1, 0)),
        (2, 2, 3, (0,)),
    ),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_source_shaped_exchange_graph_stream_all_histories_and_suffixes(
    eta: int,
    p: int,
    n: int,
    cvs: tuple[int, ...] | None,
    adjoint: bool,
) -> None:
    controlled = cvs is not None
    source = (
        import_module(
            _BASE + "projectile.select_and_prepare"
        ).ControlledMultiplexedCSwap3D(p, n, eta, cvs)
        if controlled
        else import_module(_BASE + "select_and_prepare").MultiplexedCSwap3D(p, eta)
    )
    registers = _library(source, 0).entries[0].declaration.registers
    assert registers is not None
    program = momentum_exchange_program(
        Signature(registers), controlled=controlled, adjoint=adjoint
    )
    assert verify_program_record(program) == ()
    if not adjoint:
        assert _named_inputs(_native_trace(program)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )
    count = len(cvs) if cvs is not None else 0
    address_width = (eta - 1).bit_length()
    logical = count + address_width + 3 * (eta * p + n)
    work_size = address_width + count - 1
    wires = iter(range(logical + work_size))

    def take(width: int) -> tuple[int, ...]:
        return tuple(next(wires) for _ in range(width))

    controls = take(count)
    selection = take(address_width)
    targets = tuple(tuple(take(p) for _ in range(3)) for _ in range(eta))
    junk = tuple(take(n) for _ in range(3))
    work = take(work_size)
    ports = (
        tuple((q,) for q in controls)
        + (selection,)
        + tuple(word for row in targets for word in row)
        + junk
    )
    bound = bind_register_program(program, ports, clean=work)
    assert bound.outputs == ports
    operation: MultiplexedCSwap3D | ControlledMultiplexedCSwap3D
    if cvs is None:
        operation = MultiplexedCSwap3D.make_on(
            selection=selection,
            targets=targets,
            junk=junk,
            workspace=work,
            classical_bits=tuple(range(work_size)),
            adjoint=adjoint,
        )
    else:
        operation = ControlledMultiplexedCSwap3D.make_on(
            controls=controls,
            cvs=cvs,
            selection=selection,
            targets=targets,
            junk=junk,
            workspace=work,
            classical_bits=tuple(range(work_size)),
            adjoint=adjoint,
        )
    assert_source_dependencies(
        source_slot_names(bound.circuit.instructions, work), operation.emit()
    )
    tails = {q for word in junk for q in word[p:]}
    assert not any(tails.intersection(op.wires) for op in bound.circuit.instructions)
    if logical > 15:
        return
    indices = torch.arange(1 << logical, dtype=torch.int64)
    control_mask = sum(1 << (logical - q - 1) for q in controls)
    address = sum(
        ((indices >> (logical - q - 1)) & 1) << (address_width - i - 1)
        for i, q in enumerate(selection)
    )
    ket = torch.complex(indices.double() + 1, 3 - 2 * indices.double())
    ket[address >= eta] = 0
    ket /= ket.norm()
    permutation = indices.clone()
    for row in range(eta):
        active = (address == row) & ((indices & control_mask) == control_mask)
        for axis in range(3):
            for left, right in zip(targets[row][axis], junk[axis][:p], strict=True):
                a, b = 1 << (logical - left - 1), 1 << (logical - right - 1)
                different = ((indices & a) != 0) ^ ((indices & b) != 0)
                permutation[active & different] ^= a | b
    initial = torch.zeros(1 << (logical + work_size), dtype=torch.complex128)
    initial[:: 1 << work_size] = ket
    expected = torch.zeros_like(initial)
    expected[permutation << work_size] = ket
    measurements = sum(op.name == "measure" for op in bound.circuit.instructions)
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** (measurements / 2),
            atol=1e-11,
            rtol=0,
        )
