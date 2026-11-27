"""Whole blocked source metadata and executable extra singleton bank axes."""

from importlib import import_module
from importlib.metadata import version
from itertools import product
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_select_swap_factories import (
    select_swap_factories,
)
from flagquantum.compiler.registers.lookup.qrom_declaration import QromDeclaration
from flagquantum.compiler.registers.lookup.select_swap_callees import (
    SelectSwapRouteCall,
    SelectSwapXorCall,
)
from flagquantum.compiler.registers.lookup.select_swap_declaration import (
    SelectSwapDeclaration,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests._reference_dependencies import unavailable_reference

try:
    import kickmix
    import numpy as np
    import sympy
    from numpy.typing import NDArray

    source_module = import_module("qualtran.bloqs.data_loading.select_swap_qrom")
    source_resources = import_module("qualtran.resource_counting")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def reference_runtime() -> None:
    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__", None) == "0.1" and hasattr(
        kickmix, "Simulator"
    )


def call_key(child: Any) -> tuple[Any, ...]:
    if isinstance(child, QromDeclaration):
        return (
            "QROM",
            child.data_shape,
            child.selection_bitsizes,
            child.target_bitsizes,
            child.target_shapes,
            child.num_controls,
            child.datasets,
        )
    if isinstance(child, SelectSwapRouteCall):
        return "SWZ", child.width, child.selectors, child.banks, child.adjoint
    if isinstance(child, SelectSwapXorCall):
        return "Xor", child.width
    if type(child).__name__ == "QROM":
        return (
            "QROM",
            child.data_shape,
            child.selection_bitsizes,
            child.target_bitsizes,
            child.target_shapes,
            child.num_controls,
            (
                tuple(tuple(map(int, row.flat)) for row in child.data)
                if child.has_data()
                else (None,) * len(child.target_bitsizes)
            ),
        )
    if type(child).__name__ == "SwapWithZero":
        return (
            "SWZ",
            child.target_bitsize,
            child.selection_bitsizes,
            child.n_target_registers,
            child.uncompute,
        )
    assert type(child).__name__ == "Xor", child
    return "Xor", child.dtype.num_qubits


@pytest.mark.parametrize(
    "case",
    (
        "ordinary",
        "extra_zero",
        "extra_one",
        "negative_control",
        "symbolic",
        "fractional",
        "singleton",
        "control_three",
    ),
)
@pytest.mark.parametrize("dirty", (False, True))
def test_complete_declared_blocks_calls_models_and_static_quotes(
    case: str, dirty: bool
) -> None:
    if case == "symbolic":
        n, b, k, c = sympy.symbols("N b k c", positive=True, integer=True)
        source = source_module.SelectSwapQROM.build_from_bitsize(
            n, (b,), log_block_sizes=(k,), num_controls=c, use_dirty_ancilla=dirty
        )
        base = QromDeclaration.from_shape(n, (b,), num_controls=c)
    else:
        data = np.array([0.1, 0.2, 1.1, 1.2]) if case == "fractional" else np.arange(4)
        logs = (
            (0, 0) if case == "extra_zero" else (0, 1) if case == "extra_one" else (1,)
        )
        controls = (
            -1 if case == "negative_control" else 3 if case == "control_three" else 0
        )
        if case == "singleton":
            data = np.array([1])
            logs = (0,)
        selection = ((len(data) - 1).bit_length(),)
        source = source_module.SelectSwapQROM(
            (data,),
            selection,
            (2,),
            num_controls=controls,
            log_block_sizes=logs,
            use_dirty_ancilla=dirty,
        )
        base = QromDeclaration.from_arrays(
            data,
            selection_bitsizes=selection,
            target_bitsizes=(2,),
            num_controls=controls,
        )
    native = SelectSwapDeclaration(base, source.log_block_sizes, dirty)
    assert native.is_symbolic() == source.is_symbolic()
    assert native.source_block_sizes == source.block_sizes
    assert native.source_batch_shape == source.batched_qrom_shape
    assert native.source_high_widths == source.batched_qrom_selection_bitsizes
    expected = {
        call_key(child): count
        for child, count in source.build_call_graph(
            source_resources.SympySymbolAllocator()
        ).items()
    }
    assert {
        call_key(child): count for child, count in native.calls().items()
    } == expected
    for legacy in (False, True):
        expected_model = source_resources.get_cost_value(
            source, source_resources.QECGatesCost(legacy_shims=legacy)
        )
        assert native.gate_cost(legacy_shims=legacy).asdict() == expected_model.asdict()
    assert native.static_qubit_count() == source_resources.get_cost_value(
        source, source_resources.QubitCount()
    )
    if source.has_data() and not source.is_symbolic():
        for actual, original in zip(
            native.batched_arrays, source.batched_data, strict=True
        ):
            np.testing.assert_array_equal(actual, original)
        # A caller can inspect/modify its fresh arrays without changing literals.
        inspected = native.batched_arrays
        inspected[0].flat[0] = 99
        np.testing.assert_array_equal(native.batched_arrays[0], source.batched_data[0])


@pytest.mark.parametrize("extra", (1, 3))
@pytest.mark.parametrize("controls", (0, 1))
@pytest.mark.parametrize("dirty,adjoint", tuple(product((False, True), repeat=2)))
def test_extra_zero_block_axes_preserve_source_calls_primitives_and_all_branches(
    extra: int, controls: int, dirty: bool, adjoint: bool
) -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.core.ir import CircuitIR, Instruction
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_control import compare_layers
    from tests.unit.test_legacy_arithmetic_factories import _library
    from tests.unit.test_multi_control_component import MeasurementSlots, atomic_leaves
    from tests.unit.test_register_subtraction import assert_source
    from tests.unit.test_select_swap_qrom import original_select_swap_gates

    data = np.arange(4)
    prefix = 0
    logs = (0,) * (1 + extra)
    source = source_module.SelectSwapQROM(
        (data,),
        (2,),
        (2,),
        num_controls=controls,
        log_block_sizes=logs,
        use_dirty_ancilla=dirty,
    )
    declaration = SelectSwapDeclaration(
        QromDeclaration.from_arrays(
            data, selection_bitsizes=(2,), target_bitsizes=(2,), num_controls=controls
        ),
        logs,
        dirty,
    )
    root = source.adjoint() if adjoint else source
    record = resolve_component_program(_library(root, 0), 0, select_swap_factories())
    assert verify_program_record(record) == ()
    if not adjoint:
        # The reserved physical QROM frame has extra storage views. Compare the
        # enclosing source calls literally, then its unreserved logical recipe.
        qrom_entries = {
            fn.name
            for fn in record.functions
            if fn.registers == tuple(declaration.qrom.signature)
        }
        assert qrom_entries
        route_entries = {
            fn.name
            for fn in record.functions
            if fn.registers and fn.registers[-1].name == "targets"
        }
        compare_layers(
            record, source, atomic_leaves(record) | qrom_entries | route_entries
        )
        from flagquantum.compiler.registers.lookup.qrom_component import qrom_program

        logical_qrom = qrom_program(declaration.qrom)
        compare_layers(logical_qrom, source.qrom_bloq, atomic_leaves(logical_qrom))
    for depth in (1, 2):
        alternate = resolve_component_program(
            _library(root, depth), 0, select_swap_factories()
        )
        assert verify_program_record(alternate) == ()
    logical = controls + 4
    bank = tuple(range(logical, logical + 2 * (1 << prefix)))
    borrowed = bank if dirty and not adjoint else ()
    work_size = max(0, 2 - prefix + controls - 1)
    work = tuple(range(logical + len(bank), logical + len(bank) + work_size))
    clean = (() if borrowed else bank) + work
    ports = {
        **({"control": tuple(range(controls))} if controls else {}),
        "selection": (controls, controls + 1),
        "target0_": (controls + 2, controls + 3),
    }
    bound = bind_register_program(
        record, tuple(ports.values()), clean=clean, borrowed=borrowed
    )
    reference, _ = original_select_swap_gates(
        source,
        ports,
        (tuple(bank[i : i + 2] for i in range(0, len(bank), 2)),),
        work,
        tuple(range(len(work))),
        adjoint=adjoint,
    )
    slots = MeasurementSlots(bound.circuit.instructions)
    active = {}
    remapped = []
    for gate in reference:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            old = int(metadata["classical_bit"])
            active[old] = slots[gate.wires[0]]
            metadata["classical_bit"] = active[old]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (active[bit], value) for bit, value in metadata["conditions"]
            )
        remapped.append(
            Instruction(gate.name, gate.wires, gate.params, metadata=metadata)
        )
    assert_source(bound.circuit.instructions, tuple(remapped))
    ir = (
        Circuit(bound.circuit.n_wires + 1)
        .append(FieldOperation(bound.circuit.n_wires + 1, bound.circuit.instructions))
        .to_ir()
    )
    assert CircuitIR.from_json(ir.to_json()) == ir
    initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    bank_count = (1 << len(bank)) if borrowed else 1
    for control, address, target, contents, spectator in product(
        range(1 << controls), range(4), range(4), range(bank_count), range(2)
    ):
        word = (((control << 2) | address) << 2) | target
        loaded = address if control == (1 << controls) - 1 else 0
        before = (
            (word << (len(bank) + len(work) + 1))
            | (contents << (len(work) + 1))
            | spectator
        )
        after = (
            ((word ^ loaded) << (len(bank) + len(work) + 1))
            | (contents << (len(work) + 1))
            | spectator
        )
        coefficient = torch.tensor(
            complex(1 + before, 2 - target - contents), dtype=torch.complex128
        )
        initial[before] = coefficient
        expected[after] = coefficient
    norm = initial.norm()
    initial /= norm
    expected /= norm
    events = sum(g.name == "measure" for g in ir.instructions)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, ir.instructions, history),
            expected / 2 ** (events / 2),
            atol=1e-11,
            rtol=0,
        )
    assert (
        sum(g.name in {"t", "tdg"} for g in ir.instructions)
        == declaration.gate_cost().t + 4 * declaration.gate_cost().and_bloq
    )
    assert events == declaration.gate_cost().measurement


@pytest.mark.parametrize(
    "case", ("zero_word", "scalar", "short_logs", "extra_one", "prefix_extra_zero")
)
def test_original_failed_physical_paths_keep_headers_without_executable_bodies(
    case: str,
) -> None:
    from flagquantum.core.ir.program import FunctionRecord

    data: NDArray[np.int_] = np.arange(4)
    logs: tuple[int, ...] = (1,)
    selection: tuple[int, ...] = (2,)
    width = 2
    if case == "zero_word":
        data, width = np.zeros(4, dtype=int), 0
    elif case == "scalar":
        data, logs, selection = np.asarray(1), (), ()
    elif case == "short_logs":
        data, logs, selection = np.arange(4).reshape(2, 2), (0,), (1, 1)
    elif case == "extra_one":
        logs = (0, 1)
    else:
        logs = (1, 0)
    source = source_module.SelectSwapQROM(
        (data,), selection, (width,), log_block_sizes=logs
    )
    with pytest.raises((ValueError, IndexError, AssertionError, TypeError)):
        source.decompose_bloq()
    if case == "zero_word":
        declaration = SelectSwapDeclaration(
            QromDeclaration.from_arrays(
                data, selection_bitsizes=selection, target_bitsizes=(width,)
            ),
            logs,
        )
        with pytest.raises(IndexError):
            source_resources.get_cost_value(source, source_resources.QECGatesCost())
        with pytest.raises(IndexError, match="absent target"):
            declaration.gate_cost()
    result = next(iter(select_swap_factories().values()))(
        "main",
        None,
        {
            "data_or_shape": (data,),
            "selection_bitsizes": selection,
            "target_bitsizes": (width,),
            "log_block_sizes": logs,
        },
    )
    assert isinstance(result, FunctionRecord) and result.blocks == ()
    assert [(r.name, r.bitsize, r.shape) for r in result.registers or ()] == [
        (r.name, r.bitsize, r.shape) for r in source.signature
    ]


def test_default_blocks_replacement_and_original_zip_validation() -> None:
    data = np.arange(5)
    for dirty in (False, True):
        source = source_module.SelectSwapQROM.build_from_data(
            data, target_bitsizes=(3,), use_dirty_ancilla=dirty
        )
        native = SelectSwapDeclaration(
            QromDeclaration.from_data(data, target_bitsizes=(3,)), dirty=dirty
        )
        assert native.logs == source.log_block_sizes
        assert native.with_log_block_sizes() is native
        assert (
            native.with_log_block_sizes((0, 0, 0)).logs
            == source.with_log_block_sizes((0, 0, 0)).log_block_sizes
        )
        for update in (native.with_log_block_sizes, source.with_log_block_sizes):
            with pytest.raises(AssertionError):
                update((3,))
        source = source.with_data(np.arange(5)[::-1])
        changed = native.with_data(np.arange(5)[::-1])
        assert changed.logs == native.logs
        for original, actual in zip(
            source.padded_data, changed.padded_arrays, strict=True
        ):
            np.testing.assert_array_equal(original, actual)
        np.testing.assert_array_equal(native.lookup.data[0], data)
        for update in (native.with_data, source.with_data):
            with pytest.raises(AssertionError):
                update(np.arange(6))
        with pytest.raises(ValueError, match="scalar"):
            SelectSwapDeclaration(
                QromDeclaration.from_data(
                    np.arange(6).reshape(3, 2), target_shapes=((2,),)
                ),
                (0,),
            )


def test_empty_source_component_keeps_logical_calls_and_negative_static_quote() -> None:
    from flagquantum.compiler.registers.lookup.select_swap_qrom_component import (
        select_swap_qrom_program,
    )

    source = source_module.SelectSwapQROM((), (), (), log_block_sizes=())
    native = SelectSwapDeclaration(
        QromDeclaration.from_arrays(selection_bitsizes=(), target_bitsizes=()), ()
    )
    assert {call_key(child): n for child, n in native.calls().items()} == {
        call_key(child): n
        for child, n in source.build_call_graph(
            source_resources.SympySymbolAllocator()
        ).items()
    }
    assert (
        native.gate_cost().asdict()
        == source_resources.get_cost_value(
            source, source_resources.QECGatesCost()
        ).asdict()
        == {}
    )
    assert (
        native.static_qubit_count()
        == source_resources.get_cost_value(source, source_resources.QubitCount())
        == -1
    )
    record = select_swap_qrom_program(native, ())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == ()


@pytest.mark.parametrize("log", (2, 100_000_000))
def test_source_failed_or_oversized_banks_refuse_before_allocation(
    log: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir.program import FunctionRecord

    data = np.arange(4)
    source = source_module.SelectSwapQROM((data,), (2,), (2,), log_block_sizes=(log,))
    if log == 2:
        with pytest.raises(ValueError):
            source.decompose_bloq()
    else:
        # This source property is safe: no giant power or bank is materialized.
        assert source.batched_qrom_selection_bitsizes == (2 - log,)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("A failed source layout must not allocate a bank")

    monkeypatch.setattr(ComponentProgramBuilder, "allocate", forbidden)
    result = next(iter(select_swap_factories().values()))(
        "main",
        None,
        {
            "data_or_shape": (data,),
            "selection_bitsizes": (2,),
            "target_bitsizes": (2,),
            "log_block_sizes": (log,),
        },
    )
    assert isinstance(result, FunctionRecord) and not result.blocks


@pytest.mark.parametrize("data", ((0, 1), (0, 0)))
def test_boolean_target_width_is_not_projected_into_a_different_source_load(
    data: tuple[int, ...],
) -> None:
    from flagquantum.compiler.formats.legacy_qrom_factories import qrom_factories
    from flagquantum.core.ir.program import FunctionRecord
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    original = import_module("qualtran.bloqs.data_loading.qrom").QROM(
        (np.array(data),), (1,), (True,)
    )
    with pytest.raises((TypeError, ValueError)):
        reference_arithmetic_gates(
            original, {"selection": (0,), "target0_": (1,)}, (), {}, cirq_style=True
        )
    header = next(iter(qrom_factories().values()))(
        "main",
        None,
        {
            "data_or_shape": (np.array(data),),
            "selection_bitsizes": (1,),
            "target_bitsizes": (True,),
        },
    )
    assert isinstance(header, FunctionRecord) and not header.blocks


def test_boolean_word_count_has_a_source_model_but_no_physical_integer_load() -> None:
    data = np.array([0, 1])
    source = source_module.SelectSwapQROM(
        (data,), (1,), (True,), log_block_sizes=(0,), use_dirty_ancilla=False
    )
    native = SelectSwapDeclaration(
        QromDeclaration.from_arrays(
            data, selection_bitsizes=(1,), target_bitsizes=(True,)
        ),
        (0,),
        False,
    )
    for legacy in (False, True):
        assert (
            native.gate_cost(legacy_shims=legacy).asdict()
            == source_resources.get_cost_value(
                source, source_resources.QECGatesCost(legacy_shims=legacy)
            ).asdict()
        )
    with pytest.raises(ValueError, match="omitted target"):
        native.require_physical_layout()
