"""Block constructors keep source control adders, logical junk and exact phases."""

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_block_encoding_factories import ExactPair
from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_controlled_factories import (
    controlled_via_and_factories,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_adjoint import adjoint_component_calls
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import ControlAdder
from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_z_power_program,
)
from flagquantum.compiler.registers.primitives.pauli_rotation_component import (
    z_power_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._block_encoding_component_library import (
    block_encoding_library_factories,
)
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _oracles(
    local_junk: bool = False,
) -> tuple[Any, Any, ExactPair, ExactPair, ExactPair]:
    import attrs

    np = pytest.importorskip("numpy")
    from qualtran import QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import (
        CNOT,
        CZPowGate,
        Hadamard,
        SGate,
        Toffoli,
        XGate,
        ZPowGate,
    )
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        AdjointWithSpecializedCtrl,
        SpecializeOnCtrlBit,
        get_ctrl_system_1bit_cv_from_bloqs,
    )
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    class Prepare:
        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("address", QBit()),)

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return (
                SourceRegister(
                    "junk", QBit(), side=Side.RIGHT if local_junk else Side.THRU
                ),
            )

        @property
        def l1_norm_of_coeffs(self) -> float:
            return 3.0

        def build_composite_bloq(
            self, bb: Any, address: Any, junk: Any = None
        ) -> dict[str, Any]:
            address = bb.add(Hadamard(), q=address)
            address = bb.add(SGate(), q=address)
            if local_junk:
                junk = bb.allocate(dtype=QBit())
                address, junk = bb.add(CNOT(), ctrl=address, target=junk)
            else:
                junk = bb.add(XGate(), q=junk)
            return {"address": address, "junk": junk}

    class Select:
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("address", QBit()),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("data", QBit()),)

        def build_composite_bloq(
            self, bb: Any, address: Any, data: Any
        ) -> dict[str, Any]:
            address, data = bb.add(CNOT(), ctrl=address, target=data)
            data = bb.add(ZPowGate(0.37), q=data)
            return {"address": address, "data": data}

        def get_ctrl_system(self, ctrl_spec: Any) -> Any:
            return get_ctrl_system_1bit_cv_from_bloqs(
                self,
                ctrl_spec,
                current_ctrl_bit=None,
                bloq_with_ctrl=controlled_type(),
                ctrl_reg_name="control",
            )

        def adjoint(self) -> Any:
            return AdjointWithSpecializedCtrl(self, SpecializeOnCtrlBit.ONE)

    class Controlled(Select):
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("control", QBit()),)

        def get_ctrl_system(self, ctrl_spec: Any) -> Any:
            return get_ctrl_system_1bit_cv_from_bloqs(
                self,
                ctrl_spec,
                current_ctrl_bit=1,
                bloq_with_ctrl=self,
                ctrl_reg_name="control",
            )

        def build_composite_bloq(
            self, bb: Any, address: Any, data: Any, control: Any = None
        ) -> dict[str, Any]:
            controls, data = bb.add(
                Toffoli(), ctrl=np.array([control, address]), target=data
            )
            pair = bb.add(CZPowGate(0.37), q=np.array([controls[0], data]))
            return {"control": pair[0], "address": controls[1], "data": pair[1]}

    prepare_type = attrs.frozen(
        type(
            "SmallLocalPrepare" if local_junk else "SmallPrepare",
            (Prepare, PrepareOracle),
            {"__module__": __name__},
        )
    )
    select_type = attrs.frozen(
        type("SmallSelect", (Select, SelectOracle), {"__module__": __name__})
    )
    controlled_type = attrs.frozen(
        type(
            "SmallControlledSelect",
            (Controlled, SelectOracle),
            {"__module__": __name__},
        )
    )
    bit = quantum_bit_type()
    gates = fixed_gate_factories()

    def fixed(
        suffix: str, name: str, arguments: Mapping[str, ResolvedArgument] | None = None
    ) -> ProgramRecord:
        result = gates["qualtran.bloqs.basic_gates." + suffix](
            name, None, arguments or {}
        )
        assert isinstance(result, FunctionRecord)
        return ProgramRecord((result,))

    h, s, x = (
        fixed("hadamard.Hadamard", "h"),
        fixed("s_gate.SGate", "s"),
        fixed("x_basis.XGate", "x"),
    )
    sdg = fixed("s_gate.SGate", "sdg", {"is_adjoint": True})
    prep = ComponentProgramBuilder(
        Signature(
            (
                Register("address", bit),
                Register(
                    "junk",
                    bit,
                    side=RegisterSide.RIGHT if local_junk else RegisterSide.THRU,
                ),
            )
        )
    )
    address = prep.inputs["address"]
    for gate in (h, s):
        address = prep.call_program(gate, {"q": address}, entry=gate.functions[0].name)[
            "q"
        ]
    local_cnot = fixed("cnot.CNOT", "local_cx")
    junk: BindingCollection
    if local_junk:
        junk = prep.allocate()
        copied = prep.call_program(
            local_cnot, {"ctrl": address, "target": junk}, entry="local_cx"
        )
        address, junk = copied["ctrl"], copied["target"]
    else:
        junk = prep.call_program(x, {"q": prep.inputs["junk"]}, entry="x")["q"]
    prep_record = prep.finalize({"address": address, "junk": junk})
    prep_pair = prep_record, adjoint_component_calls(
        prep_record, ((h, h), (s, sdg), (x, x), (local_cnot, local_cnot))
    )
    cnot, toffoli = fixed("cnot.CNOT", "cx"), fixed("toffoli.Toffoli", "ccx")
    phase = ProgramRecord((z_power_function("phase", 0.37),))
    inverse_phase = ProgramRecord((z_power_function("phase_inv", -0.37),))
    select = ComponentProgramBuilder(
        Signature((Register("address", bit), Register("data", bit)))
    )
    out = select.call_program(
        cnot,
        {"ctrl": select.inputs["address"], "target": select.inputs["data"]},
        entry="cx",
    )
    data = select.call_program(phase, {"q": out["target"]}, entry="phase")["q"]
    select_record = select.finalize({"address": out["ctrl"], "data": data})
    select_pair = select_record, adjoint_component_calls(
        select_record, ((cnot, cnot), (phase, inverse_phase))
    )
    controlled = ComponentProgramBuilder(
        Signature(
            (Register("control", bit), Register("address", bit), Register("data", bit))
        )
    )
    out = controlled.call_program(
        toffoli,
        {
            "ctrl": np.array(
                [controlled.inputs["control"], controlled.inputs["address"]]
            ),
            "target": controlled.inputs["data"],
        },
        entry="ccx",
    )
    controls = np.asarray(out["ctrl"])
    cp = controlled_z_power_program(0.37)
    cp_inv = controlled_z_power_program(-0.37)
    pair = np.asarray(
        controlled.call_program(cp, {"q": np.array([controls[0], out["target"]])})["q"]
    )
    control_record = controlled.finalize(
        {"control": pair[0], "address": controls[1], "data": pair[1]}
    )
    control_pair = control_record, adjoint_component_calls(
        control_record, ((toffoli, toffoli), (cp, cp_inv))
    )
    return prepare_type(), select_type(), prep_pair, select_pair, control_pair


def _factories(
    prep_pair: ExactPair, select_pair: ExactPair, control_pair: ExactPair
) -> dict[str, ComponentFactory]:
    np = pytest.importorskip("numpy")
    bit = quantum_bit_type()
    known_prep: set[FunctionRecord] = set()
    known_select: set[FunctionRecord] = set()

    def child_factory(kind: str, pair: ExactPair) -> ComponentFactory:
        def factory(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> ProgramRecord:
            assert not arguments and registers == pair[0].functions[0].registers
            result = _rename_program(pair[0], name)
            (known_prep if kind == "prepare" else known_select).add(result.functions[0])
            return result

        return factory

    def prep_owner(
        child: FunctionRecord,
    ) -> tuple[tuple[Register, ...], tuple[Register, ...], ExactPair]:
        assert child in known_prep
        return (
            (Register("address", bit),),
            tuple(
                reg
                for reg in prep_pair[0].functions[0].registers or ()
                if reg.name == "junk"
            ),
            prep_pair,
        )

    def select_owner(
        child: FunctionRecord,
    ) -> tuple[tuple[Register, ...], tuple[Register, ...], ExactPair]:
        assert child in known_select
        return (Register("address", bit),), (Register("data", bit),), select_pair

    def control_owner(
        child: FunctionRecord, value: int
    ) -> tuple[ControlAdder, tuple[ExactPair, ...]]:
        assert child in known_select

        def adder(module: ProgramRecord, control_name: str) -> ControlAdder:
            def add(
                owner: ComponentProgramBuilder,
                control: BindingCollection,
                inputs: dict[str, BindingCollection],
            ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                out = owner.call_program(
                    module,
                    {control_name: control, **inputs},
                    entry=module.functions[0].name,
                )
                return out[control_name], (out["address"], out["data"])

            return add

        if value == 1:
            return adder(control_pair[0], "control"), (control_pair,)
        controlled = []
        for i in (0, 1):

            def positive(
                original: FunctionRecord, *, index: int = i
            ) -> tuple[ProgramRecord, ControlAdder]:
                assert original == child
                return control_pair[index], adder(control_pair[index], "control")

            factory = controlled_via_and_factories(control_system=positive)[
                "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
            ]
            record = factory(
                "zero_control_" + str(i),
                None,
                {
                    "subbloq": child,
                    "ctrl_spec": ControlSpecification(((bit, np.array(0)),)),
                },
            )
            assert isinstance(record, ProgramRecord)
            controlled.append(record)
        return adder(controlled[0], "ctrl"), ((controlled[0], controlled[1]),)

    factories = block_encoding_library_factories(
        prepare_context=prep_owner,
        select_context=select_owner,
        controlled_select=control_owner,
    )
    factories[__name__ + ".SmallPrepare"] = child_factory("prepare", prep_pair)
    factories[__name__ + ".SmallLocalPrepare"] = child_factory("prepare", prep_pair)
    factories[__name__ + ".SmallSelect"] = child_factory("select", select_pair)
    return factories


@pytest.mark.parametrize("control", (None, 0, 1))
def test_lcu_source_root_and_complete_complex_operator(control: int | None) -> None:
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding

    prepare, select, prep_pair, select_pair, controlled = _oracles()
    original = LCUBlockEncoding(select, prepare, control_val=control)
    factories = _factories(prep_pair, select_pair, controlled)
    record = resolve_component_program(_library(original, 0), 0, factories)
    assert not verify_program_record(record)
    # Generic control adders may reorder keyword dictionaries. Compare named
    # operands, ordered outputs and full signatures through the shared walker.
    compare_layers(record, original, atomic_leaves(record))
    logical = 3 + int(control is not None)
    bound = bind_register_program(
        record,
        tuple((i,) for i in range(logical)),
        clean=(logical,) if control is not None else (),
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(original.tensor_contract(), dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )


@pytest.mark.parametrize(
    "example",
    ("_select_block", "_black_box_select_block", "_lcu_block", "_black_box_lcu_block"),
)
def test_all_original_forward_examples_execute_complete_hubbard_dependencies(
    example: str,
) -> None:
    from qualtran.bloqs.block_encoding import lcu_block_encoding as module

    original = getattr(module, example).make()
    record = resolve_component_program(
        _library(original, 0), 0, block_encoding_library_factories()
    )
    assert not verify_program_record(record)
    assert (
        record.functions[0].registers
        == _library(original, 0).entries[0].declaration.registers
    )
    assert _native_trace(record) == _source_trace(
        SourceCalls, original.decompose_bloq()
    )
    ports = record.functions[0].registers or ()
    inputs = []
    cursor = 0
    for port in ports:
        width = int(port.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(cursor, cursor + 8))
    )
    assert bound.outputs == tuple(inputs)
    assert all(gate.matrix is None for gate in bound.circuit.instructions)
    assert sum(gate.name == "phase" for gate in bound.circuit.instructions) == 2
    assert any(gate.name == "measure" for gate in bound.circuit.instructions)


def test_single_bit_packed_properties_remain_opaque_across_lcu_calls() -> None:
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    prepare, select, prep_pair, select_pair, controlled = _oracles()
    source = LCUBlockEncoding(BlackBoxSelect(select), BlackBoxPrepare(prepare))
    record = resolve_component_program(
        _library(source, 0), 0, _factories(prep_pair, select_pair, controlled)
    )
    assert not verify_program_record(record)
    assert all(
        reg.dtype.encoding == "opaque" for reg in record.functions[0].registers or ()
    )
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(record, ((0,), (1,), (2,)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )


@pytest.mark.parametrize("control", (0, 1))
def test_original_hubbard_control_adders_match_source_root_and_bind_all_dependencies(
    control: int,
) -> None:
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        PrepareHubbard,
        SelectHubbard,
    )

    source = LCUBlockEncoding(
        SelectHubbard(2, 2), PrepareHubbard(2, 2, 1.0, 4.0), control_val=control
    )
    record = resolve_component_program(
        _library(source, 0), 0, block_encoding_library_factories()
    )
    assert not verify_program_record(record)
    from tests.unit.test_hubbard_stored_graph import source_terminals

    terminals = source_terminals(record) | {
        fn.name for fn in record.functions if not fn.blocks
    }
    compare_layers(record, source, terminals)
    ports = record.functions[0].registers or ()
    inputs = []
    cursor = 0
    for port in ports:
        width = int(port.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(cursor, cursor + 8))
    )
    assert bound.outputs == tuple(inputs)
    assert all(gate.matrix is None for gate in bound.circuit.instructions)
    assert any(gate.name == "measure" for gate in bound.circuit.instructions)


@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("local_junk", (False, True))
def test_exact_inverse_full_channels_raw_histories_and_local_junk(
    control: int | None, local_junk: bool
) -> None:
    from itertools import product
    from math import sqrt

    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding

    from tests.unit.measurement_checks import forced_branch

    prepare, select, prep_pair, select_pair, controlled = _oracles(local_junk)
    original = LCUBlockEncoding(select, prepare, control_val=control)
    factories = _factories(prep_pair, select_pair, controlled)
    forward = resolve_component_program(_library(original, 0), 0, factories)
    inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "main", None, {"subbloq": forward.functions[0]}
    )
    assert isinstance(inverse, ProgramRecord)
    assert not verify_program_record(inverse)
    compare_layers(inverse, original.adjoint(), atomic_leaves(inverse))
    logical = 3 + int(control is not None) - int(local_junk)
    clean = tuple(range(logical, logical + int(local_junk) + int(control is not None)))
    matrices = []
    bound_pair = []
    for record, source in ((forward, original), (inverse, original.adjoint())):
        bound = bind_register_program(
            record, tuple((i,) for i in range(logical)), clean=clean
        )
        bound_pair.append(bound)
        actual = logical_operator(bound.circuit, logical)
        torch.testing.assert_close(
            actual,
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=2e-11,
            rtol=0,
        )
        matrices.append(actual)
    torch.testing.assert_close(
        matrices[1] @ matrices[0],
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )
    gates = (*bound_pair[0].circuit.instructions, *bound_pair[1].circuit.instructions)
    count = sum(g.name == "measure" for g in gates)
    state = torch.zeros(1 << bound_pair[0].circuit.n_wires, dtype=torch.complex128)
    values = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
    seed = torch.complex(values, (values % 3) - 1)
    seed /= seed.norm()
    state[:: 1 << len(clean)] = seed
    for history in product((0, 1), repeat=count):
        actual = forced_branch(state, gates, history) * sqrt(2**count)
        torch.testing.assert_close(actual, state, atol=2e-11, rtol=0)


def test_select_block_does_not_execute_signal_preparation() -> None:
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding

    prepare, select, prep_pair, select_pair, controlled = _oracles()
    source = SelectBlockEncoding(select, prepare)
    record = resolve_component_program(
        _library(source, 0), 0, _factories(prep_pair, select_pair, controlled)
    )
    compare_layers(record, source, atomic_leaves(record))
    assert len(_native_trace(record)) == 1
    assert tuple(reg.name for reg in record.functions[0].registers or ()) == (
        "address",
        "data",
    )
    bound = bind_register_program(record, ((0,), (1,)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )
    assert not bound.events
    assert tuple(g.name for g in bound.circuit.instructions) == ("cx", "phase")


def test_source_free_zero_control_local_junk_resolution_and_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    prepare, select, prep_pair, select_pair, controlled = _oracles(True)
    source = LCUBlockEncoding(select, prepare, control_val=0)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    library = _library(source, 0)
    factories = _factories(prep_pair, select_pair, controlled)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(library, 0, factories)
        bound = bind_register_program(record, ((0,), (1,), (2,)), clean=(3, 4))
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert (
            circuit.to_ir().to_dict()["instructions"]
            == bound.circuit.to_dict()["instructions"]
        )
        actual = logical_operator(bound.circuit, 3)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=0)
