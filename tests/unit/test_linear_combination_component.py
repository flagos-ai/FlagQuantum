"""Compare native packed LCU graphs with the unchanged source root."""

import pytest
import torch

from flagquantum.compiler.registers.layout.packed_oracle_component import (
    packed_oracle_component_program,
)
from flagquantum.compiler.registers.preparation.linear_combination_component import (
    linear_combination_component_program,
    linear_combination_signature,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_block_encoding_factories import _oracles
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_source_free_native_construction_binding_and_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir import CircuitIR, Instruction

    prep = ComponentProgramBuilder(Signature.build(selection=1, junk=1))
    preparation = prep.finalize(prep.inputs)
    selection = ProgramRecord(
        (
            bit_gate_function(
                "select",
                tuple(Signature.build(selection=1, system=1)),
                CircuitIR(2, (Instruction("cx", (0, 1)),)),
            ),
        )
    )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = linear_combination_component_program(
            linear_combination_signature(
                1, (0, 0), (0, 0), prepare_selection=1, prepare_junk=1
            ),
            constituent_ancilla=0,
            constituent_resource=0,
            prepare_selection=1,
            prepare_junk=1,
            prepare_pair=(preparation, preparation),
            select_pair=(selection, selection),
        )
        bound = bind_register_program(record, ((0,), (1,), (2,)))
        circuit = Circuit(3)
        circuit.append(_BoundBlockOperation(bound.circuit.instructions, 3))
        assert circuit.to_ir().instructions == (Instruction("cx", (1, 0)),)


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
def test_supplied_oracles_exact_root_calls_and_complex_operator(
    inverse: bool, controlled: bool
) -> None:
    from hashlib import sha1
    from importlib.metadata import version
    from pathlib import Path

    import kickmix
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding import linear_combination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, expected in (
        ("linear_combination.py", "a3d8aa7fafa7a6f3fcaf095826751209427cd255"),
        ("linear_combination_test.py", "bed60bc7ade281fca1e5b13e341e8b4dd07aabd9"),
    ):
        raw = Path(linear_combination.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
    prepare, select, prepare_pair, select_pair, control_pair = _oracles()
    source = linear_combination.LinearCombination(
        (Unitary(XGate()), Unitary(XGate())),
        (0.5, 0.5),
        1,
        prepare=BlackBoxPrepare(prepare),
        select=BlackBoxSelect(select),
        is_controlled=controlled,
    )

    def pack(
        child: object, pair: tuple[ProgramRecord, ProgramRecord], preparation: bool
    ) -> tuple[ProgramRecord, ProgramRecord]:
        signature = Signature(_library(child, 0).entries[0].declaration.registers or ())
        ports = pair[0].functions[0].registers or ()
        groups = (
            (("selection", (ports[0],)), ("junk", (ports[1],)))
            if preparation
            else (("selection", (ports[0],)), ("system", (ports[1],)))
        )
        forward, inverse = tuple(
            packed_oracle_component_program(
                signature, groups, pair[0], inverse=pair[1], adjoint=direction
            )
            for direction in (False, True)
        )
        return forward, inverse

    native_prepare = pack(source.prepare, prepare_pair, True)
    native_select = pack(source.select, select_pair, False)
    if controlled:
        from flagquantum.compiler.registers.component_builder import (
            BindingCollection,
            ComponentProgramBuilder,
        )
        from flagquantum.compiler.registers.component_control import (
            ControlAdder,
            control_component_hierarchy,
        )
        from flagquantum.compiler.registers.control.component_bookkeeping import (
            bookkeeping_component_control_system,
        )

        controlled_modules = []
        for direction, module in enumerate(native_select):
            child = select.adjoint() if direction else select
            upstream = source.select.adjoint() if direction else source.select
            correspondence = compare_layers(module, upstream, atomic_leaves(module))
            targets = {
                name
                for name, source_child in correspondence.items()
                if source_child == child
            }
            assert len(targets) == 1

            def positive(
                owner: ComponentProgramBuilder,
                ctrl: BindingCollection,
                values: dict[str, BindingCollection],
                *,
                index: int = direction,
            ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                out = owner.call_program(
                    control_pair[index],
                    {"control": ctrl, **values},
                    entry=control_pair[index].functions[0].name,
                )
                return out["control"], (out["address"], out["data"])

            adders: dict[str, ControlAdder] = dict.fromkeys(targets, positive)
            for function in module.functions:
                if function.name == "main" or function.name in adders:
                    continue
                try:
                    _, add = bookkeeping_component_control_system(module, function.name)
                except ValueError:
                    continue

                def bookkeeping(
                    owner: ComponentProgramBuilder,
                    ctrl: BindingCollection,
                    values: dict[str, BindingCollection],
                    *,
                    name: str = function.name,
                ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                    return ctrl, owner.call_tuple(name, values)

                adders[function.name] = bookkeeping
            controlled_modules.append(control_component_hierarchy(module, adders))
        native_select = controlled_modules[0], controlled_modules[1]
    record = linear_combination_component_program(
        linear_combination_signature(
            1,
            (0, 0),
            (0, 0),
            prepare_selection=1,
            prepare_junk=1,
            controlled=controlled,
        ),
        constituent_ancilla=0,
        constituent_resource=0,
        prepare_selection=1,
        prepare_junk=1,
        prepare_pair=native_prepare,
        select_pair=native_select,
        adjoint=inverse,
    )
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    logical = 3 + int(controlled)
    bound = bind_register_program(
        record,
        tuple((i,) for i in range(logical)),
        clean=(logical,) if controlled else (),
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )
    if controlled:
        from itertools import product
        from math import sqrt

        from tests.unit.measurement_checks import forced_branch

        matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
        index = torch.arange(1 << logical, dtype=torch.float64)
        state = torch.complex(index % 7 - 3, index % 11 - 5)
        state /= state.norm()
        clean = torch.tensor([1, 0], dtype=torch.complex128)
        count = sum(gate.name == "measure" for gate in bound.circuit.instructions)
        assert count == 1
        for history in product((0, 1), repeat=count):
            torch.testing.assert_close(
                forced_branch(
                    torch.kron(state, clean), bound.circuit.instructions, history
                ),
                torch.kron(matrix @ state, clean) / sqrt(2**count),
                atol=5e-11,
                rtol=0,
            )
