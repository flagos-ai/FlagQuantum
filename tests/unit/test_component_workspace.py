"""Replace pinned workspace construction with verified native function bodies."""

from dataclasses import replace

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_program_builder import identity
from tests.unit.test_component_rebindings import NativeRebindings
from tests.unit.test_register_workspace_contract import (
    test_allocated_word_can_be_returned_and_input_can_be_freed,
    test_free_requires_one_scalar_handle_before_reserving_a_call,
    test_stale_free_reserves_failed_instance_and_preserves_connections,
    test_workspace_survives_split_join_with_fresh_successors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

# Reuse the source scenarios with observations derived from real Program records.
__all__ = [
    "test_allocated_word_can_be_returned_and_input_can_be_freed",
    "test_workspace_survives_split_join_with_fresh_successors",
    "test_stale_free_reserves_failed_instance_and_preserves_connections",
    "test_free_requires_one_scalar_handle_before_reserving_a_call",
]


class NativeWorkspace(NativeRebindings):
    def allocate(self, n=1, dtype=None, dirty=False):
        return self.builder.allocate(n, dtype=dtype, dirty=dirty)

    def free(self, binding, dirty=False):
        return self.builder.free(binding, dirty=dirty)


@pytest.fixture(params=[NativeWorkspace], ids=["native-real-functions"])
def workspace(request):
    return request.param


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("width", [0, 1, 3])
def test_workspace_helpers_match_source_signatures_resources_and_explicit_calls(
    width, dirty
):
    from qualtran import QAny, Side
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = RegisterType("opaque", width)
    for source, helper, opcode, side in (
        (
            Allocate(QAny(width), dirty),
            allocate_word_function,
            "allocate_register",
            Side.RIGHT,
        ),
        (Free(QAny(width), dirty), free_word_function, "free_register", Side.LEFT),
    ):
        function = helper("leaf", dtype, dirty)
        assert verify_program_record(ProgramRecord((function,))) == ()
        port = function.registers[0]
        assert port.name == source.signature[0].name == "reg"
        assert port.bitsize == source.signature[0].bitsize == width
        assert port.shape == source.signature[0].shape == ()
        assert port.side.value == side.value
        (operation,) = function.blocks[0].operations
        assert operation.opcode == opcode and operation.workspace_dirty is dirty
        assert operation.callee is None
        # Source models are queried independently. Lifetime records have no
        # physical primitive emission before scheduling is admitted.
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert get_cost_value(source, QubitCount()) == width
        with pytest.raises(ValueError, match="physical pool scheduling"):
            lower_register_program(ProgramRecord((function,)), entry="leaf")


@pytest.mark.parametrize("dirty", [False, True])
def test_allocated_successor_flows_through_real_calls_before_release(dirty):
    signature = Signature([Register("reg", RegisterType("opaque", 3))])
    leaf = identity("leaf", signature)
    builder = ComponentProgramBuilder(Signature([]), [leaf])
    word = builder.allocate(3, dirty=dirty)
    successor = builder.call("leaf", {"reg": word})
    assert successor != word
    builder.free(successor, dirty=dirty)
    module = builder.finalize({})
    assert verify_program_record(module) == ()
    assert [op.callee for op in module.functions[0].blocks[0].operations][1] == "leaf"
    with pytest.raises(ValueError, match="physical pool scheduling"):
        lower_register_program(module)


@pytest.mark.parametrize("helper", [allocate_word_function, free_word_function])
@pytest.mark.parametrize("dirty", [None, 1, "yes"])
def test_workspace_construction_rejects_unadmitted_dirty_flags_before_mutation(
    helper, dirty
):
    with pytest.raises(ValueError, match="built-in boolean"):
        helper("leaf", RegisterType("opaque", 1), dirty)
    builder = ComponentProgramBuilder(Signature([]))
    with pytest.raises(ValueError, match="built-in boolean"):
        builder.allocate(dirty=dirty)
    assert not builder._operations and not builder._functions
    assert verify_program_record(builder.finalize({})) == ()


@pytest.mark.parametrize(
    "dtype",
    [RegisterType("uint", -1), RegisterType("signed", 3), RegisterType("uint", 3, 1)],
)
def test_workspace_executable_descriptor_admission_precedes_call_reservation(dtype):
    builder = ComponentProgramBuilder(Signature([]))
    with pytest.raises(ValueError, match="unsupported"):
        builder.allocate(dtype=dtype)
    assert not builder._operations
    assert not builder._functions


def test_workspace_helper_names_avoid_existing_definitions_and_entry():
    empty = FunctionRecord(
        "__fq_rebinding_0", (BlockRecord("entry", return_values=()),), registers=()
    )
    builder = ComponentProgramBuilder(Signature([]), [empty], name="__fq_rebinding_1")
    word = builder.allocate(0)
    builder.free(word)
    module = builder.finalize({})
    assert [f.name for f in module.functions] == [
        "__fq_rebinding_1",
        "__fq_rebinding_0",
        "__fq_rebinding_2",
        "__fq_rebinding_3",
    ]


def test_lowering_refuses_unreachable_lifetimes_before_transforming_any_definition():
    main = identity("main", Signature([]))
    unreachable = allocate_word_function(
        "unreachable", RegisterType("opaque", 1), False
    )
    module = ProgramRecord((main, unreachable))
    with pytest.raises(ValueError, match="physical pool scheduling"):
        lower_register_program(module)
    assert module.functions == (main, unreachable)


def test_workspace_dtype_is_derived_after_real_split_and_join():
    builder = ComponentProgramBuilder(Signature([]))
    word = builder.allocate(3)
    successor = builder.join(builder.split(word), RegisterType("uint", 3))
    builder.free(successor)
    module = builder.finalize({})
    free = module.functions[-1]
    assert free.registers[0].dtype == RegisterType("uint", 3)
    block = free.blocks[0]
    operation = replace(block.operations[0], workspace_dirty=None)
    invalid = replace(free, blocks=(replace(block, operations=(operation,)),))
    tampered = replace(module, functions=(*module.functions[:-1], invalid))
    with pytest.raises(ValueError, match="invalid register Program"):
        lower_register_program(tampered)


def test_unrelated_dirty_metadata_is_rejected_in_a_declared_register_function():
    dtype = RegisterType("opaque", 1)
    before, after = (ValueRef("entry", i, "register") for i in range(2))
    leaf = identity("leaf", Signature([Register("reg", dtype)]))
    operation = Operation(
        "call", (before,), (after,), callee="leaf", workspace_dirty=False
    )
    main = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry", (before,), operations=(operation,), return_values=(after,)
            ),
        ),
        registers=leaf.registers,
    )
    assert "program.register_structure" in {
        d.code for d in verify_program_record(ProgramRecord((main, leaf)))
    }
