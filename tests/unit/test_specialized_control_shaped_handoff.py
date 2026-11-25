"""Shaped target callbacks and symbolic declarations retain source control graphs."""

from collections import Counter
from itertools import product
from math import prod
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.specialized_component_control import (
    multi_controlled_component_program,
    specialized_component_control_system,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_component_control_selection import assert_gate_parameters
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_specialized_control_typed_handoff import packed_dtype

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def shaped_positive(
    dtype: RegisterType, original_dtype: Any, shape: tuple[int, ...]
) -> tuple[ProgramRecord, Any]:
    """Independently construct typed native calls and a source CH fixture."""
    import attrs
    from qualtran import Bloq, BloqBuilder, QBit
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature
    from qualtran.bloqs.basic_gates import CHadamard

    @attrs.frozen
    class ArrayHadamards:
        @property
        def signature(self) -> Any:
            return OriginalSignature(
                (
                    OriginalRegister("ctrl", QBit()),
                    OriginalRegister("word", original_dtype, shape),
                )
            )

        def build_composite_bloq(self, bb: BloqBuilder, **soqs: Any) -> dict[str, Any]:
            words = np.asarray(soqs["word"], dtype=object)
            result = np.empty(shape, dtype=object)
            control = soqs["ctrl"]
            for index in np.ndindex(shape):
                bits = bb.split(words[index])
                if len(bits):
                    control, bits[0] = bb.add(CHadamard(), ctrl=control, target=bits[0])
                result[index] = bb.join(bits, dtype=original_dtype)
            return {"ctrl": control, "word": result if shape else result[()]}

    control = Register("ctrl", quantum_bit_type())
    target = Register("target", quantum_bit_type())
    leaf = bit_gate_function(
        "ch", (control, target), CircuitIR(2, (Instruction("ch", (0, 1)),))
    )
    builder = ComponentProgramBuilder(
        Signature((control, Register("word", dtype, shape))), (leaf,), name="positive"
    )
    words = np.asarray(builder.inputs["word"], dtype=object)
    result = np.empty(shape, dtype=object)
    current = builder.inputs["ctrl"]
    for index in np.ndindex(shape):
        word = words[index]
        assert isinstance(word, RegisterBinding)
        bits = builder.split(word)
        if len(bits):
            outputs = builder.call_dict("ch", {"ctrl": current, "target": bits[0]})
            current, bits[0] = outputs["ctrl"], outputs["target"]
        result[index] = builder.join(bits, dtype)
    return (
        builder.finalize({"ctrl": current, "word": result if shape else result[()]}),
        type("OriginalArrayHadamards", (ArrayHadamards, Bloq), {})(),
    )


def shaped_source_gates(
    source: Any, ports: SourcePorts, work: tuple[int, ...], slots: dict[int, int]
) -> tuple[Instruction, ...]:
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import CHadamard
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, CHadamard):
            return (Instruction("ch", (*inputs["ctrl"], *inputs["target"])),), inputs
        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if not isinstance(base, (And, MultiAnd)):
            return None
        erase = bloq is not base or isinstance(base, And) and base.uncompute
        outputs = {"ctrl": inputs["ctrl"]}
        if not erase:
            outputs["target"] = work[-1:]
            if len(work) > 1:
                outputs["junk"] = work[:-1]
        return (
            reference_arithmetic_gates(
                bloq,
                inputs if erase else {**inputs, **outputs},
                (),
                slots,
                expected_outputs=outputs,
                cirq_style=True,
            ),
            outputs,
        )

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates


@pytest.mark.parametrize(
    "profile",
    (
        "opaque",
        "uint",
        "bounded",
        "montgomery",
        "fixed-2",
        "fixed0",
        "fixed3",
        "field",
        "polynomial",
    ),
)
@pytest.mark.parametrize("shape", ((), (2,), (1, 2), (2, 2), (0, 2)))
@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
def test_shaped_actual_callback_streams_complex_histories_and_output_aliases(
    profile: str, shape: tuple[int, ...], old: int, new: int
) -> None:
    from qualtran import BloqBuilder, CtrlSpec, QBit
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype, source_dtype = packed_dtype(profile)
    positive, source_positive = shaped_positive(dtype, source_dtype, shape)
    selected, adder = specialized_component_control_system(
        positive, "ctrl", old, new, entry="positive"
    )
    source, source_adder = get_ctrl_system_1bit_cv(
        source_positive,
        CtrlSpec(cvs=new),
        current_ctrl_bit=old,
        get_ctrl_bloq_and_ctrl_reg_name=lambda _: (source_positive, "ctrl"),
    )
    signature = Signature(
        (
            Register("added", quantum_bit_type()),
            Register("ctrl", quantum_bit_type()),
            Register("word", dtype, shape),
        )
    )
    builder = ComponentProgramBuilder(signature, selected.functions, name="caller")
    source_builder, inputs = BloqBuilder.from_signature(
        OriginalSignature(
            (OriginalRegister("added", QBit()), *source_positive.signature)
        )
    )
    values = {name: builder.inputs[name] for name in ("ctrl", "word")}
    original_values = {name: inputs[name] for name in ("ctrl", "word")}
    controls, data = adder(builder, builder.inputs["added"], values)
    source_controls, source_data = source_adder(
        source_builder, (inputs["added"],), original_values
    )
    caller = builder.finalize({"added": controls, "ctrl": data[0], "word": data[1]})
    source_caller = source_builder.finalize(
        **dict(
            zip(
                ("added", "ctrl", "word"), (*source_controls, *source_data), strict=True
            )
        )
    )
    assert verify_program_record(caller) == ()
    assert tuple(values) == tuple(original_values) == ("word", "ctrl")
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(SourceCalls, source_caller)
    )
    count = prod(shape)
    words = tuple(tuple(range(2 + 3 * i, 5 + 3 * i)) for i in range(count))
    logical = 2 + 3 * count
    work = (logical,)
    bound = bind_register_program(
        caller, ((0,), (1,), *words), clean=work, entry="caller"
    )
    assert bound.outputs == ((0,), (1,), *words)
    assert all(binding.register.dtype == dtype for binding in np.asarray(data[1]).flat)
    gates = bound.circuit.instructions
    slots = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    reference = shaped_source_gates(
        source,
        {"ctrl": (0, 1), "word": tuple(w for word in words for w in word)},
        work,
        slots,
    )
    assert_gate_parameters(gates, reference)
    model = get_cost_value(source, QECGatesCost())
    emitted = Counter(g.name for g in gates)
    assert emitted["ch"] == count
    assert emitted["measure"] == emitted["reset"] == model.measurement == 1
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == logical + 1
    appended = (
        DynamicCircuit(logical + 2)
        .append(FieldOperation(logical + 2, gates))
        .to_ir()
        .instructions
    )
    assert appended == gates
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    amplitudes = torch.tensor(
        [complex(i + 1, 2 - i % 13) for i in range(1 << (logical + 1))],
        dtype=torch.complex128,
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 2), dtype=torch.complex128)
    for i, amplitude in enumerate(amplitudes):
        initial[(i >> 1 << 2) | (i & 1)] = amplitude
    expected = initial.clone()
    addresses = torch.arange(expected.numel())
    active = (addresses >> logical & 3) == (new << 1 | old)
    for word in words:
        bit = 1 << (logical + 1 - word[0])
        low = addresses[active & (addresses & bit == 0)]
        high = low | bit
        left, right = expected[low].clone(), expected[high].clone()
        expected[low] = (left + right) / 2**0.5
        expected[high] = (left - right) / 2**0.5
    for outcome in (0, 1):
        actual = forced_branch(initial, appended, (outcome,))
        torch.testing.assert_close(actual, expected / 2**0.5, atol=2e-12, rtol=0)
        assert actual.reshape(-1, 4)[:, 2:].abs().max().item() < 2e-12
    for first, second in product((0, 1), repeat=2):
        restored = forced_branch(
            forced_branch(initial, appended, (first,)), appended, (second,)
        )
        torch.testing.assert_close(restored, initial / 2, atol=2e-12, rtol=0)


@pytest.mark.parametrize("width_case", ("n", "2*n+1"))
@pytest.mark.parametrize("shape", ((), (2,), (1, 2)))
def test_symbolic_target_declarations_retain_source_graph_and_explicit_binding_refusal(
    width_case: str, shape: tuple[int, ...]
) -> None:
    import attrs
    import sympy
    from qualtran import Bloq, QAny, QBit
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )

    n = sympy.Symbol("n", integer=True, positive=True)
    width = n if width_case == "n" else 2 * n + 1
    registers = (
        Register("ctrl", quantum_bit_type()),
        Register("word", RegisterType("opaque", width), shape),
    )
    leaf = FunctionRecord("declared", (), registers=registers)
    builder = ComponentProgramBuilder(Signature(registers), (leaf,), name="positive")
    positive = builder.finalize(builder.call_dict("declared", builder.inputs))

    @attrs.frozen
    class Declared:
        @property
        def signature(self) -> Any:
            return OriginalSignature(
                (
                    OriginalRegister("ctrl", QBit()),
                    OriginalRegister("word", QAny(width), shape),
                )
            )

    source = _MultiControlledFromSinglyControlled(
        (0, 1), type("OriginalDeclared", (Declared, Bloq), {})(), "ctrl"
    )
    module = multi_controlled_component_program(positive, "ctrl", (0, 1), "positive")
    assert verify_program_record(module) == ()
    assert normalize_trace(_native_trace(module, "positive")) == normalize_trace(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    with pytest.raises(ValueError, match="concrete|bodies"):
        bind_register_program(
            module,
            (
                (0,),
                (1,),
                *(tuple(range(2 + 3 * i, 5 + 3 * i)) for i in range(prod(shape))),
            ),
            clean=(20,),
            entry="positive",
        )


@pytest.mark.parametrize("shape", ((), (2,), (1, 2)))
def test_signed_fixed_target_executes_approved_storage_and_all_erasure_branches(
    shape: tuple[int, ...],
) -> None:
    from qualtran import QFxp
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.core.ir.program.register_types import fixed_point_register_type

    dtype = fixed_point_register_type(3, 1, signed=True)
    positive, original_positive = shaped_positive(dtype, QFxp(3, 1, signed=True), shape)
    module = multi_controlled_component_program(positive, "ctrl", (0, 1), "positive")
    source = _MultiControlledFromSinglyControlled((0, 1), original_positive, "ctrl")
    assert verify_program_record(module) == ()
    assert normalize_trace(_native_trace(module, "positive")) == normalize_trace(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    words = tuple(tuple(range(2 + 3 * i, 5 + 3 * i)) for i in range(prod(shape)))
    work = (2 + 3 * prod(shape),)
    gates = shaped_source_gates(
        source,
        {"ctrl": (0, 1), "word": tuple(w for word in words for w in word)},
        work,
        {work[0]: 0},
    )
    counts = Counter(g.name for g in gates)
    assert counts["ch"] == prod(shape)
    assert counts["measure"] == get_cost_value(source, QECGatesCost()).measurement == 1
    bound = bind_register_program(
        module, ((0,), (1,), *words), clean=work, entry="positive"
    )
    assert bound.outputs == ((0,), (1,), *words)
    assert_gate_parameters(bound.circuit.instructions, gates)
    logical = 2 + 3 * prod(shape)
    n = logical + 2
    before = torch.zeros(1 << n, dtype=torch.complex128)
    values = torch.arange(1 << (logical + 1), dtype=torch.int64)
    before[((values >> 1) << 2) | (values & 1)] = torch.complex(
        values.double() + 1, values.remainder(11).double() - 5
    )
    before /= before.norm()
    compute = (
        Instruction("x", (0,)),
        Instruction("ccx", (0, 1, work[0])),
        Instruction("x", (0,)),
    )
    ideal = (
        *compute,
        *(Instruction("ch", (work[0], word[0])) for word in words),
        *compute,
    )
    expected = forced_branch(before, ideal, ()) / 2**0.5
    for history in ((0,), (1,)):
        actual = forced_branch(before, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
        torch.testing.assert_close(
            actual, forced_branch(before, gates, history), atol=2e-12, rtol=0
        )


def test_symbolic_target_shape_keeps_original_construction_failure() -> None:
    import attrs
    import sympy
    from qualtran import Bloq, DecomposeTypeError, QAny, QBit
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )

    size = sympy.Symbol("size", integer=True, positive=True)

    @attrs.frozen
    class Declared:
        @property
        def signature(self) -> Any:
            return OriginalSignature(
                (
                    OriginalRegister("ctrl", QBit()),
                    OriginalRegister("word", QAny(3), (size,)),
                )
            )

    source = _MultiControlledFromSinglyControlled(
        (0, 1), type("OriginalDeclared", (Declared, Bloq), {})(), "ctrl"
    )
    registers = (
        Register("ctrl", quantum_bit_type()),
        Register("word", RegisterType("opaque", 3), (size,)),
    )
    # Both signatures retain metadata before either finite builder materializes elements.
    assert (
        source.signature.get_left("word").shape_symbolic == registers[1].shape_symbolic
    )
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        source.decompose_bloq()
    leaf = FunctionRecord("declared", (), registers=registers)
    with pytest.raises(ValueError, match="array dimensions"):
        multi_controlled_component_program(
            ProgramRecord((leaf,)), "ctrl", (0, 1), "declared"
        )


@pytest.mark.parametrize(
    "profile",
    (
        "uint",
        "bounded",
        "montgomery",
        "fixed-2",
        "fixed3",
        "field",
        "polynomial",
        "signed_fixed1",
        "signed_fixed-2",
        "empty_fixed0",
        "empty_fixed-2",
        "nonbinary3",
        "nonbinary5",
        "nonbinary9",
    ),
)
@pytest.mark.parametrize("shape", ((2,), (2, 2), (0, 2)))
@pytest.mark.parametrize("left_only", (False, True))
def test_full_array_partition_dependency_original_stream_and_logical_aliases(
    profile: str,
    shape: tuple[int, ...],
    left_only: bool,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
    from flagquantum.compiler.registers.layout.component_auto_partition import (
        auto_partition_component_program,
    )
    from tests.unit.test_component_auto_partition import (
        partition_gates,
        source_partition,
    )

    if profile.startswith("nonbinary"):
        from qualtran import QGF

        from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

        characteristic, degree = {
            "nonbinary3": (3, 1),
            "nonbinary5": (5, 1),
            "nonbinary9": (3, 2),
        }[profile]
        dtype = FiniteFieldEncoding(characteristic, degree).register_type()
        original_dtype = QGF(characteristic, degree)
    elif profile.startswith(("signed_fixed", "empty_fixed")):
        from qualtran import QFxp

        from flagquantum.core.ir.program.register_types import fixed_point_register_type

        signed = profile.startswith("signed_")
        fraction = int(
            profile.removeprefix("signed_fixed" if signed else "empty_fixed")
        )
        word_width = 3 if signed else 0
        dtype = fixed_point_register_type(word_width, fraction, signed)
        original_dtype = QFxp(word_width, fraction, signed=signed)
    else:
        dtype, original_dtype = packed_dtype(profile)
    assert isinstance(dtype.num_qubits, int)
    word_width = dtype.num_qubits
    positive, original_positive = shaped_positive(dtype, original_dtype, shape)
    count = prod(shape)
    width = word_width * count
    layout = AutoPartition(
        (("ctrl", (("ctrl", 1),)), ("packed", (("word", width),))), left_only=left_only
    )
    module = auto_partition_component_program(
        positive, layout, entry="positive", name="packed"
    )
    source = source_partition(original_positive, layout)
    assert verify_program_record(module) == ()
    assert normalize_trace(_native_trace(module, "packed")) == normalize_trace(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    words = tuple(
        tuple(range(1 + word_width * i, 1 + word_width * (i + 1))) for i in range(count)
    )
    packed = tuple(range(1, width + 1))
    bound = bind_register_program(module, ((0,), packed), entry="packed")
    outputs = ((0,), *words) if left_only else ((0,), packed)
    assert bound.outputs == outputs
    ports = {"ctrl": (0,), "packed": packed}
    desired = {"ctrl": (0,), "word": packed} if left_only else ports
    reference = partition_gates(source, original_positive, ports, desired, (), {})
    assert_gate_parameters(bound.circuit.instructions, reference)
    assert bound.events == ()
    counts = Counter(g.name for g in bound.circuit.instructions)
    active_count = count if word_width else 0
    assert counts == ({"ch": active_count} if active_count else {})
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == 0
    assert model.clifford == 4 * active_count and model.rotation == 2 * active_count
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == width + 1
    before = torch.arange(1 << (width + 2), dtype=torch.float64)
    before = torch.complex(before + 1, before.remainder(11) - 5)
    before /= before.norm()
    expected_gates = tuple(Instruction("ch", (0, word[0])) for word in words if word)
    expected_state = forced_branch(before, expected_gates, ())
    circuit = DynamicCircuit(width + 2).append(
        FieldOperation(width + 2, bound.circuit.instructions)
    )
    actual_state = forced_branch(before, circuit.to_ir().instructions, ())
    torch.testing.assert_close(actual_state, expected_state, atol=2e-12, rtol=0)
