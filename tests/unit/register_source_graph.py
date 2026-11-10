"""Read original component connections without flattening physical lifetimes."""

from collections.abc import Callable
from typing import Any

import pytest

from flagquantum.core.ir import Instruction

SourcePorts = dict[str, tuple[int, ...]]
SourceResult = tuple[tuple[Instruction, ...], SourcePorts]
SourceLeaf = Callable[[Any, SourcePorts, SourcePorts], SourceResult | None]


def walk_source_graph(
    bloq: Any,
    inputs: SourcePorts,
    desired: SourcePorts,
    leaf: SourceLeaf,
    *,
    bindings: dict[Any, tuple[int, ...]] | None = None,
) -> SourceResult:
    """Bind original Soquet connections; only the caller binds leaf physical work."""
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import CompositeBloq, LeftDangle, Soquet
    from qualtran.bloqs.bookkeeping import Cast, Join, Split

    result = leaf(bloq, inputs, desired)
    if result is not None:
        return result
    if isinstance(bloq, (Cast, Split, Join)):
        assert set(inputs) == {"reg"}
        assert len(inputs["reg"]) == bloq.signature.get_left("reg").total_bits()
        assert len(inputs["reg"]) == bloq.signature.get_right("reg").total_bits()
        cirq = pytest.importorskip("cirq")

        register = bloq.signature.get_left("reg")
        qureg = np.asarray(
            [cirq.LineQubit(wire) for wire in inputs["reg"]], dtype=object
        ).reshape((*register.shape, int(register.bitsize)))
        operation, quregs = bloq.as_cirq_op(qubit_manager=None, reg=qureg)
        assert operation is None
        assert all(isinstance(wire, cirq.LineQubit) for wire in quregs["reg"].flat)
        outputs = {"reg": tuple(wire.x for wire in quregs["reg"].flat)}
        assert outputs == inputs
        assert all(outputs[key] == value for key, value in desired.items())
        return (), outputs
    cb = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
    mapping: dict[Any, tuple[int, ...]] = dict(bindings or {})

    def soqs(value: Any) -> tuple[Any, ...]:
        return tuple(np.asarray(value, dtype=object).flat)

    def bind(register: Any, values: tuple[Any, ...], wires: tuple[int, ...]) -> None:
        width = int(register.bitsize)
        assert len(wires) == len(values) * width
        for i, soq in enumerate(values):
            word = wires[i * width : (i + 1) * width]
            assert soq not in mapping or mapping[soq] == word
            mapping[soq] = word

    for register in cb.signature.lefts():
        bind(
            register,
            tuple(Soquet(LeftDangle, register, idx) for idx in register.all_idxs()),
            inputs[register.name],
        )
    final = cb.final_soqs()
    for register in cb.signature.rights():
        if register.name in desired:
            bind(register, soqs(final[register.name]), desired[register.name])
    gates: list[Instruction] = []
    for instance, old_inputs, old_outputs in cb.iter_bloqsoqs():
        child_inputs = {
            name: tuple(wire for soq in soqs(value) for wire in mapping[soq])
            for name, value in old_inputs.items()
        }
        child_outputs = {}
        right_registers = tuple(instance.bloq.signature.rights())
        for register, value in zip(right_registers, old_outputs, strict=True):
            values = soqs(value)
            if all(soq in mapping for soq in values):
                child_outputs[register.name] = tuple(
                    wire for soq in values for wire in mapping[soq]
                )
        child_gates, actual_outputs = walk_source_graph(
            instance.bloq, child_inputs, child_outputs, leaf
        )
        gates.extend(child_gates)
        for register, value in zip(right_registers, old_outputs, strict=True):
            bind(register, soqs(value), actual_outputs[register.name])
    outputs = {
        name: tuple(wire for soq in soqs(value) for wire in mapping[soq])
        for name, value in final.items()
    }
    assert all(outputs[name] == wires for name, wires in desired.items())
    return tuple(gates), outputs


def source_control_predicate_leaf(
    bloq: Any, inputs: SourcePorts, work: tuple[int, ...], bits: tuple[int, ...]
) -> SourceResult | None:
    """Bind the locked CtrlSpecAnd and its adjoint to one caller-owned workspace."""
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    is_adjoint = isinstance(getattr(bloq, "subbloq", None), CtrlSpecAnd)
    if not (isinstance(bloq, CtrlSpecAnd) or is_adjoint):
        return None
    outputs = {
        reg.name: inputs[reg.name]
        for reg in bloq.signature.rights()
        if reg.name in inputs
    }
    if not is_adjoint:
        outputs["target"] = work[-1:]
        if work[:-1]:
            outputs["junk"] = work[:-1]
    return (
        reference_arithmetic_gates(
            bloq,
            inputs,
            () if is_adjoint else work,
            dict(zip(work, bits, strict=True)),
            expected_outputs=outputs,
        ),
        outputs,
    )


def source_partition_leaf(
    bloq: Any, inputs: SourcePorts, desired: SourcePorts
) -> SourceResult | None:
    """Read the original packed Partition mapping without flattening call scopes."""
    from qualtran.bloqs.bookkeeping import Partition

    if not isinstance(bloq, Partition):
        return None
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    quregs = {
        reg.name: np.array([cirq.LineQubit(wire) for wire in inputs[reg.name]]).reshape(
            (*reg.shape, int(reg.bitsize))
        )
        for reg in bloq.signature.lefts()
    }
    operation, returned = bloq.as_cirq_op(qubit_manager=None, **quregs)
    assert operation is None
    outputs = {
        name: tuple(wire.x for wire in word.flat) for name, word in returned.items()
    }
    assert all(outputs[name] == wires for name, wires in desired.items())
    return (), outputs
