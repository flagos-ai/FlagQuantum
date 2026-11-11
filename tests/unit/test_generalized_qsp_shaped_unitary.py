"""Packed and row-major shaped U ports preserve source GQSP semantics."""

from collections.abc import Mapping
from itertools import product
from typing import Any

import cirq
import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.generalized_qsp import (
    GeneralizedQSP,
    UnitaryEmitter,
)
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import (
    ControlAdder,
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase
from tests.unit.test_generalized_qsp_entangling_unitary import (
    _add_cnot,
    assert_costs,
    entangling_unitary,
)
from tests.unit.test_generalized_qsp_stored_unitary import (
    _add_inverse_phase,
    source_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
LAYOUTS = ((2, ()), (1, (1, 2)), (2, (1, 2)))


def neutral_rebinding(name: str) -> ControlAdder:
    """Original bookkeeping control keeps the predicate separate and unchanged."""

    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_dict(name, values)
        return control, tuple(outputs.values())

    return add


def shaped_unitary(
    width: int, shape: tuple[int, ...]
) -> tuple[dict[tuple[int | None, bool], ProgramRecord], Any]:
    """Expose packed or shaped THRU words around the accepted entangling child."""
    from qualtran import BloqBuilder, QBit, QUInt
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    base, u = entangling_unitary(1)
    dtype = quantum_bit_type() if width == 1 else unsigned_register_type(width)
    signature = Signature([Register("word", dtype, shape)])
    definitions = (
        *(f for f in base[None, False].functions if f.name != "main"),
        split_word_function("unpack", dtype),
        join_word_function("pack", dtype),
    )
    native = ComponentProgramBuilder(signature, definitions)
    bb, source = BloqBuilder.from_signature(
        SourceSignature(
            [
                SourceRegister(
                    "word", QBit() if width == 1 else QUInt(width), shape=shape
                )
            ]
        )
    )
    native_words = np.asarray(native.inputs["word"], dtype=object)
    source_words = np.asarray(source["word"], dtype=object)
    native_bits, source_bits = [], []
    for index in np.ndindex(shape):
        if width == 1:
            native_bits.append(native_words[index])
            source_bits.append(source_words[index])
        else:
            unpacked = native.call_dict("unpack", {"reg": native_words[index]})["reg"]
            assert isinstance(unpacked, np.ndarray)
            native_bits.extend(unpacked)
            source_bits.extend(bb.split(source_words[index]))
    for i in range(0, len(native_bits), 2):
        outputs = native.call_dict(
            "entangle", {"a": native_bits[i], "b": native_bits[i + 1]}
        )
        native_bits[i : i + 2] = [outputs["a"], outputs["b"]]
        outputs = bb.add_d(u, a=source_bits[i], b=source_bits[i + 1])
        source_bits[i : i + 2] = [outputs["a"], outputs["b"]]
    native_output = np.empty(shape, dtype=object)
    source_output = np.empty(shape, dtype=object)
    for ordinal, index in enumerate(np.ndindex(shape)):
        if width == 1:
            native_output[index] = native_bits[ordinal]
            source_output[index] = source_bits[ordinal]
        else:
            native_output[index] = native.call_dict(
                "pack",
                {
                    "reg": np.array(
                        native_bits[ordinal * width : (ordinal + 1) * width],
                        dtype=object,
                    )
                },
            )["reg"]
            source_output[index] = bb.join(
                source_bits[ordinal * width : (ordinal + 1) * width], dtype=QUInt(width)
            )
    forward = native.finalize({"word": native_output if shape else native_output[()]})
    inverse = adjoint_component_hierarchy(
        forward, {"h": "h", "cn": "cn", "p": "pi", "unpack": "pack", "pack": "unpack"}
    )
    programs: dict[tuple[int | None, bool], ProgramRecord] = {
        (None, False): forward,
        (None, True): inverse,
    }
    for adjoint, record in ((False, forward), (True, inverse)):
        for value in (0, 1):
            programs[value, adjoint] = control_component_via_predicate(
                record,
                {
                    "h": add_h,
                    "cn": _add_cnot,
                    "pi" if adjoint else "p": (
                        _add_inverse_phase if adjoint else add_phase
                    ),
                    "unpack": neutral_rebinding("unpack"),
                    "pack": neutral_rebinding("pack"),
                },
                (Register("ctrl", quantum_bit_type()),),
                (ControlRegister(1, (value,)),),
            )
    return programs, bb.finalize(word=source_output if shape else source_output[()])


def emitter(
    programs: dict[tuple[int | None, bool], ProgramRecord], width: int
) -> UnitaryEmitter:
    def emit(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        words = tuple(
            ports["word"][i : i + width] for i in range(0, len(ports["word"]), width)
        )
        inputs = words if value is None else ((signal,), *words)
        bound = bind_register_program(programs[value, adjoint], inputs, clean=work)
        assert bound.outputs == inputs
        return bound.circuit.instructions

    return emit


def binding(
    width: int, shape: tuple[int, ...]
) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    data_width = width * int(np.prod(shape))
    physical = tuple(reversed(range(1, data_width + 2)))
    signal, data = physical[0], physical[1:]
    order = (signal, *data, 0, data_width + 2)
    return signal, data, (0,), order


@pytest.mark.parametrize("width,shape", LAYOUTS)
@pytest.mark.parametrize("value", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_typed_shaped_control_and_adjoint_source_instructions(
    width: int, shape: tuple[int, ...], value: int | None, adjoint: bool
) -> None:
    from qualtran import CtrlSpec

    from tests.unit.register_source_graph import SourcePorts

    programs, source = shaped_unitary(width, shape)
    signal, data, work, _ = binding(width, shape)
    if adjoint:
        source = source.adjoint()
    ports: SourcePorts = {"word": data}
    if value is not None:
        source = source.controlled(CtrlSpec(cvs=value))
        ports["ctrl"] = (signal,)
    gates = emitter(programs, width)(signal, {"word": data}, work, value, adjoint)
    assert gates == source_gates(source, ports, len(data) // 2)
    assert tuple(programs[None, False].functions[0].registers or ()) == (
        Register(
            "word",
            quantum_bit_type() if width == 1 else unsigned_register_type(width),
            shape,
        ),
    )


@pytest.mark.parametrize("width,shape", LAYOUTS)
@pytest.mark.parametrize("length", [2, 3])
@pytest.mark.parametrize("negative_power", [0, 1, 4])
def test_full_gqsp_typed_words_all_branches_amplitudes_and_work_zero(
    width: int, shape: tuple[int, ...], length: int, negative_power: int
) -> None:
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial

    programs, u = shaped_unitary(width, shape)
    signal, data, work, order = binding(width, shape)
    p = tuple(random_qsp_polynomial(length, random_state=np.random.RandomState(17)))
    source = Source.from_qsp_polynomial(u, p, negative_power=negative_power)
    native = GeneralizedQSP.from_qsp_polynomial(
        (("word", len(data)),),
        emitter(programs, width),
        p,
        negative_power=negative_power,
    )
    gates = native.emit(signal, data, work)
    pairs = len(data) // 2
    assert gates == source_gates(source, {"signal": (signal,), "word": data}, pairs)
    measurements = pairs * (length - 1)
    assert sum(g.name == "measure" for g in gates) == measurements
    assert_costs(source, length, negative_power, pairs, system_width=len(data))
    reference = cirq.unitary(source)
    dimension = 1 << (len(data) + 1)
    coherent = np.array([complex(i % 5 - 2, i % 7 - 3) for i in range(dimension)])
    coherent /= np.linalg.norm(coherent)
    for initial in (*np.eye(dimension, dtype=complex), coherent):
        expected = np.einsum("ij,j->i", reference, initial, optimize=False)
        assert np.isfinite(reference).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(
            expected,
            (torch.tensor(reference) @ torch.tensor(initial)).numpy(),
            atol=3e-15,
            rtol=0,
        )
        logical = np.kron(initial, [1, 0, 0, 0]).reshape((2,) * len(order))
        state = torch.tensor(logical.transpose(tuple(np.argsort(order))).flatten())
        for history in product((0, 1), repeat=measurements):
            actual = (
                forced_branch(state, gates, history)
                .numpy()
                .reshape((2,) * len(order))
                .transpose(order)
                .reshape(dimension, 4)
            )
            np.testing.assert_allclose(
                actual[:, 0],
                expected * 2 ** (-measurements / 2),
                atol=1e-13,
                rtol=0,
            )
            np.testing.assert_array_equal(actual[:, 1:], 0)
        circuit = DynamicCircuit(len(order), inputs=state, dtype=torch.complex128)
        circuit.append(native.make_on(signal=signal, target=data, workspace=work))
        assert circuit.to_ir().instructions == gates
        actual = (
            run_dynamic(circuit, shots=1, seed=23)
            .final_states[0]
            .detach()
            .numpy()
            .reshape((2,) * len(order))
            .transpose(order)
            .reshape(dimension, 4)
        )
        np.testing.assert_allclose(actual[:, 0], expected, atol=2e-13, rtol=0)
        np.testing.assert_array_equal(actual[:, 1:], 0)


@pytest.mark.parametrize("width,shape", LAYOUTS)
def test_expansion_replacement_and_wrong_binding_rejection(
    width: int, shape: tuple[int, ...]
) -> None:
    programs, _ = shaped_unitary(width, shape)
    expanded = {
        key: expand_component_calls(record)[0] for key, record in programs.items()
    }
    signal, data, work, _ = binding(width, shape)
    for key in programs:
        assert emitter(programs, width)(signal, {"word": data}, work, *key) == emitter(
            expanded, width
        )(signal, {"word": data}, work, *key)
    with pytest.raises(ValueError, match="widths"):
        emitter(programs, width)(signal, {"word": data[:-1]}, work, 1, False)
    with pytest.raises(ValueError, match="workspace"):
        emitter(programs, width)(signal, {"word": data}, (), 1, True)
