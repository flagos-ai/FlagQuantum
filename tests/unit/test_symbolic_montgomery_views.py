"""Approved known-size Montgomery views retain symbolic metadata and raw phase."""

from importlib import import_module

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    import sympy
    import torch

    import_module("cirq")
    import_module("kickmix")
    from qualtran import BloqBuilder, BQUInt, QFxp, QMontgomeryUInt
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.bookkeeping import Join, Split
except ModuleNotFoundError as error:
    unavailable_reference(error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
    montgomery_register_type,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (3, 5))
def test_live_split_join_preserves_modulus_full_complex_state_and_no_work(
    width: int,
) -> None:
    p = sympy.Symbol("p")
    source_dtype = QMontgomeryUInt(width, p)
    native_dtype = montgomery_register_type(width, p)
    builder, values = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("x", source_dtype),))
    )
    bits = builder.add(Split(source_dtype), reg=values["x"])
    result = builder.add(Join(source_dtype), reg=bits)
    source = builder.finalize(x=result)
    operator = np.asarray(source.tensor_contract())
    np.testing.assert_array_equal(operator, np.eye(1 << width))
    factories = bookkeeping_factories()
    split = factories["qualtran.bloqs.bookkeeping.split.Split"](
        "split", None, {"dtype": native_dtype}
    )
    join = factories["qualtran.bloqs.bookkeeping.join.Join"](
        "join", None, {"dtype": native_dtype}
    )
    assert isinstance(split, FunctionRecord) and isinstance(join, FunctionRecord)
    program = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=native_dtype), (split, join)
    )
    bits = program.call_dict("split", {"reg": program.inputs["x"]})["reg"]
    x = program.call_dict("join", {"reg": bits})["reg"]
    record = program.finalize({"x": x})
    wires = tuple(reversed(range(1, width + 1)))
    bound = bind_register_program(record, (wires,))
    assert not bound.circuit.instructions and not bound.events
    assert bound.outputs == (wires,)
    registers = record.functions[0].registers
    assert registers is not None and registers[0].dtype == native_dtype
    values = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    state = torch.complex(values, 3 - values)
    state /= state.norm()
    circuit = DynamicCircuit(width + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=2, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), state.expand(2, -1), rtol=0, atol=2e-15
    )


@pytest.mark.parametrize("kind", ("Split", "Join"))
def test_source_refused_symbolic_interpretations_remain_refused(kind: str) -> None:
    p = sympy.Symbol("p")
    constructor = Split if kind == "Split" else Join
    factory = bookkeeping_factories()[
        "qualtran.bloqs.bookkeeping." + kind.lower() + "." + kind
    ]
    for source_dtype, native_dtype in (
        (QFxp(3, p), fixed_point_register_type(3, p)),
        (BQUInt(3, p), bounded_unsigned_register_type(3, p)),
    ):
        with pytest.raises(ValueError):
            constructor(source_dtype)
        with pytest.raises(SerializationError, match="symbolic source dtype"):
            factory("main", None, {"dtype": native_dtype})
    n = sympy.Symbol("n")
    with pytest.raises(ValueError):
        constructor(QMontgomeryUInt(n, p))
    with pytest.raises(SerializationError, match="symbolic source dtype"):
        factory("main", None, {"dtype": montgomery_register_type(n, p)})
