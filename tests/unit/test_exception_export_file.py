"""Check frozen source exception exports against native handling boundaries."""

import hashlib
from importlib import import_module
from pathlib import Path

import pytest

from examples.algorithms.register_graphs import parallel_program
from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import (
    NoComponentExpansionError,
    expand_component_calls,
    expand_component_calls_once,
)
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, FlagQuantumError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/exception/__init__.py"
)


def test_all_frozen_exception_exports_and_real_triggers() -> None:
    contents = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
        == "7603935118da18e88de4b01ef497291de111e762"
    )
    source = import_module("qualtran.exception")
    bloq = import_module("qualtran._infra.bloq")
    composite = import_module("qualtran._infra.composite_bloq")
    api = import_module("qualtran")
    example = import_module("qualtran.bloqs.for_testing").TestAtom()
    definitions = {
        "DecomposeTypeError": (bloq, TypeError),
        "DecomposeNotImplementedError": (bloq, NotImplementedError),
        "BloqError": (composite, ValueError),
        "DidNotFlattenAnythingError": (composite, ValueError),
    }
    assert tuple(source.__all__) == tuple(definitions)
    for name, (owner, base) in definitions.items():
        exported = getattr(source, name)
        assert exported is getattr(owner, name)
        assert issubclass(exported, base)
    with pytest.raises(source.DecomposeTypeError, match="atomic"):
        example.decompose_bloq()
    missing = type("MissingRule", (bloq.Bloq,), {"signature": example.signature})()
    with pytest.raises(source.DecomposeNotImplementedError, match="decomposition"):
        missing.decompose_bloq()
    with pytest.raises(source.DidNotFlattenAnythingError):
        example.as_composite_bloq().flatten_once()
    builder, _ = api.BloqBuilder.from_signature(example.signature)
    with pytest.raises(source.BloqError, match="named `q`"):
        builder.add(example)


def test_native_atomic_admission_and_no_change_termination() -> None:
    record = parallel_program()
    leaf = next(function for function in record.functions if function.name == "atom")
    atomic = ProgramRecord(functions=(leaf,))
    admitted, labels = expand_component_root_for_tensor(
        atomic, lambda _: False, entry="atom", full_flatten=True
    )
    assert labels == (0,)
    assert admitted.functions[0].blocks[0].operations[0].callee == "atom"
    assert admitted.functions[-1] is leaf

    with pytest.raises(NoComponentExpansionError, match="no selected"):
        expand_component_calls_once(record)
    unchanged, labels = expand_component_calls(record)
    assert unchanged is record
    assert labels == (0, 1, 2, 3, 4)


def test_native_binding_failure_and_capability_catch_contract() -> None:
    record = parallel_program()
    leaf = next(function for function in record.functions if function.name == "atom")
    builder = ComponentProgramBuilder(
        Signature((Register("q", quantum_bit_type()),)), (leaf,)
    )
    with pytest.raises(RegisterBindingError, match="named `q`") as caught:
        builder.call_dict("atom", {})
    assert isinstance(caught.value, ValueError)

    circuit = DynamicCircuit(1)
    circuit.measure(0, classical_bit=0)
    instructions = circuit.to_ir().instructions
    with pytest.raises(
        CapabilityError, match="Cannot invert instruction 'measure'"
    ) as capability:
        circuit.adjoint()
    assert isinstance(capability.value, NotImplementedError)
    assert isinstance(capability.value, FlagQuantumError)
    assert circuit.to_ir().instructions == instructions
