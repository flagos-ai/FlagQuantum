"""Shared Fourier ownership, exact generic inverses and source-free execution."""

import base64
import os
import subprocess
import sys
from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives import fermionic_fourier as algorithm
from flagquantum.algorithms.primitives.hubbard_plaquette import HubbardPlaquette
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.legacy_component_library import (
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fermionic_fourier_factories import (
    fermionic_fourier_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.phase import fermionic_fourier as shared
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("k,n,flag", tuple(product((-2, 0, 2), (-3, 8), (False, True))))
def test_generic_adjoint_matches_source_graph_gates_and_full_complex_state(
    k: int, n: int, flag: bool
) -> None:
    from qualtran import Adjoint
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT

    source = Adjoint(TwoBitFFFT(k, n, is_adjoint=flag))
    records = tuple(
        resolve_component_program(
            _library(source, depth), 0, fermionic_fourier_factories()
        )
        for depth in (0, 1, 2)
    )
    # Original discovery gives the unused forward argument different wire IDs
    # at depth zero versus deeper libraries. Verify both complete definitions,
    # including all fields of their reachable helpers, independently of names.
    forward = shared.two_bit_fermionic_fourier_program(k, n)
    for candidate in records:
        assert candidate.functions[0] == records[0].functions[0]
        assert matches_component_definition(candidate, "main", records[0])
        roots = tuple(
            function.name
            for function in candidate.functions
            if function.name.removeprefix("legacy_component_").isdigit()
        )
        assert len(roots) == 1
        assert matches_component_definition(candidate, roots[0], forward)
    record = records[0]
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(record, ((2,), (0,)))
    expected_native = bind_register_program(
        shared.two_bit_fermionic_fourier_program(k, n, adjoint=True), ((2,), (0,))
    )
    assert bound.circuit == expected_native.circuit
    assert bound.events == ()
    assert bound.classical_outputs == ()
    assert bound.outputs == ((2,), (0,))
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "rz": 1,
        "sdg": 3,
        "h": 6,
        "cx": 3,
        "tdg": 1,
        "t": 1,
    }
    matrix = torch.as_tensor(
        source.decompose_bloq().tensor_contract(), dtype=torch.complex128
    )
    contiguous = bind_register_program(record, ((0,), (1,)))
    torch.testing.assert_close(
        logical_operator(contiguous.circuit, 2), matrix, atol=3e-12, rtol=0
    )
    initial = torch.tensor(
        [1 + 1j, 2 - 3j, -4j, 2, 3 + 2j, -1, 4 - 2j, 5j], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for column, amplitude in enumerate(initial):
        pair = ((column & 1) << 1) | ((column >> 2) & 1)
        for row in range(4):
            target = (column & 2) | ((row >> 1) & 1) | ((row & 1) << 2)
            expected[target] += matrix[row, pair] * amplitude
    circuit = Circuit(3, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(3, bound.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().reshape(-1), expected, atol=3e-12, rtol=0
    )


def test_algorithm_and_hubbard_use_the_replaced_shared_gate_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = shared.two_bit_fermionic_fourier_ir
    expected = HubbardPlaquette.make_on(modes=(3, 1, 0, 2), timestep=0.123)
    seen: list[tuple[int, int, int, int, bool]] = []

    def replacement(
        x: int, y: int, k: int, n: int, *, adjoint: bool = False
    ) -> tuple[Instruction, ...]:
        seen.append((x, y, k, n, adjoint))
        return original(x, y, k, n, adjoint=adjoint)

    monkeypatch.setattr(shared, "two_bit_fermionic_fourier_ir", replacement)
    assert algorithm.two_bit_fermionic_fourier_ir(2, 0, 2, 3) == original(2, 0, 2, 3)
    actual = HubbardPlaquette.make_on(modes=(3, 1, 0, 2), timestep=0.123)
    assert actual.instructions == expected.instructions
    assert len(seen) == 3


def test_constructor_executes_after_disabling_algorithm_program_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def disabled(k: int, n: int, *, adjoint: bool = False) -> ProgramRecord:
        raise AssertionError("Constructor must use the shared Compiler owner")

    monkeypatch.setattr(algorithm, "two_bit_fermionic_fourier_program", disabled)
    record = fermionic_fourier_factories()[
        "qualtran.bloqs.qft.two_bit_ffft.TwoBitFFFT"
    ]("main", None, {"k": 2, "n": 3})
    assert isinstance(record, ProgramRecord)
    assert bind_register_program(record, ((0,), (1,))).events == ()


def test_unknown_adjoint_requires_its_explicit_owner() -> None:
    child = FunctionRecord(
        "other",
        (),
        registers=tuple(shared.source_two_bit_fermionic_fourier_signature()),
    )
    with pytest.raises(SerializationError, match="adjoint context"):
        fermionic_fourier_factories()["qualtran._infra.adjoint.Adjoint"](
            "inverse", child.registers, {"subbloq": child}
        )
    seen: list[str] = []

    def external(
        name: str, registers: tuple[Register, ...] | None, arguments: object
    ) -> FunctionRecord:
        seen.append(name)
        raise NotImplementedError("Independent exact inverse owner")

    with pytest.raises(NotImplementedError, match="Independent"):
        fermionic_fourier_factories(adjoint_factory=external)[
            "qualtran._infra.adjoint.Adjoint"
        ]("inverse", child.registers, {"subbloq": child})
    assert seen == ["inverse"]


def test_captured_generic_inverse_resolves_without_sdk_or_algorithms_imports() -> None:
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT

    payload = serialize_component_library(_library(TwoBitFFFT(2, 3).adjoint(), 0))
    script = """import base64,sys,importlib.abc
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'qualtran','cirq','kickmix'} or fullname.startswith('flagquantum.algorithms'):
   raise AssertionError('Forbidden dependency: '+fullname)
sys.meta_path.insert(0,Reject())
from flagquantum.compiler.formats.legacy_component_library import deserialize_component_library
from flagquantum.compiler.formats.legacy_component_resolution import resolve_component_program
from flagquantum.compiler.formats.legacy_fermionic_fourier_factories import fermionic_fourier_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
library=deserialize_component_library(base64.b64decode(sys.stdin.buffer.read()))
record=resolve_component_program(library,0,fermionic_fourier_factories())
bound=bind_register_program(record,((2,),(0,)))
assert len(bound.circuit.instructions)==15 and bound.events==()
assert bound.outputs==((2,),(0,))
print('Source-free shared Fourier inverse verified')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=base64.b64encode(payload),
        capture_output=True,
        timeout=45,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
