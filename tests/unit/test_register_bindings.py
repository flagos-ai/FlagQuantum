"""Replace pinned binding construction and traversal with native responsibilities."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import FrozenInstanceError
from itertools import product
from operator import is_
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import sympy

from flagquantum.core.ir.program.model import ValueRef
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.value_math import SymbolicInt

if TYPE_CHECKING:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import QCDType
    from qualtran._infra.quantum_graph import Soquet
    from qualtran._infra.registers import Register as SourceRegister

BindingT = TypeVar("BindingT", bound=Hashable)
Index: TypeAlias = tuple[int, ...]
BindingPort: TypeAlias = "Register | SourceRegister"


class BindingContract(Protocol[BindingT]):
    """One typed test interface exercised by both real binding implementations."""

    @staticmethod
    def binding(register: Register, index: int | Index = ()) -> BindingT: ...

    @staticmethod
    def index(binding: BindingT) -> Index: ...

    @staticmethod
    def name(binding: BindingT) -> str: ...

    @staticmethod
    def process(
        registers: Iterable[Register],
        bindings: Mapping[str, object],
        visit: Callable[[BindingT, BindingPort, Index], None],
    ) -> None: ...


pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)


class SourceContract:
    """Test-only source-backed stand-in for the internal binding contract."""

    @staticmethod
    def register(register: Register) -> SourceRegister:
        from qualtran._infra.data_types import (
            BQUInt,
            CBit,
            QAny,
            QBit,
            QInt,
            QIntOnesComp,
            QMontgomeryUInt,
            QUInt,
        )
        from qualtran._infra.registers import Register as SourceRegister
        from qualtran._infra.registers import Side

        dtype = register.dtype
        source_dtype: QCDType[int]
        if dtype.num_cbits:
            source_dtype = CBit()
        elif dtype.encoding == "bit":
            source_dtype = QBit()
        elif "bound" in dict(dtype.parameters):
            source_dtype = BQUInt(dtype.num_qubits, dict(dtype.parameters)["bound"])
        elif dtype.encoding == "montgomery":
            source_dtype = QMontgomeryUInt(
                dtype.num_qubits, dict(dtype.parameters).get("modulus")
            )
        else:
            source_dtype = {
                "opaque": QAny,
                "uint": QUInt,
                "signed": QInt,
                "ones_complement": QIntOnesComp,
            }[dtype.encoding](dtype.num_qubits)
        return SourceRegister(
            register.name,
            source_dtype,
            shape=register.shape_symbolic,
            side=getattr(Side, "__getitem__")(register.side.name),
        )

    @classmethod
    def binding(cls, register: Register, index: int | Index = ()) -> Soquet:
        from qualtran._infra.quantum_graph import LeftDangle, Soquet

        return Soquet(LeftDangle, cls.register(register), index)

    @staticmethod
    def index(binding: Soquet) -> Index:
        return binding.idx

    @staticmethod
    def name(binding: Soquet) -> str:
        return binding.reg.name

    @classmethod
    def process(
        cls,
        registers: Iterable[Register],
        bindings: Mapping[str, object],
        visit: Callable[[Soquet, BindingPort, Index], None],
    ) -> None:
        from qualtran._infra import composite_bloq

        # Negative cases deliberately send malformed values to the source validator.
        process: Callable[
            [
                Iterable[SourceRegister],
                Mapping[str, object],
                str,
                Callable[[Soquet, BindingPort, Index], None],
            ],
            None,
        ] = getattr(composite_bloq, "_process_soquets")
        return process(map(cls.register, registers), bindings, "consumer", visit)


class NativeContract:
    @staticmethod
    def binding(register: Register, index: int | Index = ()) -> RegisterBinding:
        from flagquantum.core.ir.program.register_bindings import RegisterBinding

        return RegisterBinding(ValueRef("entry", 0, "qubit"), register, index)

    @staticmethod
    def index(binding: RegisterBinding) -> Index:
        return binding.index

    @staticmethod
    def name(binding: RegisterBinding) -> str:
        return binding.register.name

    @staticmethod
    def process(
        registers: Iterable[Register],
        bindings: Mapping[str, object],
        visit: Callable[[RegisterBinding, BindingPort, Index], None],
    ) -> None:
        from flagquantum.compiler.registers.component_bindings import (
            process_register_bindings,
        )

        return process_register_bindings(registers, bindings, "consumer", visit)


@pytest.fixture(params=[SourceContract, NativeContract], ids=["source", "native"])
def backend(
    request: pytest.FixtureRequest,
) -> type[SourceContract] | type[NativeContract]:
    if request.param is SourceContract:
        return SourceContract
    assert request.param is NativeContract
    return NativeContract


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2), (N, 2)])
@pytest.mark.parametrize("index", [(), 0, 1, -1, (0, 0), (-1, 1), (2, 0)])
def test_index_conversion_and_validation(
    backend: type[BindingContract[BindingT]],
    shape: tuple[SymbolicInt, ...],
    index: int | Index,
) -> None:
    register = Register("word", RegisterType("uint", N), shape)
    try:
        expected = SourceContract.binding(register, index)
    except ValueError as error:
        diagnostic = (
            "is symbolic. Cannot get real-valued shape."
            if "is symbolic" in str(error)
            else str(error).split(" for ")[0]
        )
        with pytest.raises(ValueError, match=re.escape(diagnostic)):
            backend.binding(register, index)
    else:
        actual = backend.binding(register, index)
        assert backend.index(actual) == expected.idx
        assert backend.name(actual) == "word"
        assert hash(actual) == hash(backend.binding(register, index))


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
@pytest.mark.parametrize("width", [0, 1, 3, N])
def test_ordered_scalar_array_and_zero_width_binding(
    backend: type[BindingContract[BindingT]], shape: tuple[int, ...], width: SymbolicInt
) -> None:
    register = Register("words", RegisterType("uint", width), shape)
    values = np.empty(shape, dtype=object)
    for index in register.all_idxs():
        values[index] = backend.binding(register, index)
    seen: list[tuple[Index, str, Index]] = []
    result = backend.process(
        [register],
        {"words": values},
        lambda value, port, index: seen.append(
            (backend.index(value), port.name, index)
        ),
    )
    assert result is None
    assert seen == [(index, "words", index) for index in register.all_idxs()]


@pytest.mark.parametrize(
    "case", ["missing", "extra", "short", "long", "ragged", "wrong"]
)
def test_failures_preserve_partial_callback_order(
    backend: type[BindingContract[BindingT]], case: str
) -> None:
    register = Register("word", RegisterType("uint", 3), (2,))
    a, b = (backend.binding(register, (i,)) for i in range(2))
    bindings: dict[str, object] = {"word": [a, b]}
    expected_seen: list[int]
    error: type[Exception] | None
    expected_seen, error = [0, 1], None
    if case == "missing":
        bindings = {}
        expected_seen, error = [], ValueError
    elif case == "extra":
        bindings["extra"] = a
        error = ValueError
    elif case == "short":
        bindings["word"] = [a]
        expected_seen, error = [0], IndexError
    elif case == "long":
        bindings["word"] = [a, b, a]
    elif case == "ragged":
        bindings["word"] = [[a], [a, b]]
        expected_seen, error = [], ValueError
    else:
        bindings["word"] = [a, object()]
        expected_seen, error = [0], AssertionError
    seen: list[int] = []

    def visit(value: BindingT, port: BindingPort, index: Index) -> None:
        seen.append(index[0])

    if error is None:
        backend.process([register], bindings, visit)
    else:
        with pytest.raises(error):
            backend.process([register], bindings, visit)
    assert seen == expected_seen


@pytest.mark.parametrize(
    "source_kind,target_kind", tuple(product(["uint", "opaque", "signed"], repeat=2))
)
@pytest.mark.parametrize("widths", [(1, 1), (3, 3), (2, 3)])
def test_callback_precedes_loose_type_consistency(
    backend: type[BindingContract[BindingT]],
    source_kind: str,
    target_kind: str,
    widths: tuple[int, int],
) -> None:
    from qualtran._infra.data_types import check_dtypes_consistent

    origin = Register("origin", RegisterType(source_kind, widths[0]))
    target = Register("target", RegisterType(target_kind, widths[1]))
    binding = backend.binding(origin)
    expected = check_dtypes_consistent(
        SourceContract.register(origin).dtype, SourceContract.register(target).dtype
    )
    seen: list[tuple[BindingT, str, Index]] = []

    def visit(value: BindingT, port: BindingPort, index: Index) -> None:
        seen.append((value, port.name, index))

    if expected:
        backend.process([target], {"target": binding}, visit)
    else:
        with pytest.raises(ValueError, match="register dtypes are not consistent"):
            backend.process([target], {"target": binding}, visit)
    assert seen == [(binding, "target", ())]


def test_callback_failure_precedes_dtype_and_surplus_names(
    backend: type[BindingContract[BindingT]],
) -> None:
    origin = Register("origin", RegisterType("uint", 2))
    target = Register("target", RegisterType("uint", 3))
    binding = backend.binding(origin)
    seen: list[BindingT] = []

    def visit(value: BindingT, port: BindingPort, index: Index) -> None:
        seen.append(value)
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        backend.process([target], {"target": binding, "extra": object()}, visit)
    assert seen == [binding]


@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
def test_real_builder_initial_bindings_match_native_port_trace(
    backend: type[BindingContract[BindingT]], consumer: str
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder, _process_soquets
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.data_loading.qrom import QROM

    operation = (
        Add(QUInt(2), QUInt(3))
        if consumer == "add"
        else QROM.build_from_data(np.array([1, 2, 0, 3]), target_bitsizes=(2,))
    )
    _, initial = BloqBuilder.from_signature(operation.signature)
    expected: list[tuple[str, str, Index]] = []
    _process_soquets(
        operation.signature.lefts(),
        initial,
        "consumer",
        lambda value, port, index: expected.append((value.reg.name, port.name, index)),
    )
    registers = [
        Register(port.name, RegisterType("opaque", port.bitsize), port.shape)
        for port in operation.signature.lefts()
    ]
    bindings = {port.name: backend.binding(port) for port in reversed(registers)}
    actual: list[tuple[str, str, Index]] = []
    backend.process(
        registers,
        bindings,
        lambda value, port, index: actual.append(
            (backend.name(value), port.name, index)
        ),
    )
    assert actual == expected


def test_native_binding_preserves_ssa_identity_without_mutation() -> None:
    from flagquantum.core.ir.program.register_bindings import RegisterBinding

    register = Register("word", RegisterType("uint", 3), (2,))
    value = ValueRef("entry", 7, "qubit")
    binding = RegisterBinding(value, register, 0)
    assert binding.value is value
    assert binding.register is register
    assert binding.index == (0,)
    assert binding != RegisterBinding(ValueRef("entry", 8, "qubit"), register, 0)
    with pytest.raises(FrozenInstanceError):
        setattr(binding, "index", (1,))


@pytest.mark.parametrize(
    "case", ["empty", "scalar_array", "dtype_failure", "stale", "missing_after_empty"]
)
def test_binding_callback_can_enforce_single_consumption(
    backend: type[BindingContract[BindingT]], case: str
) -> None:
    register = Register("word", RegisterType("uint", 3))
    binding = backend.binding(register)
    available = {binding}
    effects: list[str] = []

    def consume(value: BindingT, port: BindingPort, index: Index) -> None:
        available.remove(value)
        effects.append(port.name)

    if case == "empty":
        empty = Register("empty", register.dtype, (0,))
        backend.process([empty], {"empty": object()}, consume)
        assert effects == []
        assert available == {binding}
    elif case == "scalar_array":
        with pytest.raises(AssertionError):
            backend.process([register], {"word": [binding]}, consume)
        assert available == {binding}
        assert effects == []
    elif case == "missing_after_empty":
        empty = Register("empty", register.dtype, (0,))
        with pytest.raises(ValueError, match="named `empty`"):
            backend.process([empty], {}, consume)
        assert effects == []
    else:
        target = Register("target", RegisterType("uint", 4))
        with pytest.raises(ValueError, match="register dtypes are not consistent"):
            backend.process([target], {"target": binding}, consume)
        assert available == set()
        assert effects == ["target"]
        if case == "stale":
            with pytest.raises(KeyError):
                backend.process([register], {"word": binding}, consume)
            assert effects == ["target"]


@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
def test_direction_is_selected_by_caller_not_binding_traversal(
    backend: type[BindingContract[BindingT]], side: str
) -> None:
    from flagquantum.core.ir.program.registers import RegisterSide
    from flagquantum.core.ir.program.signatures import Signature

    register = Register("word", RegisterType("uint", 3), side=RegisterSide[side])
    signature = Signature([register])
    binding = backend.binding(register)
    ports: Iterable[Register]
    for ports in (signature.lefts(), signature.rights()):
        ports = tuple(ports)
        seen: list[str] = []

        def visit(
            value: BindingT,
            port: BindingPort,
            index: Index,
            seen: list[str] = seen,
        ) -> None:
            seen.append(port.name)

        backend.process(
            ports,
            {"word": binding} if ports else {},
            visit,
        )
        assert seen == [port.name for port in ports]


def test_shared_type_policy_is_called_after_actual_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers import component_bindings

    register = Register("word", RegisterType("uint", 3))
    binding = NativeContract.binding(register)
    events: list[str | tuple[RegisterType, RegisterType]] = []

    def reject(left: RegisterType, right: RegisterType) -> bool:
        events.append((left, right))
        return False

    monkeypatch.setattr(component_bindings, "check_register_types_consistent", reject)
    with pytest.raises(component_bindings.RegisterBindingError):
        NativeContract.process(
            [register], {"word": binding}, lambda *args: events.append("visited")
        )
    assert events == ["visited", (register.dtype, register.dtype)]


@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
@pytest.mark.parametrize("factory_created", [False, True], ids=["manual", "created"])
def test_bindings_feed_append_with_exact_measured_channels_and_resources(
    backend: type[BindingContract[BindingT]], consumer: str, factory_created: bool
) -> None:
    from collections import Counter

    import torch
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from flagquantum.compiler.registers.arithmetic.addition import add_ir
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source: Bloq
    groups: tuple[tuple[int, ...], ...]
    work: tuple[int, ...]
    if consumer == "add":
        registers = Signature(
            [
                Register("a", RegisterType("uint", 2)),
                Register("b", RegisterType("uint", 3)),
            ]
        )
        source = Add(QUInt(2), QUInt(3))
        groups, work = ((0, 1), (2, 3, 4)), (5, 6)
    else:
        table = QromTable.from_data([1, 3, 0, 2], target_bitsizes=(2,))
        registers = table.signature
        source = QROM.build_from_data(np.array([1, 3, 0, 2]), target_bitsizes=(2,))
        groups, work = ((0, 1), (2, 3)), (4,)

    bindings: dict[str, object]
    if factory_created:
        from tests.unit.test_register_binding_creation import (
            create_native,
            create_source,
        )

        create = create_source if is_(backend, SourceContract) else create_native
        bindings = {port.name: create(port) for port in registers.lefts()}
    else:
        bindings = {port.name: backend.binding(port) for port in registers.lefts()}
    physical: dict[object, tuple[int, ...]] = {
        bindings[port.name]: wires
        for port, wires in zip(registers.lefts(), groups, strict=True)
    }
    ports: dict[str, tuple[int, ...]] = {}
    backend.process(
        registers.lefts(),
        dict(reversed(list(bindings.items()))),
        lambda value, port, index: ports.update({port.name: physical[value]}),
    )
    slots = tuple(range(len(work)))
    gates = (
        add_ir(ports["a"], ports["b"], work, slots)
        if consumer == "add"
        else table.bind_packed(
            tuple(w for words in ports.values() for w in words), work, slots
        )
    )
    original = reference_arithmetic_gates(
        source,
        ports,
        # Add reverses qalloc order into its low-bit-first carry register.
        tuple(reversed(work)) if consumer == "add" else work,
        dict(zip(work, slots, strict=True)),
        cirq_style=True,
    )
    assert_source(gates, original)
    emitted = Counter(gate.name for gate in gates)
    model = get_cost_value(source, QECGatesCost())
    assert emitted["t"] + emitted["tdg"] == model.total_t_count()
    assert emitted["measure"] == model.measurement
    data_width = sum(map(len, groups))
    width = data_width + len(work)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for word in range(1 << data_width):
        amplitude = complex((word % 5) - 2, (word % 3) - 1)
        state[word << len(work)] = torch.as_tensor(
            amplitude, dtype=state.dtype, device=state.device
        )
        if consumer == "add":
            left, right = word >> 3, word & 7
            output = (left << 3) | ((left + right) & 7)
        else:
            address, target = word >> 2, word & 3
            output = (address << 2) | (target ^ [1, 3, 0, 2][address])
        expected[output << len(work)] = torch.as_tensor(
            amplitude, dtype=expected.dtype, device=expected.device
        )
    expected /= state.norm()
    state /= state.norm()
    events = emitted["measure"]
    for history in product((0, 1), repeat=events):
        actual = forced_branch(state, gates, history)
        reference = forced_branch(state, original, history)
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), rtol=0, atol=2e-12
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << len(work))[:, 1:]) == 0
