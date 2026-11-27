"""Pinned call/return contract to be reused by the native Program builder.

The source-backed fake deliberately executes the unmodified builder. Passing
these scenarios establishes requirements, not native graph implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

import attrs

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from tests.unit.test_register_bindings import SourceContract

if TYPE_CHECKING:
    from qualtran._infra.composite_bloq import SoquetT
    from qualtran._infra.quantum_graph import Soquet
    from qualtran._infra.registers import Signature as SourceSignature

Identity: TypeAlias = tuple[str | int, str, tuple[int, ...]]
CallForm: TypeAlias = Literal["dict", "tuple", "auto"]

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def port(
    name: str = "word",
    width: SymbolicInt = 3,
    shape: tuple[SymbolicInt, ...] = (),
    side: RegisterSide = RegisterSide.THRU,
) -> Register:
    return Register(name, RegisterType("uint", width), shape, side)


class SourceCalls:
    """Replaceable builder boundary with normalized observations only.

    Values remain opaque to scenarios. Normalized identities expose producer,
    port and index, so sharing a producer cannot collapse shaped elements.
    """

    def __init__(self, signature: Signature) -> None:
        from qualtran._infra.composite_bloq import BloqBuilder, BloqError

        self.error_type = BloqError
        self.builder, self.inputs = BloqBuilder.from_signature(
            self.signature(signature)
        )

    @staticmethod
    def signature(signature: Iterable[Register]) -> SourceSignature:
        from qualtran._infra.registers import Signature as SourceSignature

        return SourceSignature(map(SourceContract.register, signature))

    @staticmethod
    def identity(binding: Soquet) -> Identity:
        from qualtran._infra.quantum_graph import BloqInstance, LeftDangle, RightDangle

        owner = binding.binst
        producer: str | int
        if owner == LeftDangle:
            producer = "entry"
        elif owner == RightDangle:
            producer = "return"
        else:
            assert isinstance(owner, BloqInstance)
            producer = owner.i
        return producer, binding.reg.name, binding.idx

    @classmethod
    def identities(cls, collection: object) -> tuple[Identity, ...]:
        return tuple(cls.identity(value) for value in np.asarray(collection).flat)

    @property
    def available(self) -> set[Identity]:
        return {self.identity(value) for value in self.builder._available}

    @property
    def instances(self) -> tuple[int, ...]:
        return tuple(sorted(instance.i for instance in self.builder._binsts))

    @property
    def connections(self) -> tuple[tuple[Identity, Identity], ...]:
        return tuple(
            (self.identity(connection.left), self.identity(connection.right))
            for connection in self.builder._cxns
        )

    def call(
        self,
        name: str,
        signature: Iterable[Register],
        bindings: Mapping[str, object],
        form: CallForm = "dict",
    ) -> Any:
        from qualtran._infra.bloq import Bloq

        @attrs.frozen
        class Leaf(Bloq):
            label: str
            ports: tuple[Register, ...]

            @cached_property
            def signature(self) -> SourceSignature:
                return SourceCalls.signature(self.ports)

        leaf = Leaf(name, tuple(signature))
        # These raw SDK entry points validate malformed inputs and choose arity.
        add = {
            "dict": getattr(self.builder, "add_d"),
            "tuple": getattr(self.builder, "add_t"),
            "auto": getattr(self.builder, "add"),
        }[form]
        return add(leaf, **bindings)

    def finalize(self, bindings: Mapping[str, object]) -> Self:
        from qualtran._infra.composite_bloq import CompositeBloq

        program = getattr(self.builder, "finalize")(**bindings)
        assert isinstance(program, CompositeBloq)
        self.program = program
        return self

    def final_bindings(self) -> dict[str, SoquetT[np.object_]]:
        return self.program.final_soqs()


@pytest.fixture(params=[SourceCalls], ids=["source-fake"])
def calls(request: pytest.FixtureRequest) -> type[SourceCalls]:
    # The native builder replaces this factory, without changing scenarios.
    factory = request.param
    assert isinstance(factory, type) and issubclass(factory, SourceCalls)
    return factory


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
@pytest.mark.parametrize("width", [0, 1, 3])
def test_through_call_versions_every_index_and_preserves_port_order(
    calls: type[SourceCalls], shape: tuple[int, ...], width: int
) -> None:
    signature = Signature([port("z", width, shape), port("a", 2)])
    builder = calls(signature)
    before = builder.available
    result = builder.call("identity", signature, dict(reversed(builder.inputs.items())))
    assert list(result) == ["z", "a"]
    expected = {
        (0, reg.name, index) for reg in signature.rights() for index in reg.all_idxs()
    }
    assert builder.available == expected
    assert not before & expected
    assert builder.instances == (0,)
    assert len(builder.connections) == len(before)
    for reg in signature:
        expected_indices = tuple(reg.all_idxs())
        assert (
            tuple(item[2] for item in builder.identities(result[reg.name]))
            == expected_indices
        )
    graph = builder.finalize(result)
    assert builder.available == set()
    assert len(graph.connections) == 2 * len(before)
    assert builder.connections[-1][1] == ("return", "a", ())


@pytest.mark.parametrize("form", ["dict", "tuple", "auto"])
@pytest.mark.parametrize("output_count", [0, 1, 2])
def test_output_arity_and_left_only_consumption(
    calls: type[SourceCalls], form: CallForm, output_count: int
) -> None:
    inputs = [port("input", side=RegisterSide.LEFT)]
    outputs = [port(f"out{i}", side=RegisterSide.RIGHT) for i in range(output_count)]
    builder = calls(Signature([*inputs, *outputs]))
    result = builder.call(
        "replace", Signature([*inputs, *outputs]), builder.inputs, form
    )
    if form == "dict":
        assert list(result) == [reg.name for reg in outputs]
        returned = result
    elif form == "auto" and output_count == 0:
        assert result is None
        returned = {}
    else:
        values = (result,) if form == "auto" and output_count == 1 else result
        assert len(values) == output_count
        returned = dict(zip((reg.name for reg in outputs), values, strict=True))
    builder.finalize(returned)
    assert builder.available == set()
    assert len(builder.connections) == 1 + output_count


def test_opposite_sides_can_reuse_name_with_different_widths(
    calls: type[SourceCalls],
) -> None:
    signature = Signature(
        [
            port("word", 2, side=RegisterSide.LEFT),
            port("word", 5, side=RegisterSide.RIGHT),
        ]
    )
    builder = calls(signature)
    result = builder.call("change_width", signature, builder.inputs)
    assert builder.identities(result["word"]) == ((0, "word", ()),)
    builder.finalize(result)
    assert builder.connections == (
        (("entry", "word", ()), (0, "word", ())),
        ((0, "word", ()), ("return", "word", ())),
    )


@pytest.mark.parametrize("failure", ["missing", "extra", "dtype", "duplicate", "short"])
def test_failed_call_keeps_instance_and_consumed_prefix(
    calls: type[SourceCalls], failure: str
) -> None:
    signature = Signature([port("first"), port("second")])
    builder = calls(signature)
    bindings: dict[str, object] = dict(builder.inputs)
    callee = signature
    consumed = 1
    exception: type[Exception] = builder.error_type
    if failure == "missing":
        bindings.pop("second")
    elif failure == "extra":
        bindings["surplus"] = bindings["first"]
        consumed = 2
    elif failure == "dtype":
        callee = Signature([port("first", 4), port("second")])
    elif failure == "duplicate":
        bindings["second"] = bindings["first"]
    else:
        callee = Signature([port("first"), port("second", shape=(2,))])
        bindings["second"] = [bindings["second"]]
        consumed = 2
        exception = IndexError
    with pytest.raises(exception):
        builder.call("failing", callee, bindings)
    initial_instances = builder.instances
    assert initial_instances == (0,)
    assert len(builder.connections) == consumed
    assert all(target[0] == 0 for _, target in builder.connections)
    assert builder.available == ({("entry", "second", ())} if consumed == 1 else set())
    # A failed call still reserves its index. A successful no-port call is next.
    assert builder.call("empty", Signature([]), {}) == {}
    assert builder.instances == (0, 1)


def test_reusing_consumed_output_fails_before_adding_another_connection(
    calls: type[SourceCalls],
) -> None:
    signature = Signature([port()])
    builder = calls(signature)
    first = builder.call("first", signature, builder.inputs)
    second = builder.call("second", signature, first)
    with pytest.raises(builder.error_type, match="not an available"):
        builder.call("stale", signature, first)
    assert builder.instances == (0, 1, 2)
    assert len(builder.connections) == 2
    assert builder.available == {(1, "word", ())}
    builder.finalize(second)


@pytest.mark.parametrize(
    "failure", ["missing", "extra", "dtype", "duplicate", "unused"]
)
def test_failed_return_preserves_prefix_and_rejects_unconsumed_values(
    calls: type[SourceCalls], failure: str
) -> None:
    signature = Signature([port("first"), port("second")])
    if failure == "dtype":
        signature = Signature(
            [
                port("first", 3, side=RegisterSide.LEFT),
                port("first", 4, side=RegisterSide.RIGHT),
                port("second"),
            ]
        )
    if failure == "unused":
        signature = Signature([port("first"), port("second", side=RegisterSide.LEFT)])
    builder = calls(signature)
    bindings = dict(builder.inputs)
    consumed = 1
    if failure in {"missing", "unused"}:
        bindings.pop("second")
    elif failure == "extra":
        bindings["surplus"] = bindings["first"]
        consumed = 2
    elif failure == "duplicate":
        bindings["second"] = bindings["first"]
    with pytest.raises(builder.error_type):
        builder.finalize(bindings)
    assert builder.instances == ()
    assert len(builder.connections) == consumed
    assert all(target[0] == "return" for _, target in builder.connections)
    assert builder.available == ({("entry", "second", ())} if consumed == 1 else set())
    with pytest.raises(builder.error_type, match="not an available"):
        builder.finalize(bindings)
    assert len(builder.connections) == consumed


def test_explicit_empty_return_accepts_empty_or_consumed_lefts(
    calls: type[SourceCalls],
) -> None:
    for signature in [Signature([]), Signature([port(side=RegisterSide.LEFT)])]:
        builder = calls(signature)
        if builder.inputs:
            builder.call("consume", signature, builder.inputs)
        graph = builder.finalize({})
        assert builder.available == set()
        assert graph.final_bindings() == {}


def test_source_empty_shaped_right_has_no_dangle_connection() -> None:
    calls = SourceCalls
    signature = Signature([port(shape=(0, 2))])
    builder = calls(signature)
    result = builder.call("empty_array", signature, builder.inputs)
    assert result["word"].shape == (0, 2)
    graph = builder.finalize(result)
    assert builder.available == set()
    assert builder.connections == ()
    # Source final_soqs omits even the declared key when RightDangle is absent.
    assert graph.final_bindings() == {}


def test_source_classical_binding_is_also_single_consumption(
    calls: type[SourceCalls],
) -> None:
    classical = Register("bit", RegisterType("classical_bit", 0, 1))
    signature = Signature([classical])
    builder = calls(signature)
    result = builder.call("classical", signature, builder.inputs)
    with pytest.raises(builder.error_type, match="not an available"):
        builder.call("classical_again", signature, builder.inputs)
    builder.finalize(result)


@pytest.mark.parametrize("widths", [(2, 3), (0, 2)])
def test_actual_add_signature_consumes_both_ports_and_exposes_successors(
    calls: type[SourceCalls], widths: tuple[int, int]
) -> None:
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.addition import Add

    a_width, b_width = widths
    source = Add(QUInt(a_width), QUInt(b_width)).signature
    signature = Signature(
        [
            port(reg.name, reg.dtype.num_qubits, reg.shape, RegisterSide[reg.side.name])
            for reg in source
        ]
    )
    builder = calls(signature)
    first = builder.call("add", signature, builder.inputs)
    result = builder.call("add", signature, first)
    assert list(result) == [reg.name for reg in source.rights()]
    builder.finalize(result)
    assert len(builder.connections) == 3 * len(list(signature.lefts()))


def test_array_alias_rejected_at_second_row_major_use(calls: type[SourceCalls]) -> None:
    signature = Signature([port(shape=(2, 2))])
    builder = calls(signature)
    assert isinstance(builder.inputs["word"], np.ndarray)
    values = builder.inputs["word"].copy()
    values[1, 0] = values[0, 0]
    with pytest.raises(builder.error_type, match="not an available"):
        builder.call("alias", signature, {"word": values})
    assert [target[2] for _, target in builder.connections] == [(0, 0), (0, 1)]
    assert builder.available == {("entry", "word", (1, 0)), ("entry", "word", (1, 1))}
