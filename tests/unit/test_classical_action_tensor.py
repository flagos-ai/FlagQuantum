"""Phase-free classical tensors match the pinned source, including its boundaries."""

from collections import Counter
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.bit_encoding import (
    bit_array_from_bits,
    bit_array_to_bits,
    bit_from_bits,
)
from flagquantum.simulation.integer_encoding import (
    integer_array_from_bits,
    integer_array_to_bits,
    integer_from_bits,
)
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def decode(register: Register, bits: NDArray[Any]) -> object:
    if register.dtype.encoding == "bit":
        return (
            bit_array_from_bits(bits)
            if register.shape
            else bit_from_bits(list(bits.flat))
        )
    width = int(register.bitsize)
    signed = register.dtype.encoding == "signed"
    return (
        integer_array_from_bits(bits, width, signed=signed)
        if register.shape
        else integer_from_bits(list(bits.flat), width, signed=signed)
    )


def encode(register: Register, value: NDArray[Any]) -> NDArray[np.uint8]:
    if register.dtype.encoding == "bit":
        return bit_array_to_bits(value)
    return integer_array_to_bits(
        value, int(register.bitsize), signed=register.dtype.encoding == "signed"
    )


def signature_from_source(source: Any) -> Signature:
    return Signature(
        Register(
            reg.name,
            (
                classical_bit_type()
                if reg.dtype.num_cbits
                else (
                    quantum_bit_type()
                    if type(reg.dtype).__name__ == "QBit"
                    else (
                        signed_register_type(int(reg.bitsize))
                        if type(reg.dtype).__name__ == "QInt"
                        else unsigned_register_type(int(reg.bitsize))
                    )
                )
            ),
            reg.shape,
            RegisterSide[reg.side.name],
        )
        for reg in source
    )


def native(source: Any) -> NDArray[np.float64]:
    return classical_action_tensor(
        signature_from_source(source.signature),
        source.call_classically,
        decode,
        encode,
        operation_name=str(source),
    )


@pytest.mark.parametrize("kind", ["x", "swap", "toffoli", "add", "xor"])
def test_original_five_suppliers_dense_tensor_binding_and_complex_action(
    kind: str,
) -> None:
    from qualtran._infra.data_types import QAny, QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.bitwise import Xor
    from qualtran.bloqs.basic_gates.swap import TwoBitCSwap
    from qualtran.bloqs.basic_gates.toffoli import Toffoli
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
        my_tensors_from_classical_action,
    )

    from tests.unit.test_register_tensor_binding import source_bindings

    source = {
        "x": XGate(),
        "swap": TwoBitCSwap(),
        "toffoli": Toffoli(),
        "add": Add(QUInt(2)),
        "xor": Xor(QAny(2)),
    }[kind]
    sig = signature_from_source(source.signature)
    actual = native(source)
    expected = bloq_to_dense_via_classical_action(source)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, source.tensor_contract(), atol=2e-14, rtol=0)
    labels: dict[tuple[str, int], int] = {}
    src_in, inc = source_bindings(source.signature.lefts(), "in", labels)
    src_out, out = source_bindings(source.signature.rights(), "out", labels)
    original = my_tensors_from_classical_action(source, src_in, src_out)[0]
    node = bind_register_tensor(
        torch.from_numpy(actual).to(torch.complex128), sig, inc, out
    )
    np.testing.assert_array_equal(node.tensor.numpy(), original.data)
    assert node.labels == tuple(labels[index] for index in original.inds)
    matrix = contract_register_tensors((node,), sig, inc, out)
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    state = torch.complex(
        torch.arange(1, actual.shape[1] + 1, dtype=torch.float64),
        torch.arange(actual.shape[1], 0, -1, dtype=torch.float64),
    )
    state = state / state.norm() * torch.exp(1j * theta)
    evolved = matrix @ state
    torch.testing.assert_close(
        evolved, torch.from_numpy(expected).to(torch.complex128) @ state
    )
    (grad,) = torch.autograd.grad(evolved.real.sum(), (theta,))
    torch.testing.assert_close(grad, -evolved.imag.sum())


@pytest.mark.parametrize("shape", [(), (2,), (1, 2)])
@pytest.mark.parametrize("signed", [False, True])
def test_scalar_and_array_codec_callbacks_and_enumeration_order(
    shape: tuple[int, ...], signed: bool
) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import QInt, QUInt
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    class Identity(Bloq):
        @property
        def signature(self) -> Any:
            return SourceSignature(
                [SourceRegister("value", QInt(2) if signed else QUInt(2), shape=shape)]
            )

        def on_classical_vals(self, **kwargs: Any) -> Any:
            return kwargs

    source = Identity()
    calls = []

    def action(**values: Any) -> tuple[object, ...]:
        calls.append(values["value"])
        outputs = source.call_classically(**values)
        assert isinstance(outputs, tuple)
        return outputs

    def run() -> NDArray[np.float64]:
        return classical_action_tensor(
            signature_from_source(source.signature),
            action,
            decode,
            encode,
            operation_name=str(source),
        )

    try:
        expected = bloq_to_dense_via_classical_action(source)
    except ValueError as error:
        with pytest.raises(ValueError) as observed:
            run()
        assert str(observed.value) == str(error)
        return
    actual = run()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, np.eye(actual.shape[0]))
    assert len(calls) == actual.shape[1]
    np.testing.assert_array_equal(calls[0], np.full(shape, 0))
    np.testing.assert_array_equal(calls[-1], np.full(shape, -1 if signed else 3))


@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU", "SCALAR"])
def test_directional_and_scalar_shapes_match_original(side: str) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Side
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    class Constant(Bloq):
        @property
        def signature(self) -> Any:
            return SourceSignature(
                []
                if side == "SCALAR"
                else [SourceRegister("q", QBit(), side=Side[side])]
            )

        def on_classical_vals(self, **kwargs: Any) -> Any:
            return {} if side in ("SCALAR", "LEFT") else {"q": 1}

    source = Constant()
    actual = native(source)
    np.testing.assert_array_equal(actual, bloq_to_dense_via_classical_action(source))
    assert actual.ndim == {"LEFT": 1, "RIGHT": 1, "THRU": 2, "SCALAR": 0}[side]
    if side == "THRU":
        assert np.array_equal(
            actual, np.array([[0, 0], [1, 1]])
        )  # Non-injective source action is not rejected.


def test_phase_is_not_reconstructed_and_hadamard_is_not_classical() -> None:
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.z_basis import ZGate
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    for source in (ZGate(), Hadamard()):
        try:
            expected = bloq_to_dense_via_classical_action(source)
        except (ValueError, NotImplementedError) as error:
            with pytest.raises(type(error)):
                native(source)
        else:
            np.testing.assert_array_equal(native(source), expected)
            assert not np.allclose(expected, source.tensor_contract())


@pytest.mark.parametrize("shape", [(), (2,), (0,)])
def test_classical_bit_and_empty_source_boundaries(shape: tuple[int, ...]) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import CBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    class Identity(Bloq):
        @property
        def signature(self) -> Any:
            return SourceSignature([SourceRegister("q", CBit(), shape=shape)])

        def on_classical_vals(self, **kwargs: Any) -> Any:
            return kwargs

    source = Identity()
    try:
        expected = bloq_to_dense_via_classical_action(source)
    except (ValueError, NotImplementedError) as error:
        with pytest.raises(type(error)) as observed:
            native(source)
        assert str(observed.value) == str(error)
    else:
        np.testing.assert_array_equal(native(source), expected)


@pytest.mark.parametrize("exception", [ValueError, KeyError, NotImplementedError])
def test_callback_failures_preserve_wrapping_and_causes(
    exception: type[Exception],
) -> None:
    error = exception("callback failed")

    def action(**values: object) -> tuple[object, ...]:
        raise error

    with pytest.raises(exception) as observed:
        classical_action_tensor(
            Signature.build(q=1), action, decode, encode, operation_name="fixture"
        )
    if exception is ValueError:
        assert observed.value.__cause__ is error
        assert (
            str(observed.value) == "cannot compute tensor for fixture: callback failed"
        )
    else:
        assert observed.value is error


def test_size_refusal_precedes_action_and_allocation() -> None:
    sig = Signature([Register("q", unsigned_register_type(21))])

    def action(**values: object) -> tuple[object, ...]:
        raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="42 total qubits"):
        classical_action_tensor(sig, action, decode, encode, operation_name="huge")


def test_measured_add_supplier_replacement_full_branches_and_work() -> None:
    from itertools import product

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_component_tensor_expansion import (
        test_measured_add_consumer_preserves_gates_all_history_pairs_and_work,
    )

    record, source = wrapped_add(2)
    sig = Signature(record.functions[0].registers or ())

    # The native shared integer codecs decode the existing UINT metadata.
    def action(**values: object) -> tuple[object, ...]:
        outputs = source.call_classically(**values)
        assert isinstance(outputs, tuple)
        return outputs

    matrix = torch.from_numpy(
        classical_action_tensor(sig, action, decode, encode, operation_name=str(source))
    ).to(torch.complex128)
    initial = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    initial /= initial.norm()
    expected = matrix @ initial
    bound = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    state = torch.zeros(256, dtype=torch.complex128)
    state[::4] = initial
    for history in product((0, 1), repeat=4):
        branch = forced_branch(state, bound.circuit.instructions, history).reshape(
            64, 4
        )
        torch.testing.assert_close(branch[:, 0], expected / 4, atol=3e-14, rtol=0)
        assert torch.count_nonzero(branch[:, 1:]) == 0
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "h": 12,
        "t": 8,
        "cx": 40,
        "tdg": 8,
        "s": 4,
        "measure": 4,
        "cz": 4,
        "reset": 4,
    }
    test_measured_add_consumer_preserves_gates_all_history_pairs_and_work(False)


@pytest.mark.parametrize("outputs", [(), (0,), (0, 1), (2,)])
def test_raw_output_tuple_truncation_and_index_failures_match_source(
    outputs: tuple[int, ...],
) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    class RawAction(Bloq):
        @property
        def signature(self) -> Any:
            return SourceSignature.build(q=1)

        def call_classically(self, **values: Any) -> tuple[int, ...]:
            return outputs

    source = RawAction()
    try:
        expected = bloq_to_dense_via_classical_action(source)
    except (IndexError, ValueError) as error:
        with pytest.raises(type(error)) as observed:
            native(source)
        assert str(observed.value) == str(error)
    else:
        np.testing.assert_array_equal(native(source), expected)


@pytest.mark.parametrize("shape", [(), (2,)])
def test_zero_width_codec_failure_is_not_repaired(shape: tuple[int, ...]) -> None:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import QUInt
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    class EmptyWord(Bloq):
        @property
        def signature(self) -> Any:
            return SourceSignature([SourceRegister("q", QUInt(0), shape=shape)])

        def on_classical_vals(self, **values: Any) -> Any:
            return values

    source = EmptyWord()
    try:
        expected = bloq_to_dense_via_classical_action(source)
    except (ValueError, AssertionError) as error:
        with pytest.raises(type(error)) as observed:
            native(source)
        assert str(observed.value) == str(error)
    else:
        np.testing.assert_array_equal(native(source), expected)
