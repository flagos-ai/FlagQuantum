"""Classical control tensors retain source phases and drive measured parity graphs."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from itertools import product

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import parity_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.classical_parity import (
    classical_parity_nodes,
)
from flagquantum.simulation.tensor_network.component_superoperator_nodes import (
    component_superoperator_nodes,
)
from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.controlled_tensor import (
    controlled_register_tensor_node,
)
from flagquantum.simulation.tensor_network.measurement_copy import (
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_tensor_nodes import order
from tests.unit.test_program_owned_measurement import measured_function
from tests.unit.test_register_classical_parity import feedback_function
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


@pytest.mark.parametrize("polarity", [0, 1])
@pytest.mark.parametrize("kind", ["x", "z", "s"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_original_classical_control_raw_axes_full_tensor_and_both_polarities(
    polarity: int, kind: str, dtype: torch.dtype
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import SGate, XGate, ZGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        contract_register_tensors,
    )

    leaf = {"x": XGate, "z": ZGate, "s": SGate}[kind]()
    original = ClassicallyControlled(leaf, CtrlSpec(CBit(), cvs=polarity))
    controls = native_signature(original.ctrl_regs)
    child = native_signature(leaf.signature)
    inc, out = {"ctrl": ((0,),), "q": ((1,),)}, {"ctrl": ((2,),), "q": ((3,),)}
    calls: list[int] = []

    def child_data() -> NDArray[np.complex128]:
        calls.append(1)
        return np.asarray(leaf.tensor_contract(), dtype=np.complex128)

    node = controlled_register_tensor_node(
        controls,
        child,
        [np.array(polarity)],
        child_data,
        inc,
        out,
        torch.empty((), dtype=dtype),
    )
    source = original.my_tensors({"ctrl": "a", "q": "b"}, {"ctrl": "c", "q": "d"})[0]
    assert calls == [1] and node.labels == (2, 3, 0, 1)
    assert source.inds == (("c", 0), ("d", 0), ("a", 0), ("b", 0))
    np.testing.assert_array_equal(node.tensor.numpy(), source.data)
    sig = Signature((*controls, *child))
    actual = contract_register_tensors((node,), sig, inc, out)
    np.testing.assert_array_equal(actual.numpy(), original.tensor_contract())


def graph(width: int, polarity: bool) -> ProgramRecord:
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("q", quantum_bit_type(), (width,), RegisterSide.LEFT),
                Register("targets", quantum_bit_type(), (2,)),
                Register("spectator", quantum_bit_type()),
                Register("c", classical_bit_type(), (width,), RegisterSide.RIGHT),
                Register("p", classical_bit_type(), side=RegisterSide.RIGHT),
            )
        ),
        (
            measured_function("measure"),
            parity_word_function("parity", width),
            feedback_function(polarity),
        ),
    )
    bits = np.array(
        [
            builder.call_dict("measure", {"q": q})["c"]
            for q in np.asarray(builder.inputs["q"]).flat
        ],
        dtype=object,
    )
    parity = builder.call_dict("parity", {"x": bits})
    result = builder.call_dict(
        "feedback", {"c": parity["parity"], "targets": builder.inputs["targets"]}
    )
    return builder.finalize(
        {
            "c": parity["x"],
            "p": result["c"],
            "targets": result["targets"],
            "spectator": builder.inputs["spectator"],
        }
    )


def graph_supplier(
    width: int,
    polarity: bool,
    like: torch.Tensor,
    *,
    original: bool = False,
    raw: list[TensorNetworkNode] | None = None,
    environments: list[int] | None = None,
) -> Callable[
    [FunctionRecord, Ports, Ports, Callable[[], int]], Iterable[TensorNetworkNode | int]
]:
    def supply(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        nodes: tuple[TensorNetworkNode, ...]
        traced: tuple[int, ...] = ()
        if function.name == "measure":
            env = fresh()
            if environments is not None:
                environments.append(env)
            nodes = (
                z_measurement_copy_node(inc["q"][0][0], out["c"][0][0], env, like),
            )
            traced = (env,)
        elif function.name == "parity":
            if original:
                from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity

                table = torch.zeros((2,) * (2 * width + 1), dtype=like.dtype)
                for bits in product((0, 1), repeat=width):
                    x, p = Parity(width).call_classically(x=np.array(bits, dtype=int))
                    table[(*map(int, x), int(p), *bits)] = 1
                nodes = (
                    bind_register_tensor(
                        table, Signature(function.registers or ()), inc, out
                    ),
                )
            else:
                nodes = classical_parity_nodes(width, inc, out, fresh, like)
        else:
            from qualtran import CBit, CtrlSpec
            from qualtran.bloqs.basic_gates import XGate, ZGate
            from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

            intermediate = fresh()
            feedback_nodes = []
            for i, leaf in enumerate((ZGate(), XGate())):
                source = ClassicallyControlled(
                    leaf, CtrlSpec(CBit(), cvs=int(polarity))
                )
                incoming = {
                    "ctrl": ((inc["c"][0][0] if i == 0 else intermediate,),),
                    "q": inc["targets"][i : i + 1],
                }
                outgoing = {
                    "ctrl": ((intermediate if i == 0 else out["c"][0][0],),),
                    "q": out["targets"][i : i + 1],
                }
                if original:
                    tensor = source.my_tensors(
                        {"ctrl": incoming["ctrl"][0][0], "q": incoming["q"][0][0]},
                        {"ctrl": outgoing["ctrl"][0][0], "q": outgoing["q"][0][0]},
                    )[0]
                    feedback_nodes.append(
                        TensorNetworkNode(
                            torch.as_tensor(tensor.data, dtype=like.dtype),
                            tuple(edge for edge, _ in tensor.inds),
                        )
                    )
                else:

                    child = np.asarray(leaf.tensor_contract(), dtype=np.complex128)

                    def child_data(
                        value: NDArray[np.complex128] = child,
                    ) -> NDArray[np.complex128]:
                        return value

                    feedback_nodes.append(
                        controlled_register_tensor_node(
                            native_signature(source.ctrl_regs),
                            native_signature(leaf.signature),
                            [np.array(int(polarity))],
                            child_data,
                            incoming,
                            outgoing,
                            like,
                        )
                    )
            nodes = tuple(feedback_nodes)
        if raw is not None:
            raw.extend(nodes)
        yield from nodes
        yield from traced

    return supply


@pytest.mark.parametrize("width", [0, 1, 2, 3])
@pytest.mark.parametrize("polarity", [False, True])
def test_full_instrument_all_measurement_branches_relative_gradient_and_append(
    width: int, polarity: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity

    record = graph(width, polarity)
    like = torch.empty((), dtype=torch.complex128)
    raw: list[TensorNetworkNode] = []
    env: list[int] = []
    _, inc, out = component_superoperator_nodes(
        record,
        order(record),
        graph_supplier(width, polarity, like, raw=raw, environments=env),
        like,
    )
    source_raw: list[TensorNetworkNode] = []
    source_env: list[int] = []
    _, source_in, source_out = component_superoperator_nodes(
        record,
        order(record),
        graph_supplier(
            width,
            polarity,
            like,
            original=True,
            raw=source_raw,
            environments=source_env,
        ),
        like,
    )
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    size = 1 << (width + 3)
    state = torch.arange(1, size + 1, dtype=torch.float64) * torch.exp(
        1j * theta * torch.arange(size)
    )
    state = state / state.norm()

    def instrument(
        nodes: list[TensorNetworkNode],
        incoming: Ports,
        outgoing: Ports,
        environment: list[int],
    ) -> torch.Tensor:
        inputs = tuple(
            label
            for name in ("q", "targets", "spectator")
            for word in incoming[name]
            for label in word
        )
        outputs = tuple(
            label
            for name in ("c", "p", "targets", "spectator")
            for word in outgoing[name]
            for label in word
        )
        # Direct-through spectator is supplied explicitly here because raw collects callee leaves only.
        identity = TensorNetworkNode(
            torch.eye(2, dtype=like.dtype),
            (outgoing["spectator"][0][0], incoming["spectator"][0][0]),
        )
        value, _ = _contract_nodes_greedy(
            (
                *nodes,
                identity,
                TensorNetworkNode(state.reshape((2,) * (width + 3)), inputs),
            ),
            (*outputs, *environment),
        )
        return value

    actual = instrument(raw, inc, out, env)
    reference = instrument(source_raw, source_in, source_out, source_env)
    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-15)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), theta, retain_graph=True)[0],
        torch.autograd.grad(reference.real.sum(), theta, retain_graph=True)[0],
        rtol=0,
        atol=2e-14,
    )
    physical = tuple((i,) for i in range(width + 3))
    bound = bind_register_program(record, physical, clean=(width + 3,))
    instructions = (
        Circuit(width + 4)
        .append(FieldOperation(width + 4, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert instructions == bound.circuit.instructions
    counts = Counter(g.name for g in instructions)
    assert counts == Counter({"measure": width, "classical_xor": 1, "z": 1, "x": 1})
    assert [gate.name for gate in instructions] == [
        *(["measure"] * width),
        "classical_xor",
        "z",
        "x",
    ]
    for i, gate in enumerate(instructions[:width]):
        assert gate.wires == (i,) and gate.metadata["classical_bit"] == i
    xor = instructions[width]
    assert xor.params == {"sources": tuple(range(width))}
    assert xor.metadata["classical_bit"] == width
    for offset, gate in enumerate(instructions[-2:]):
        assert gate.wires == (width + offset,)
        assert gate.metadata["conditions"] == ((width, int(polarity)),)

    initial = torch.zeros(1 << (width + 4), dtype=like.dtype)
    initial[::2] = state.detach()
    for bits in product((0, 1), repeat=width):
        _, parity = Parity(width).call_classically(x=np.array(bits, dtype=int))
        branch = forced_branch(initial, instructions, bits).reshape((2,) * (width + 4))
        assert torch.count_nonzero(branch[..., 1]) == 0
        classical_axes = (*bits, int(parity))
        target = actual[classical_axes].detach()
        for env_bits in product((0, 1), repeat=width):
            expected = (
                branch.reshape(1 << width, 2, 2, 2, 2)[
                    sum(bit << (width - 1 - i) for i, bit in enumerate(env_bits)),
                    ...,
                    0,
                ]
                if env_bits == bits
                else torch.zeros((2, 2, 2), dtype=like.dtype)
            )
            torch.testing.assert_close(
                target[(slice(None),) * 3 + env_bits], expected, rtol=0, atol=2e-15
            )

    # Prepared density uses exactly the same doubling/trace boundary as the unprepared channel.
    def density(
        nodes: list[TensorNetworkNode],
        incoming: Ports,
        outgoing: Ports,
        environment: list[int],
    ) -> torch.Tensor:
        inputs = tuple(
            label
            for name in ("q", "targets", "spectator")
            for word in incoming[name]
            for label in word
        )
        identity = TensorNetworkNode(
            torch.eye(2, dtype=like.dtype),
            (outgoing["spectator"][0][0], incoming["spectator"][0][0]),
        )
        paired = trace_conjugate_indices(
            tensor_conjugate_pairs(
                (
                    *nodes,
                    identity,
                    TensorNetworkNode(
                        state.detach().reshape((2,) * (width + 3)), inputs
                    ),
                )
            ),
            tuple(environment),
        )
        right = Signature(
            r
            for r in record.functions[0].registers or ()
            if r.side != RegisterSide.LEFT
        )
        right = Signature(
            Register(r.name, r.dtype, r.shape, RegisterSide.RIGHT) for r in right
        )
        return contract_register_superoperator(paired, right, {}, outgoing)

    torch.testing.assert_close(
        density(raw, inc, out, env),
        density(source_raw, source_in, source_out, source_env),
        rtol=0,
        atol=2e-15,
    )


def test_real_phase_loss_duplicate_bindings_and_callback_failure_are_explicit() -> None:
    control = Signature([Register("ctrl", classical_bit_type())])
    child = Signature.build(q=1)
    inc, out = {"ctrl": ((0,),), "q": ((1,),)}, {"ctrl": ((2,),), "q": ((3,),)}
    with pytest.raises(ValueError, match="complex prototype"):
        controlled_register_tensor_node(
            control,
            child,
            [np.array(1)],
            lambda: np.diag([1, 1j]),
            inc,
            out,
            torch.empty(()),
        )

    def fail() -> NDArray[np.complex128]:
        raise LookupError("child unavailable")

    with pytest.raises(ValueError, match="distinct"):
        controlled_register_tensor_node(
            control, child, [np.array(1)], fail, inc, inc, torch.empty(())
        )
    with pytest.raises(LookupError, match="unavailable"):
        controlled_register_tensor_node(
            control, child, [np.array(1)], fail, inc, out, torch.empty(())
        )
    node = controlled_register_tensor_node(
        control,
        child,
        [np.array(0)],
        lambda: np.eye(2, dtype=np.complex128),
        inc,
        out,
        torch.empty((), dtype=torch.float64),
    )
    assert node.tensor.dtype == torch.float64
