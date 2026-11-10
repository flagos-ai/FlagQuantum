"""Mixed SSA channels match unchanged source graphs and every joint branch."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import discard_word_function
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.matrices import H_MATRIX
from flagquantum.simulation.tensor_network.component_superoperator_nodes import (
    component_superoperator_nodes,
)
from flagquantum.simulation.tensor_network.measurement_copy import (
    x_measurement_node,
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_tensor_outer_indices,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_tensor_nodes import order
from tests.unit.test_program_owned_measurement import measured_function

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


def mixed_graph(
    discard: bool, idle_shape: tuple[int, ...] = ()
) -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Discard, MeasureX, MeasureZ

    sig = Signature(
        [
            Register("q", quantum_bit_type(), (2,), RegisterSide.LEFT),
            Register("spectator", quantum_bit_type()),
            Register("idle", classical_bit_type(), idle_shape),
            *(
                []
                if discard
                else [Register("c", classical_bit_type(), (2,), RegisterSide.RIGHT)]
            ),
        ]
    )
    builder = ComponentProgramBuilder(
        sig,
        (
            measured_function("first"),
            measure_x_word_function("second"),
            discard_word_function("drop"),
        ),
    )
    q = np.asarray(builder.inputs["q"])
    c = [
        builder.call_dict(name, {"q": q[i]})["c"]
        for i, name in enumerate(("first", "second"))
    ]
    outputs = {"spectator": builder.inputs["spectator"], "idle": builder.inputs["idle"]}
    if discard:
        for value in c:
            builder.call_dict("drop", {"c": value})
    else:
        outputs["c"] = np.array(c[::-1], dtype=object)
    source_sig = SourceSignature(
        [
            SourceRegister("q", QBit(), (2,), Side.LEFT),
            SourceRegister("spectator", QBit()),
            SourceRegister("idle", CBit(), idle_shape),
            *([] if discard else [SourceRegister("c", CBit(), (2,), Side.RIGHT)]),
        ]
    )
    bb, vals = BloqBuilder.from_signature(source_sig)
    bits = [
        bb.add(leaf, q=vals["q"][i]) for i, leaf in enumerate((MeasureZ(), MeasureX()))
    ]
    source_out = {"spectator": vals["spectator"], "idle": vals["idle"]}
    if discard:
        for bit in bits:
            bb.add(Discard(), c=bit)
    else:
        source_out["c"] = np.array(bits[::-1], dtype=object)
    return builder.finalize(outputs), bb.finalize(**source_out)


def supplier(
    like: torch.Tensor,
    *,
    original: bool = False,
    environments: list[int] | None = None,
    ports: list[int] | None = None,
    raw: list[TensorNetworkNode] | None = None,
) -> Callable[
    [FunctionRecord, Ports, Ports, Callable[[], int]], Iterable[TensorNetworkNode | int]
]:
    def supply(
        function: FunctionRecord,
        incoming: Ports,
        outgoing: Ports,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        if ports is not None:
            ports.extend(
                label
                for mapping in (incoming, outgoing)
                for words in mapping.values()
                for word in words
                for label in word
            )
        if function.name == "drop":
            yield incoming["c"][0][0]
            return
        q, c, env = incoming["q"][0][0], outgoing["c"][0][0], fresh()
        if environments is not None:
            environments.append(env)
        if original:
            from qualtran.bloqs.basic_gates import MeasureX, MeasureZ

            leaf = MeasureZ() if function.name == "first" else MeasureX()
            tensor, sentinel = leaf.my_tensors({"q": q}, {"c": c})
            mapping = {(q, 0): q, (c, 0): c, sentinel.ind_tuple: env}
            node = TensorNetworkNode(
                torch.as_tensor(tensor.data.real, dtype=like.dtype),
                tuple(mapping[idx] for idx in tensor.inds),
            )
        else:
            node = (
                z_measurement_copy_node
                if function.name == "first"
                else x_measurement_node
            )(q, c, env, like)
        if raw is not None:
            raw.append(node)
        yield node
        yield env

    return supply


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("idle_shape", [(), (0,)])
def test_full_original_channel_connectivity_replacement_and_fresh_environment(
    dtype: torch.dtype, discard: bool, idle_shape: tuple[int, ...]
) -> None:
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    record, source = mixed_graph(discard, idle_shape)
    like = torch.empty((), dtype=dtype)
    env: list[int] = []
    ports: list[int] = []
    nodes, inc, out = component_superoperator_nodes(
        record, order(record), supplier(like, environments=env, ports=ports), like
    )
    assert len(env) == len(set(env)) == 2 and not set(env) & set(ports)
    sig = Signature(record.functions[0].registers or ())
    expected = cbloq_to_superquimb(source)
    actual = contract_register_superoperator(nodes, sig, inc, out)
    tol = 2e-6 if dtype in {torch.float32, torch.complex64} else 2e-14
    np.testing.assert_allclose(
        actual.numpy(),
        quimb_to_dense(expected, source.signature, superoperator=True),
        atol=tol,
        rtol=0,
    )
    other, left, right = component_superoperator_nodes(
        record, order(record), supplier(like, original=True), like
    )
    assert (inc, out) == (left, right)
    torch.testing.assert_close(
        contract_register_superoperator(other, sig, left, right),
        actual,
        atol=tol,
        rtol=0,
    )
    assert len(nodes) == len(expected.tensors) == 6 + 2 * int(idle_shape != (0,))
    for native, tensor in zip(nodes, expected.tensors, strict=True):
        np.testing.assert_array_equal(native.tensor.resolve_conj().numpy(), tensor.data)
    names = register_tensor_outer_indices(sig, inc, out, superoperator=True)
    # Source placeholders use input/output; the shared native identity uses output/input.
    source_rows = [
        tensor.inds if i < 4 else tensor.inds[::-1]
        for i, tensor in enumerate(expected.tensors)
    ]
    external = {
        label: label
        for row in source_rows
        for label in row
        if isinstance(label, tuple)
        and len(label) == 4
        and label[-1] in {"lf", "lb", "rf", "rb"}
    }

    def canonical(rows: Any, external: Any) -> Any:
        internal: dict[Any, int] = {}
        return [
            tuple(
                (
                    external[label]
                    if label in external
                    else ("internal", internal.setdefault(label, len(internal)))
                )
                for label in row
            )
            for row in rows
        ]

    assert canonical([n.labels for n in nodes], names) == canonical(
        source_rows, external
    )
    # Repeated invocations own separate local namespaces and do not consume global labels.
    again, again_in, again_out = component_superoperator_nodes(
        record, order(record), supplier(like), like
    )
    assert (inc, out, [n.labels for n in nodes]) == (
        again_in,
        again_out,
        [n.labels for n in again],
    )


@pytest.mark.parametrize("discard", [False, True])
def test_append_all_complex_branches_gradient_work_zero_and_separate_resources(
    discard: bool,
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX, MeasureZ
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    record, source = mixed_graph(discard)
    like = torch.empty((), dtype=torch.complex128)
    raw: list[TensorNetworkNode] = []
    env: list[int] = []
    _, inc, out = component_superoperator_nodes(
        record, order(record), supplier(like, raw=raw, environments=env), like
    )
    raw_instrument, _ = _contract_nodes_greedy(
        raw,
        (
            inc["q"][0][0],
            inc["q"][1][0],
            *env,
            *(
                [out["c"][1][0], out["c"][0][0]]
                if not discard
                else [raw[0].labels[1], raw[1].labels[2]]
            ),
        ),
    )
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    initial = torch.arange(1, 9, dtype=torch.float64).to(torch.complex128)
    initial = initial * torch.exp(1j * theta * torch.arange(8))
    initial = initial / initial.norm()
    observed = torch.einsum(
        "ijefab,ijs->abefs", raw_instrument, initial.reshape(2, 2, 2)
    )
    z = torch.from_numpy(MeasureZ().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    x = torch.from_numpy(MeasureX().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    expected = torch.einsum("iae,jfb,ijs->abefs", z, x, initial.reshape(2, 2, 2))
    torch.testing.assert_close(observed, expected)
    grad = torch.autograd.grad(observed.real.sum(), theta, retain_graph=True)[0]
    source_grad = torch.autograd.grad(expected.real.sum(), theta)[0]
    torch.testing.assert_close(grad, source_grad)
    for idle in (0, 1):
        bound = bind_register_program(
            record,
            ((0,), (1,), (2,), ()),
            clean=(3,),
            classical_inputs=(None, None, None, idle),
        )
        gates = (
            Circuit(4)
            .append(FieldOperation(4, bound.circuit.instructions))
            .to_ir()
            .instructions
        )
        assert gates == bound.circuit.instructions
        counts = Counter(g.name for g in gates)
        assert counts == {"h": 1, "measure": 2, "classical_xor": 1}
        start = torch.zeros(16, dtype=torch.complex128)
        start[::2] = initial.detach()
        for a, b in product((0, 1), repeat=2):
            branch = forced_branch(start, gates, (a, b)).reshape(2, 2, 2, 2)
            assert torch.count_nonzero(branch[..., 1]) == 0
            restored = torch.einsum("ij,ejk->eik", H_MATRIX, branch[..., 0])
            torch.testing.assert_close(
                restored, expected[a, b].detach(), atol=3e-16, rtol=0
            )
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == 2 and model.clifford == 0 and model.total_t_count() == 0


def test_classical_direct_arrays_empty_graph_and_pre_callback_rejections() -> None:
    like = torch.empty((), dtype=torch.complex128)
    sig = Signature([Register("idle", classical_bit_type(), (2,))])
    builder = ComponentProgramBuilder(sig)
    record = builder.finalize({"idle": np.asarray(builder.inputs["idle"])[::-1]})

    def fail(*args: Any) -> Iterable[TensorNetworkNode | int]:
        raise AssertionError("supplier must not run")

    nodes, inc, out = component_superoperator_nodes(record, (), fail, like)
    actual = contract_register_superoperator(nodes, sig, inc, out)
    permutation = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=like.dtype
    )
    torch.testing.assert_close(
        actual, torch.einsum("ai,bj->abij", permutation, permutation)
    )
    empty = ComponentProgramBuilder(Signature(())).finalize({})
    assert component_superoperator_nodes(empty, (), fail, like) == ((), {}, {})
    with pytest.raises(ValueError, match="order"):
        component_superoperator_nodes(record, (0,), fail, like)
    with pytest.raises(ValueError, match="unknown"):
        component_superoperator_nodes(record, (), fail, like, "missing")
    invalid = replace(record, functions=record.functions + record.functions)
    with pytest.raises(ValueError, match="invalid"):
        component_superoperator_nodes(invalid, (), fail, like)


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing", "internal", "external", "bool", "supplier"]
)
def test_invalid_discard_and_supplier_failure_propagate(problem: str) -> None:
    record, _ = mixed_graph(False)
    like = torch.empty((), dtype=torch.complex128)
    base = supplier(like)

    def broken(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        if problem == "supplier":
            raise LookupError("missing numerical leaf")
        yield from base(function, inc, out, fresh)
        if function.name == "first":
            if problem == "duplicate":
                env = fresh()
                yield TensorNetworkNode(torch.ones(2), (env,))
                yield env
                yield env
            elif problem == "missing":
                yield fresh()
            elif problem == "internal":
                env = fresh()
                yield TensorNetworkNode(torch.eye(2), (env, env))
                yield env
            elif problem == "external":
                yield inc["q"][0][0]
            else:
                yield True

    error = LookupError if problem == "supplier" else ValueError
    with pytest.raises(error):
        component_superoperator_nodes(record, order(record), broken, like)
