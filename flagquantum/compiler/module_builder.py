"""Private Module frontend capture over canonical IR and a Torch binding graph."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import torch
from torch.fx import GraphModule, Interpreter, Node

from ..circuit import Circuit
from ..core.ir import CircuitIR, Instruction, IRSerializationError, _encode_value
from ..core.parameters import Parameter
from ..errors import CapabilityError, ExecutionError

# Classical binding operations, not a quantum language or backend registry.
_UNARY = frozenset(
    ("neg", "sin", "cos", "tan", "tanh", "exp", "log", "sqrt", "sigmoid")
)
_BINARY = frozenset(
    ("add", "sub", "mul", "div", "pow", "gt", "ge", "lt", "le", "eq", "ne")
)
_SHAPES = frozenset(
    (
        "select",
        "slice",
        "view",
        "reshape",
        "_unsafe_view",
        "unsqueeze",
        "squeeze",
        "permute",
        "transpose",
        "t",
        "expand",
        "stack",
        "cat",
        "unbind",
    )
)
_OVERLOADS = {
    "add": {"Tensor", "Scalar"},
    "sub": {"Tensor", "Scalar"},
    "mul": {"Tensor", "Scalar"},
    "div": {"Tensor", "Scalar"},
    "pow": {"Tensor_Scalar", "Tensor_Tensor", "Scalar"},
    **{name: {"Tensor", "Scalar"} for name in ("gt", "ge", "lt", "le", "eq", "ne")},
    "select": {"int"},
    "slice": {"Tensor"},
    "squeeze": {"default", "dim"},
    "transpose": {"int"},
    "unbind": {"int"},
}
_OTHER = frozenset(
    ("detach", "alias", "clone", "_to_copy", "where", "relu", "zeros_like", "ones_like")
)
_SUPPORTED = _UNARY | _BINARY | _SHAPES | _OTHER


def _operation(target: Any) -> str | None:
    if not isinstance(target, torch._ops.OpOverload):
        return None
    namespace, name, overload = str(target).split(".")
    if namespace != "aten":
        return None
    admitted = _OVERLOADS.get(name, {"self"} if name == "where" else {"default"})
    if name in _SUPPORTED and overload in admitted:
        return name
    return None


def _jax_bindings(graph: GraphModule, *inputs: Any) -> Any:
    """Lower a previously admitted FX graph during JAX tracing only."""
    import jax
    import jax.numpy as jnp

    class Bindings(Interpreter):
        def run_node(self, node: Node) -> Any:
            value = super().run_node(node)
            expected = node.meta.get("val")
            if isinstance(expected, torch.Tensor):
                # Preserve Torch's scalar promotion and explicit input casts.
                return jnp.asarray(
                    value, dtype=str(expected.dtype).removeprefix("torch.")
                )
            return value

        def call_function(self, target: Any, args: Any, kwargs: Any) -> Any:
            if target is operator.getitem:
                return operator.getitem(*args)
            name = _operation(target)
            value = args[0]
            if name in _UNARY:
                function = (
                    jax.nn.sigmoid
                    if name == "sigmoid"
                    else getattr(jnp, "negative" if name == "neg" else name)
                )
                return function(value)
            if name in _BINARY:
                other = args[1]
                if name in {"add", "sub"}:
                    other = other * kwargs.get("alpha", 1)
                function = {
                    "sub": "subtract",
                    "mul": "multiply",
                    "div": "divide",
                    "pow": "power",
                    "gt": "greater",
                    "ge": "greater_equal",
                    "lt": "less",
                    "le": "less_equal",
                    "eq": "equal",
                    "ne": "not_equal",
                }.get(name, name)
                return getattr(jnp, function)(value, other)
            if name == "select":
                return jnp.take(value, args[2], axis=args[1])
            if name == "slice":
                axis = args[1] if len(args) > 1 else 0
                start = args[2] if len(args) > 2 else 0
                end = args[3] if len(args) > 3 else None
                step = args[4] if len(args) > 4 else 1
                index = [slice(None)] * value.ndim
                index[axis] = slice(start, end, step)
                return value[tuple(index)]
            if name in {"view", "reshape", "_unsafe_view"}:
                return jnp.reshape(value, tuple(args[1]))
            if name == "unsqueeze":
                return jnp.expand_dims(value, args[1])
            if name == "squeeze":
                if len(args) == 1:
                    return jnp.squeeze(value)
                return (
                    jnp.squeeze(value, args[1]) if value.shape[args[1]] == 1 else value
                )
            if name == "permute":
                return jnp.transpose(value, tuple(args[1]))
            if name == "transpose":
                return jnp.swapaxes(value, args[1], args[2])
            if name == "t":
                return jnp.transpose(value)
            if name == "expand":
                shape = (1,) * (len(args[1]) - value.ndim) + value.shape
                return jnp.broadcast_to(
                    value,
                    tuple(
                        old if new == -1 else new
                        for old, new in zip(shape, args[1], strict=True)
                    ),
                )
            if name in {"stack", "cat"}:
                return (jnp.stack if name == "stack" else jnp.concatenate)(
                    value, axis=args[1] if len(args) > 1 else 0
                )
            if name == "unbind":
                axis = args[1] if len(args) > 1 else 0
                return tuple(
                    jnp.take(value, index, axis=axis)
                    for index in range(value.shape[axis])
                )
            if name == "detach":
                return jax.lax.stop_gradient(value)
            if name in {"alias", "clone", "_to_copy"}:
                return value
            if name == "where":
                return jnp.where(*args)
            if name == "relu":
                # Torch chooses zero derivative at zero; jnp.maximum does not.
                return jnp.where(value > 0, value, jnp.zeros_like(value))
            if name in {"zeros_like", "ones_like"}:
                return getattr(jnp, name)(value)
            raise TypeError(f"unadmitted Module binding operation: {target}")

    # Interpreter environments must not escape a trace or retain its tracers.
    return Bindings(graph).run(*inputs)


@dataclass(frozen=True)
class CapturedModuleBuilder:
    """Compiler-owned ephemeral capture; Simulation receives only its Core IR."""

    template: CircuitIR
    binding_graph: GraphModule
    slots: tuple[tuple[int, str], ...]
    binding_names: tuple[str, ...]
    constructor: Mapping[str, Any]
    circuit_type: type[Circuit]
    has_inputs: bool
    tensor_constants: tuple[torch.Tensor, ...]
    group_names: tuple[str, ...]

    def checkpoint_signature(self) -> str | None:
        """Fingerprint the symbolic program and the certified binding computation."""
        if self.circuit_type is not Circuit:
            return None
        indices: dict[Node, int] = {}

        def encode(value: Any) -> Any:
            if isinstance(value, Node):
                return {"node": indices[value]}
            if isinstance(value, Mapping):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("unsupported binding mapping key")
                return {"mapping": {key: encode(item) for key, item in value.items()}}
            if isinstance(value, (tuple, list)):
                kind = "tuple" if isinstance(value, tuple) else "list"
                return {kind: [encode(item) for item in value]}
            if isinstance(
                value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
            ):
                return {type(value).__name__: str(value)}
            return _encode_value(value)

        nodes: list[list[Any]] = []
        try:
            for node in self.binding_graph.graph.nodes:
                if node.op == "placeholder":
                    value = node.meta.get("val")
                    if not isinstance(value, torch.Tensor):
                        return None
                    entry = [list(value.shape), str(value.dtype)]
                elif node.op == "get_attr":
                    value = self.binding_graph
                    for part in str(node.target).split("."):
                        value = getattr(value, part)
                    if not isinstance(value, torch.Tensor) or value.requires_grad:
                        return None
                    entry = encode(value)
                elif node.op == "call_function":
                    target = node.target
                    if target is operator.getitem:
                        operation = "operator.getitem"
                    elif (
                        isinstance(target, torch._ops.OpOverload)
                        and (
                            _operation(target) is not None
                            or target is torch.ops.aten.lift_fresh_copy.default
                        )
                        and not target._schema.is_mutable
                        and torch.Tag.nondeterministic_seeded not in target.tags
                        and torch.Tag.nondeterministic_bitwise not in target.tags
                    ):
                        operation = str(target)
                    else:
                        return None
                    entry = [operation, encode(node.args), encode(node.kwargs)]
                elif node.op == "output":
                    entry = encode(node.args)
                else:
                    return None
                indices[node] = len(nodes)
                nodes.append([node.op, entry])
            payload = json.dumps(
                {
                    "encoding": 1,
                    "torch": str(torch.__version__),
                    "template": self.template.to_dict(),
                    "groups": self.group_names,
                    "has_inputs": self.has_inputs,
                    "slots": self.slots,
                    "bindings": self.binding_names,
                    "nodes": nodes,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (
            AttributeError,
            IRSerializationError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def jax_builder(self) -> tuple[Callable[..., Circuit] | None, str]:
        """Return canonical lowering, or an explicit legacy compatibility reason."""
        if self.group_names:
            return None, "named_parameter_groups"
        if any(item.matrix is not None for item in self.template.instructions):
            raise CapabilityError(
                "JAX Module does not lower explicit instruction matrices; use PyTorch"
            )
        if self.circuit_type is not Circuit:
            return None, "custom_circuit_type"
        for node in self.binding_graph.graph.nodes:
            if node.op in {"placeholder", "output"}:
                continue
            if node.op != "call_function":
                return None, f"unsupported_binding_node:{node.op}"
            if node.target is operator.getitem:
                continue
            if _operation(node.target) is None:
                return None, f"unsupported_binding_operation:{node.target}"
            if _operation(node.target) == "_to_copy" and node.kwargs.get("device"):
                source = node.args[0]
                if isinstance(source, Node) and (
                    source.meta["val"].device != node.meta["val"].device
                ):
                    return None, "binding_device_transfer"
            target = node.target
            if target._schema.is_mutable or any(
                tag in target.tags
                for tag in (
                    torch.Tag.nondeterministic_seeded,
                    torch.Tag.nondeterministic_bitwise,
                )
            ):
                return None, f"impure_binding_operation:{target}"

        def build(parameters: Any, inputs: Any = None) -> Circuit:
            raw = parameters.reshape(parameters.shape)
            tensors = (raw, inputs) if self.has_inputs else (raw,)
            values = _jax_bindings(self.binding_graph, *tensors)
            if not isinstance(values, (tuple, list)):
                values = (values,)
            mapping = dict(zip(self.binding_names, values, strict=True))
            circuit = Circuit.from_ir(self.template, **self.constructor)
            return circuit.bind_parameters(mapping)

        return build, "canonical_ir_fx_bindings"


def capture_module_builder(
    builder: Callable[[Any, torch.Tensor | None], Circuit | CircuitIR],
    tensor_inputs: tuple[torch.Tensor, ...],
    *,
    names: tuple[str, ...],
    has_inputs: bool,
) -> CapturedModuleBuilder:
    """Capture one static builder using the reviewed Module frontend boundary.

    Args:
        builder: Existing Runtime invocation adapter, with parameters and inputs.
        tensor_inputs: Current flattened parameters followed by optional inputs.
        names: Named parameter order, or empty for a single parameter tensor.
        has_inputs: Whether the final flat tensor is the explicit input.

    Returns:
        Canonical symbolic IR, its binding graph and constructor properties.

    Raises:
        ExecutionError: The existing static-topology capture requirement fails.
        TypeError: Dynamic Circuit inputs remain unsupported.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    captured: list[Circuit] = []

    def extract(*flat_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        count = len(flat_inputs) - int(has_inputs)
        raw = flat_inputs[:count]
        parameters = (
            {name: raw[index] for index, name in enumerate(names)} if names else raw[0]
        )
        circuit = builder(parameters, flat_inputs[-1] if has_inputs else None)
        if not isinstance(circuit, Circuit):
            raise TypeError("compiled circuit builder must return a Circuit")
        if any(
            isinstance(item.matrix, torch.Tensor) and item.matrix.requires_grad
            for item in circuit.to_ir().instructions
        ):
            raise TypeError("compiled builders do not yet support dynamic matrices")
        captured[:] = [circuit]
        return tuple(
            value
            for item in circuit.to_ir().instructions
            for value in item.params.values()
            if isinstance(value, torch.Tensor)
        )

    try:
        # These bindings are differentiated again. Post-autograd defaults may
        # rewrite detach to alias, which would erase the user's gradient cuts.
        graph = make_fx(extract, decomposition_table={})(*tensor_inputs)
    except Exception as error:
        raise ExecutionError(
            "Module builder compilation requires static circuit topology; "
            "do not branch on Tensor values when choosing gates or qubits"
        ) from error
    graph.graph.eliminate_dead_code()
    graph.recompile()
    if not captured:
        raise ExecutionError("compiled builder did not capture a Circuit")
    circuit = captured[-1]
    if circuit._inputs is not None:
        raise TypeError(
            "compiled builders do not yet support dynamic Circuit inputs; encode inputs in gate parameters instead"
        )
    ir = circuit.to_ir()
    slots = tuple(
        (index, key)
        for index, item in enumerate(ir.instructions)
        for key, value in item.params.items()
        if isinstance(value, torch.Tensor)
    )
    used = set(circuit.parameter_names)
    binding_names = []
    for index in range(len(slots)):
        name = f"__fq_module_slot_{index}"
        while name in used:
            name = "_" + name
        used.add(name)
        binding_names.append(name)
    symbols = {
        slot: Parameter(name) for slot, name in zip(slots, binding_names, strict=True)
    }
    template = replace(
        ir,
        instructions=tuple(
            Instruction(
                item.name,
                item.wires,
                {
                    key: symbols.get((index, key), value)
                    for key, value in item.params.items()
                },
                item.matrix,
                item.metadata,
            )
            for index, item in enumerate(ir.instructions)
        ),
    )
    tensors = {
        id(value): value
        for _, value in (*graph.named_buffers(), *graph.named_parameters())
    }
    return CapturedModuleBuilder(
        template=template,
        binding_graph=graph,
        slots=slots,
        binding_names=tuple(binding_names),
        constructor=MappingProxyType(dict(circuit.circuit_param)),
        circuit_type=type(circuit),
        has_inputs=has_inputs,
        tensor_constants=tuple(tensors.values()),
        group_names=names,
    )
