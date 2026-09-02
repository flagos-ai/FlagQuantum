"""Private, deterministic lowering to the universal RX/RY/RZ/CX profile."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..bindings import RuntimeBindingRef, SymbolicExpression, SymbolicParameter
from ..diagnostics import Diagnostic, DiagnosticCode
from ..ir.modules import Block, QuantumModule, Region
from ..ir.operations import FrozenAttributes, Operation
from ..ir.schemas import circuit_ir_v1_schema_registry
from ..ir.values import ValueId, ValueRef
from ..ir.verifier import verify_module
from .base import PassDescriptor, PassResult


@dataclass(frozen=True)
class TargetGateSetProfile:
    name: str
    version: str
    native_operations: tuple[str, ...]
    certified_source_operations: tuple[str, ...]


UNIVERSAL_RX_RY_RZ_CX_V1 = TargetGateSetProfile(
    "universal_rx_ry_rz_cx",
    "1.0",
    ("quantum.rx", "quantum.ry", "quantum.rz", "quantum.cx"),
    (
        "quantum.x",
        "quantum.y",
        "quantum.z",
        "quantum.h",
        "quantum.s",
        "quantum.sdg",
        "quantum.t",
        "quantum.tdg",
        "quantum.sx",
        "quantum.sxdg",
        "quantum.rx",
        "quantum.ry",
        "quantum.rz",
        "quantum.phase",
        "quantum.u1",
        "quantum.u2",
        "quantum.u3",
        "quantum.cx",
        "quantum.cy",
        "quantum.cz",
        "quantum.swap",
        "quantum.crx",
        "quantum.cry",
        "quantum.crz",
        "quantum.cphase",
        "quantum.rxx",
        "quantum.ryy",
        "quantum.rzz",
        "quantum.ccx",
        "quantum.cswap",
    ),
)

_Spec = tuple[str, tuple[int, ...], dict[str, object]]
_PARAMETRIC = {
    "quantum.phase",
    "quantum.u1",
    "quantum.u2",
    "quantum.u3",
    "quantum.crx",
    "quantum.cry",
    "quantum.crz",
    "quantum.cphase",
    "quantum.rxx",
    "quantum.ryy",
    "quantum.rzz",
}


def _diagnostic(message: str) -> Diagnostic:
    return Diagnostic(
        DiagnosticCode.UNKNOWN_OPERATION,
        message,
        notes=("target decomposition never falls back or leaves illegal gates",),
    )


def _tensor(value: object) -> torch.Tensor | None:
    if not isinstance(value, FrozenAttributes) or value.get("kind") != "tensor":
        return None
    dtype = getattr(torch, str(value.get("dtype")), None)
    if dtype is None:
        return None
    return torch.tensor(value.get("data"), dtype=dtype).reshape(
        tuple(value.get("shape", ()))
    )


def _freeze_tensor(value: torch.Tensor) -> FrozenAttributes:
    tensor = value.detach().cpu()
    data = tensor.item() if tensor.ndim == 0 else tensor.tolist()
    return FrozenAttributes(
        {
            "kind": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": tuple(tensor.shape),
            "data": data,
        }
    )


def _scale(value: object, factor: float) -> object:
    if factor == 1.0:
        return value
    if isinstance(value, (RuntimeBindingRef, SymbolicParameter, SymbolicExpression)):
        return SymbolicExpression("mul", (value, factor))
    tensor = _tensor(value)
    if tensor is not None:
        return _freeze_tensor(tensor * factor)
    try:
        return value * factor  # type: ignore[operator]
    except TypeError as exc:
        raise ValueError("decomposition parameter cannot be scaled") from exc


def _gate(name: str, wires: tuple[int, ...], **attributes: object) -> _Spec:
    return (f"quantum.{name}", wires, attributes)


def _h(wire: int) -> list[_Spec]:
    return [
        _gate("rz", (wire,), theta=math.pi),
        _gate("ry", (wire,), theta=math.pi / 2),
    ]


def _rzz(theta: object, left: int, right: int) -> list[_Spec]:
    return [
        _gate("cx", (left, right)),
        _gate("rz", (right,), theta=theta),
        _gate("cx", (left, right)),
    ]


def _ccx(control_0: int, control_1: int, target: int) -> list[_Spec]:
    return [
        *_h(target),
        _gate("cx", (control_1, target)),
        _gate("rz", (target,), theta=-math.pi / 4),
        _gate("cx", (control_0, target)),
        _gate("rz", (target,), theta=math.pi / 4),
        _gate("cx", (control_1, target)),
        _gate("rz", (target,), theta=-math.pi / 4),
        _gate("cx", (control_0, target)),
        _gate("rz", (control_1,), theta=math.pi / 4),
        _gate("rz", (target,), theta=math.pi / 4),
        *_h(target),
        _gate("cx", (control_0, control_1)),
        _gate("rz", (control_0,), theta=math.pi / 4),
        _gate("rz", (control_1,), theta=-math.pi / 4),
        _gate("cx", (control_0, control_1)),
    ]


def _expand(operation: Operation) -> list[_Spec] | None:
    name = operation.name
    attributes = operation.attributes
    if name in UNIVERSAL_RX_RY_RZ_CX_V1.native_operations:
        return [(name, tuple(range(len(operation.operands))), dict(attributes))]
    fixed = {
        "quantum.i": [],
        "quantum.x": [_gate("rx", (0,), theta=math.pi)],
        "quantum.y": [_gate("ry", (0,), theta=math.pi)],
        "quantum.z": [_gate("rz", (0,), theta=math.pi)],
        "quantum.h": _h(0),
        "quantum.s": [_gate("rz", (0,), theta=math.pi / 2)],
        "quantum.sdg": [_gate("rz", (0,), theta=-math.pi / 2)],
        "quantum.t": [_gate("rz", (0,), theta=math.pi / 4)],
        "quantum.tdg": [_gate("rz", (0,), theta=-math.pi / 4)],
        "quantum.sx": [_gate("rx", (0,), theta=math.pi / 2)],
        "quantum.sxdg": [_gate("rx", (0,), theta=-math.pi / 2)],
        "quantum.cy": [
            _gate("rz", (1,), theta=-math.pi / 2),
            _gate("cx", (0, 1)),
            _gate("rz", (1,), theta=math.pi / 2),
        ],
        "quantum.cz": [*_h(1), _gate("cx", (0, 1)), *_h(1)],
        "quantum.swap": [
            _gate("cx", (0, 1)),
            _gate("cx", (1, 0)),
            _gate("cx", (0, 1)),
        ],
        "quantum.ccx": _ccx(0, 1, 2),
        "quantum.cswap": [
            _gate("cx", (2, 1)),
            *_ccx(0, 1, 2),
            _gate("cx", (2, 1)),
        ],
    }
    if name in fixed:
        return fixed[name]
    if name not in _PARAMETRIC:
        return None
    if name in {"quantum.phase", "quantum.u1"}:
        return [_gate("rz", (0,), theta=attributes["theta"])]
    if name == "quantum.u2":
        return [
            _gate("rz", (0,), theta=attributes["lbd"]),
            _gate("ry", (0,), theta=math.pi / 2),
            _gate("rz", (0,), theta=attributes["phi"]),
        ]
    if name == "quantum.u3":
        return [
            _gate("rz", (0,), theta=attributes["lbd"]),
            _gate("ry", (0,), theta=attributes["theta"]),
            _gate("rz", (0,), theta=attributes["phi"]),
        ]
    theta = attributes["theta"]
    if name == "quantum.crz":
        return [
            _gate("rz", (1,), theta=_scale(theta, 0.5)),
            _gate("cx", (0, 1)),
            _gate("rz", (1,), theta=_scale(theta, -0.5)),
            _gate("cx", (0, 1)),
        ]
    if name == "quantum.cry":
        return [
            _gate("ry", (1,), theta=_scale(theta, 0.5)),
            _gate("cx", (0, 1)),
            _gate("ry", (1,), theta=_scale(theta, -0.5)),
            _gate("cx", (0, 1)),
        ]
    if name == "quantum.crx":
        return [
            *_h(1),
            _gate("rz", (1,), theta=_scale(theta, 0.5)),
            _gate("cx", (0, 1)),
            _gate("rz", (1,), theta=_scale(theta, -0.5)),
            _gate("cx", (0, 1)),
            *_h(1),
        ]
    if name == "quantum.cphase":
        half = _scale(theta, 0.5)
        return [
            _gate("rz", (0,), theta=half),
            _gate("cx", (0, 1)),
            _gate("rz", (1,), theta=_scale(theta, -0.5)),
            _gate("cx", (0, 1)),
            _gate("rz", (1,), theta=half),
        ]
    if name == "quantum.rzz":
        return _rzz(theta, 0, 1)
    if name == "quantum.rxx":
        return [*_h(0), *_h(1), *_rzz(theta, 0, 1), *_h(0), *_h(1)]
    if name == "quantum.ryy":
        return [
            _gate("rx", (0,), theta=-math.pi / 2),
            _gate("rx", (1,), theta=-math.pi / 2),
            *_rzz(theta, 0, 1),
            _gate("rx", (0,), theta=math.pi / 2),
            _gate("rx", (1,), theta=math.pi / 2),
        ]
    return None


def _next_intermediate_index(module: QuantumModule) -> int:
    indices = [
        value.id.index
        for block in module.body.blocks
        for operation in block.operations
        for value in (*operation.operands, *operation.results)
        if value.id.scope == "decomposition"
    ]
    return max(indices, default=-1) + 1


def _materialize(
    operation: Operation, specs: list[_Spec], next_index: int
) -> tuple[list[Operation], int]:
    if not specs:
        raise ValueError("identity must be removed before target decomposition")
    last_touch = {
        wire: max(index for index, (_, wires, _) in enumerate(specs) if wire in wires)
        for wire in range(len(operation.operands))
    }
    current = list(operation.operands)
    emitted = []
    for spec_index, (name, wires, attributes) in enumerate(specs):
        operands = tuple(current[wire] for wire in wires)
        results = []
        for wire, operand in zip(wires, operands, strict=True):
            if last_touch[wire] == spec_index:
                result = operation.results[wire]
            else:
                result = ValueRef(ValueId(next_index, "decomposition"), operand.type)
                next_index += 1
            current[wire] = result
            results.append(result)
        emitted.append(
            Operation(
                name,
                operands,
                tuple(results),
                attributes,
                location=operation.location,
            )
        )
    return emitted, next_index


class DecomposeToTargetGateSetPass:
    descriptor = PassDescriptor(
        "decompose_to_universal_rx_ry_rz_cx",
        "1.0",
        options={
            "profile": UNIVERSAL_RX_RY_RZ_CX_V1.name,
            "profile_version": UNIVERSAL_RX_RY_RZ_CX_V1.version,
            "native_operations": UNIVERSAL_RX_RY_RZ_CX_V1.native_operations,
            "certified_source_operations": (
                UNIVERSAL_RX_RY_RZ_CX_V1.certified_source_operations
            ),
            "precondition": "phase2_batch_a_identity_removal",
        },
        program_identity_policy="transform",
    )

    def run(self, module: QuantumModule) -> PassResult:
        if len(module.body.blocks) != 1:
            return PassResult(
                module, diagnostics=(_diagnostic("one block is required"),)
            )
        block = module.body.blocks[0]
        output: list[Operation] = []
        next_index = _next_intermediate_index(module)
        rewritten = 0
        for operation in block.operations:
            specs = _expand(operation)
            if specs is None:
                return PassResult(
                    module,
                    diagnostics=(
                        _diagnostic(
                            f"operation {operation.name!r} has no certified decomposition "
                            f"for profile {UNIVERSAL_RX_RY_RZ_CX_V1.name!r}"
                        ),
                    ),
                )
            if len(specs) == 1 and specs[0][0] == operation.name:
                output.append(operation)
                continue
            try:
                materialized, next_index = _materialize(operation, specs, next_index)
            except ValueError as exc:
                return PassResult(module, diagnostics=(_diagnostic(str(exc)),))
            output.extend(materialized)
            rewritten += 1
        if not rewritten:
            return PassResult(
                module,
                preserved_analyses=frozenset({"def_use", "qubit_lifetime"}),
                statistics={"operations_rewritten": 0},
            )
        candidate = QuantumModule(
            Region((Block(block.arguments, tuple(output)),)),
            revision=module.revision + 1,
        )
        verification = verify_module(candidate, circuit_ir_v1_schema_registry())
        if not verification.ok:
            return PassResult(
                candidate,
                changed=True,
                diagnostics=verification.diagnostics,
                statistics={"operations_rewritten": rewritten},
            )
        return PassResult(
            candidate,
            changed=True,
            statistics={
                "operations_rewritten": rewritten,
                "operations_emitted": len(output),
            },
        )


__all__ = [
    "DecomposeToTargetGateSetPass",
    "TargetGateSetProfile",
    "UNIVERSAL_RX_RY_RZ_CX_V1",
]
