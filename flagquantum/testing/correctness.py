"""Deterministic generated correctness cases and reproducible failures."""

from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.compiler.operator_lowering import DEFAULT_LOWERING_REGISTRY
from flagquantum.compiler.qcis import emit_qcis
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.noise import (
    amplitude_damping_channel,
    bit_flip_channel,
    depolarizing_channel,
    phase_flip_channel,
)
from flagquantum.runtime.backends.jax import run_jax_sharded_statevector
from flagquantum.simulation.density_matrix import density_matrix_from_ir

CERTIFICATION_VERSION = "flagquantum_correctness_v1"


@dataclass(frozen=True)
class TolerancePolicy:
    version: str = "tolerance_v1"
    dtype: str = "complex64"
    method: str = "exact"
    max_depth: int = 16
    approximation: str = "none"
    atol: float = 1e-5
    rtol: float = 1e-5

    def accepts(
        self, *, dtype: str, method: str, depth: int, approximation: str
    ) -> bool:
        return (
            dtype == self.dtype
            and method == self.method
            and depth <= self.max_depth
            and approximation == self.approximation
        )


@dataclass(frozen=True)
class CertificationCase:
    operator: str
    backend: str
    comparison: str
    tolerance_version: str


@dataclass(frozen=True)
class CertificationResult:
    case: CertificationCase
    executed: bool
    passed: bool
    detail: str


def certification_matrix() -> tuple[CertificationCase, ...]:
    """Cover every declared supported operator/backend pair exactly once."""

    manifest = DEFAULT_LOWERING_REGISTRY.manifest()["backends"]
    cases = []
    numerical = {"pytorch", "jax", "mps", "tensor_network"}
    for backend, capability in manifest.items():
        for operator in capability["supported"]:
            cases.append(
                CertificationCase(
                    operator=operator,
                    backend=backend,
                    comparison="numerical" if backend in numerical else "round_trip",
                    tolerance_version="tolerance_v1",
                )
            )
    return tuple(cases)


def generate_circuit_ir(*, seed: int, n_wires: int = 3, depth: int = 8) -> CircuitIR:
    """Generate a minimal deterministic unitary IR inside the common envelope."""

    if n_wires < 3 or depth < 0:
        raise ValueError("generated envelope requires n_wires >= 3 and depth >= 0")
    rng = random.Random(seed)
    schemas = tuple(schema for schema in OPERATOR_SCHEMAS.values() if schema.unitary)
    instructions = []
    for _ in range(depth):
        schema = rng.choice(schemas)
        wires = tuple(rng.sample(range(n_wires), schema.arity))
        params = {name: rng.uniform(-3.14159, 3.14159) for name in schema.parameters}
        instructions.append(Instruction(name=schema.opcode, wires=wires, params=params))
    return CircuitIR(n_wires=n_wires, instructions=tuple(instructions))


def execute_certification_case(case: CertificationCase) -> CertificationResult:
    """Execute one declared lowering through its real local backend surface."""

    import torch

    import flagquantum as fq
    import flagquantum.backends as fqb
    import flagquantum.deployment as fqd

    schema = OPERATOR_SCHEMAS[case.operator]
    params = {name: 0.23 for name in schema.parameters}
    instruction = Instruction(
        name=case.operator,
        wires=tuple(range(schema.arity)),
        params=params,
    )
    ir = CircuitIR(n_wires=max(3, schema.arity), instructions=(instruction,))
    DEFAULT_LOWERING_REGISTRY.validate_ir(case.backend, ir)
    if schema.channel:
        factories = {
            "bit_flip": bit_flip_channel,
            "phase_flip": phase_flip_channel,
            "depolarizing": depolarizing_channel,
            "amplitude_damping": amplitude_damping_channel,
        }
        channel = factories[case.operator](0.23)
        channel_ir = CircuitIR(
            n_wires=3,
            instructions=(
                Instruction(
                    name=case.operator,
                    wires=(0,),
                    matrix=channel.kraus,
                    metadata={"is_channel": True},
                ),
            ),
        )
        rho = density_matrix_from_ir(channel_ir)
        passed = torch.allclose(
            torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real,
            torch.ones(1),
            atol=1e-6,
        )
    else:
        dense = fq.Circuit.from_ir(ir).state()
        if case.backend == "pytorch":
            candidate = dense
        elif case.backend == "mps":
            candidate = fqb.run_mps(ir, max_bond=None, cutoff=0.0).to_statevector()
        elif case.backend == "tensor_network":
            candidate = fqb.run_tensor_network(ir).state()
        elif case.backend == "jax":
            candidate = run_jax_sharded_statevector(ir, world_size=1).state()
        elif case.backend == "qasm":
            text = emit_openqasm(ir)
            passed = bool(text.strip())
            return CertificationResult(case, True, passed, "qasm_serialization")
        elif case.backend == "qcis":
            text = emit_qcis(ir)
            passed = bool(text.strip())
            return CertificationResult(case, True, passed, "qcis_serialization")
        elif case.backend == "provider":
            package = fqd.create_deployment_package(ir, optimize=False)
            result = fqd.LocalSimulatorProvider().run(package)
            passed = (
                bool(package.qasm.strip())
                and sum(result.counts.values()) == package.shots
            )
            return CertificationResult(case, True, passed, "provider_packaging")
        else:  # pragma: no cover - registry controls this branch
            raise ValueError(f"unknown certification backend {case.backend!r}")
        passed = torch.allclose(torch.as_tensor(candidate), dense, atol=1e-5, rtol=1e-5)
    return CertificationResult(case, True, bool(passed), "numerical_execution")


@dataclass(frozen=True)
class FailureArtifact:
    """Minimal JSON failure record sufficient for exact local reproduction."""

    certification_version: str
    seed: int
    backend: str
    property_name: str
    ir: dict[str, Any]
    environment: dict[str, str]
    error: str

    @classmethod
    def capture(
        cls,
        *,
        seed: int,
        backend: str,
        property_name: str,
        ir: CircuitIR,
        error: Exception,
        fails: Callable[[CircuitIR], bool] | None = None,
    ) -> "FailureArtifact":
        minimized = ir
        if fails is not None and fails(ir):
            instructions = list(ir.instructions)
            changed = True
            while changed and len(instructions) > 1:
                changed = False
                for index in range(len(instructions)):
                    candidate = CircuitIR(
                        n_wires=ir.n_wires,
                        instructions=tuple(
                            instructions[:index] + instructions[index + 1 :]
                        ),
                    )
                    if fails(candidate):
                        instructions.pop(index)
                        changed = True
                        break
            minimized = CircuitIR(n_wires=ir.n_wires, instructions=tuple(instructions))
        return cls(
            certification_version=CERTIFICATION_VERSION,
            seed=seed,
            backend=backend,
            property_name=property_name,
            ir=minimized.to_dict(),
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            error=f"{type(error).__name__}: {error}",
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
