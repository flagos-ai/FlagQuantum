"""Emulate a declared target locally without submitting a task.

This module is the target-directed local entry point. It takes the same
``CloudBackendProfile`` a remote adapter would submit against, compiles the
program for that target with the same passes a submission uses, proves that the
emitted payload describes the compiled program, and then executes that compiled
program on the local CPU. Nothing here contacts a provider, and no function in
this module submits work or decodes a provider result.

Four stages, each keeping its own evidence:

1. **Declare the target.** A ``TargetCapabilitySnapshot`` is assembled from two
   sources. The local device and precision facts come from the observed CPU
   probe in ``flagquantum.compute``. The qubit capacity, native gate set, result
   formats, and program limits are *declared* facts taken from the profile, and
   the declaration is pinned by a digest of the profile payload. The snapshot's
   target identity names an emulator, so a reader cannot mistake it for a
   snapshot of the device that supplied the declaration.
2. **Compile for the target.** Topology routing, native-gate legalization,
   directed-CX repair, lowering-capability validation, capability matching, and
   scheduling run exactly as they do for a submission, which is why the
   diagnostics below are the target's own.
3. **Emit and verify.** The payload for the chosen emission profile is produced
   and then re-parsed, so the emulation reports what the target would have been
   sent together with an identity that ties the text back to the compiled
   program.
4. **Execute locally.** The compiled program runs on the local CPU. Noise is
   opt-in: without it the result carries the noiseless execution of the compiled
   program, and with it the result also carries a second execution. That second
   execution names where its channels came from -- a caller's ``noise_model`` or
   a target's ``device_profile`` calibration -- the representation the planner
   selected, and whether the channel evolution is exact, so an approximate
   representation is reported as approximate instead of passing for the device.

**What is not here.** Local execution is not the target. The declared facts
describe what the profile advertises; they are not measurements of hardware. A
noiseless record is a simulation of the compiled program and not of the device.
Compilation, emission, and conformance failures raise the compiler's own
``TargetLegalizationError``, ``TargetEmissionError``, and
``TargetConformanceError``, so a caller reads the same diagnostic it would read
before submitting.

A noise model is applied to the *compiled* program, so its rules name the
target's native opcodes plus ``swap`` when routing inserted one. The returned
diagnostic lists the native gate set and every decomposition that produced it.
A ``device_profile`` is applied to the same program and is refused when it
cannot time an instruction the compiled program contains, rather than raising a
bare lookup error from inside the noise lowering.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import torch

from ..compiler.directed_topology import DirectedCouplingMap
from ..compiler.native_gate_legalization import GateDecompositionRecord
from ..compiler.physical_plan import build_physical_circuit_plan
from ..compiler.routing import CouplingMap
from ..compiler.target_conformance import verify_target_emission
from ..compiler.target_emission import (
    EMISSION_PROFILES,
    TargetEmissionProfile,
    emit_legalized_target,
)
from ..compiler.target_legalization import legalize_circuit_for_target
from ..compute.cpu_target_capabilities import probe_local_cpu_target_capabilities
from ..core.ir import CircuitIR, MeasurementNode, ensure_circuit_ir
from ..core.operator_schema import canonical_opcode, get_operator_schema
from ..core.target_capabilities import (
    CapabilityFact,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from ..deployment.cloud import CloudBackendProfile
from ..errors import ExecutionError
from ..noise import NoiseModel
from ..noise.device_profile import DeviceNoiseProfile
from ..observables import lower_outputs, samples
from ..runtime.execution import run as run_program
from ..runtime.execution_plan import ExecutionPlan
from ..runtime.options import ExecutionOptions
from ..runtime.result import MeasurementResult

LOCAL_TARGET_EMULATION_SCHEMA = "flagquantum.local_target_emulation.v1"
EMULATED_TARGET_ADAPTER_VERSION = "1.0"
EMULATED_TARGET_PROVIDER = "flagquantum.emulated_target"
#: The class this module claims, so an emulation snapshot is never read as a
#: snapshot of the device whose profile supplied the declared facts.
EMULATED_TARGET_CLASS = "emulated_target"
#: The source kind of the declared facts. The remote adapters observe a
#: provider's device listing; this module observes the profile that listing
#: produced, which is why a declaration can carry observable-grade evidence.
EMULATED_TARGET_DECLARATION_SOURCE_KIND = "emulated_target_profile_observation"
#: Result formats the emulator offers. Both are produced from one sample
#: tensor: the histogram below reproduces the executor's own counts encoding.
_EMULATED_RESULT_FORMATS = ("counts", "samples")
#: Request kinds this entry point can execute. The static text emission profile
#: refuses every other measurement request, so the refusal is stated here first.
_SUPPORTED_REQUEST_KINDS = frozenset({"samples", "counts"})
#: Representation that truncates the state, so an execution record can say so.
_APPROXIMATE_REPRESENTATION = "mps"
#: Where a noisy record's channels came from. A caller's model and a device
#: calibration are different claims about the world, and the record says which
#: one it is carrying rather than leaving a reader to guess.
NOISE_SOURCE_NONE = "none"
NOISE_SOURCE_CALLER_MODEL = "caller_model"
NOISE_SOURCE_DEVICE_PROFILE = "device_profile"


class TargetEmulationError(ExecutionError):
    """A local target emulation could not be prepared or executed."""


@dataclass(frozen=True)
class EmulatedExecutionRecord:
    """One local execution of the target-compiled program.

    ``noise_source`` is what the channels represent: nothing, a model the caller
    wrote, or the calibration a device profile declares. The representation is
    what the planner chose; ``exact_channel`` says whether that representation
    evolved the channel exactly, and ``approximate`` whether the state itself was
    truncated.
    """

    representation: str
    exact_channel: bool
    approximate: bool
    noise_source: str
    noise_model_identity: str | None
    counts: Mapping[str, int]
    shots: int
    seed: int | None


@dataclass(frozen=True)
class TargetCompilationDiagnostic:
    """What the target's own passes did to the program.

    Every identity is the one the compiling stage published, unmodified. Those
    identities fold in the target snapshot, and the local probe stamps each
    snapshot with its capture time, so identities are evidence about one
    emulation rather than a value two emulations can be expected to share. The
    emitted program and the histogram are reproducible; the identities are
    per-capture.
    """

    backend: str
    emission_profile: str
    media_type: str
    target_snapshot_id: str
    native_opcodes: tuple[str, ...]
    decompositions: tuple[GateDecompositionRecord, ...]
    source_instruction_count: int
    routed_instruction_count: int
    inserted_swap_count: int
    routing_strategy: str | None
    direction_semantics: str
    final_logical_to_physical: tuple[int, ...]
    #: Empty unless the target is directed: an undirected run has no allocated
    #: result projection to report.
    logical_result_physical_slots: tuple[int, ...]
    allocation_identity: str | None
    physical_plan_version: str
    logical_wire_count: int
    physical_slot_count: int
    schedule_depth: int
    maximum_parallel_width: int
    schedule_identity: str
    target_legalization_identity: str
    emission_identity: str
    conformance_identity: str
    parsed_operation_count: int

    @property
    def rewritten(self) -> bool:
        """True when at least one instruction needed a native-gate rewrite."""

        return bool(self.decompositions)


@dataclass(frozen=True)
class LocalTargetEmulation:
    """A target-compiled program, its diagnostics, and its local results."""

    target_identity: TargetIdentity
    target_snapshot_id: str
    emission_profile: str
    media_type: str
    emitted_program: str
    declared_native_gates: tuple[str, ...]
    declared_logical_capacity: int
    precision: str
    shots: int
    seed: int | None
    diagnostic: TargetCompilationDiagnostic
    noiseless: EmulatedExecutionRecord
    noisy: EmulatedExecutionRecord | None = None
    schema: str = LOCAL_TARGET_EMULATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Serialize the emulation as evidence, without any provider claim."""

        return {
            "schema": self.schema,
            "target_identity": {
                "target_id": self.target_identity.target_id,
                "target_class": self.target_identity.target_class,
                "provider": self.target_identity.provider,
                "provider_version": self.target_identity.provider_version,
                "target_revision": self.target_identity.target_revision,
                "environment_id": self.target_identity.environment_id,
            },
            "target_snapshot_id": self.target_snapshot_id,
            "emission_profile": self.emission_profile,
            "media_type": self.media_type,
            "declared_native_gates": list(self.declared_native_gates),
            "declared_logical_capacity": self.declared_logical_capacity,
            "precision": self.precision,
            "shots": self.shots,
            "seed": self.seed,
            "emitted_program_sha256": hashlib.sha256(
                self.emitted_program.encode("utf-8")
            ).hexdigest(),
            "diagnostic": {
                "backend": self.diagnostic.backend,
                "source_instruction_count": self.diagnostic.source_instruction_count,
                "routed_instruction_count": self.diagnostic.routed_instruction_count,
                "inserted_swap_count": self.diagnostic.inserted_swap_count,
                "routing_strategy": self.diagnostic.routing_strategy,
                "direction_semantics": self.diagnostic.direction_semantics,
                "final_logical_to_physical": list(
                    self.diagnostic.final_logical_to_physical
                ),
                "logical_result_physical_slots": list(
                    self.diagnostic.logical_result_physical_slots
                ),
                "allocation_identity": self.diagnostic.allocation_identity,
                "decompositions": [
                    {
                        "instruction_index": item.instruction_index,
                        "source_opcode": item.source_opcode,
                        "replacement_opcodes": list(item.replacement_opcodes),
                    }
                    for item in self.diagnostic.decompositions
                ],
                "physical_plan_version": self.diagnostic.physical_plan_version,
                "physical_slot_count": self.diagnostic.physical_slot_count,
                "schedule_depth": self.diagnostic.schedule_depth,
                "maximum_parallel_width": self.diagnostic.maximum_parallel_width,
                "schedule_identity": self.diagnostic.schedule_identity,
                "target_legalization_identity": (
                    self.diagnostic.target_legalization_identity
                ),
                "emission_identity": self.diagnostic.emission_identity,
                "conformance_identity": self.diagnostic.conformance_identity,
                "parsed_operation_count": self.diagnostic.parsed_operation_count,
            },
            "noiseless": _record_payload(self.noiseless),
            "noisy": None if self.noisy is None else _record_payload(self.noisy),
        }


def _record_payload(record: EmulatedExecutionRecord) -> dict[str, Any]:
    return {
        "representation": record.representation,
        "exact_channel": record.exact_channel,
        "approximate": record.approximate,
        "noise_source": record.noise_source,
        "noise_model_identity": record.noise_model_identity,
        "shots": record.shots,
        "seed": record.seed,
        "counts": dict(record.counts),
    }


def _requested_kind(measurement: MeasurementNode) -> str:
    return str(measurement.metadata.get("fq_output_kind", measurement.kind))


def _prepare_program(
    ir: CircuitIR,
    *,
    shots: int,
    seed: int | None,
) -> CircuitIR:
    """Return the program carrying the terminal samples request emission needs.

    The static text emission profile can only describe a terminal
    full-register samples request. A program that asks for nothing gets one; a
    program that asks for counts already carries the identical request under a
    different result encoding, and the histogram this module reports from the
    sample tensor reproduces the executor's counts encoding exactly.
    """

    if not ir.measurements:
        requested = lower_outputs(samples(), n_wires=ir.n_wires, shots=shots, seed=seed)
        return replace(ir, measurements=tuple(requested or ()))
    if len(ir.measurements) != 1:
        raise TargetEmulationError(
            "target emulation requires exactly one terminal measurement request; "
            f"the program declares {len(ir.measurements)}"
        )
    measurement = ir.measurements[0]
    kind = _requested_kind(measurement)
    if kind not in _SUPPORTED_REQUEST_KINDS:
        supported = ", ".join(sorted(_SUPPORTED_REQUEST_KINDS))
        raise TargetEmulationError(
            f"target emulation supports {supported} requests; the program "
            f"declares {kind!r}"
        )
    expected = tuple(range(ir.n_wires))
    if tuple(measurement.wires) != expected:
        raise TargetEmulationError(
            "target emulation requires a terminal full-register request; the "
            f"program measures wires {tuple(measurement.wires)} of {ir.n_wires}"
        )
    if kind == "samples" and measurement.kind == "sample":
        return ir
    normalized = MeasurementNode(
        kind="sample",
        wires=tuple(measurement.wires),
        shots=measurement.shots,
        metadata={**measurement.metadata, "fq_output_kind": "samples"},
    )
    return replace(ir, measurements=(normalized,))


def _select_emission_profile(
    target: CloudBackendProfile,
    emission_profile: str | None,
) -> TargetEmissionProfile:
    if emission_profile is not None:
        selected = EMISSION_PROFILES.get(str(emission_profile).strip().lower())
        if selected is None:
            supported = ", ".join(sorted(EMISSION_PROFILES))
            raise TargetEmulationError(
                f"unsupported target emission profile {emission_profile!r}; "
                f"expected one of: {supported}"
            )
        return selected
    if target.supports_openqasm:
        return EMISSION_PROFILES["openqasm-3.0"]
    if target.supports_qcis:
        return EMISSION_PROFILES["qcis-1.0"]
    raise TargetEmulationError(
        f"target {target.provider}:{target.name} declares neither OpenQASM nor "
        "QCIS emission, so there is no text profile to emulate"
    )


def _declared_native_gates(target: CloudBackendProfile) -> tuple[str, ...]:
    if not target.basis_gates:
        raise TargetEmulationError(
            f"target {target.provider}:{target.name} does not declare basis "
            "gates, so there is no native gate set to legalize against"
        )
    declared: set[str] = set()
    for item in target.basis_gates:
        if not isinstance(item, str) or not item.strip():
            raise TargetEmulationError(
                "target basis gates must be non-empty gate names"
            )
        opcode = canonical_opcode(item)
        if get_operator_schema(opcode) is None:
            raise TargetEmulationError(
                f"target basis gate {item!r} names unknown FlagQuantum gate "
                f"{opcode!r}; a declared gate set must be expressed with the "
                "operator schemas the compiler can lower"
            )
        declared.add(opcode)
    return tuple(sorted(declared))


def _declaration_digest(
    target: CloudBackendProfile,
    *,
    emission_profile: str,
    native_gates: tuple[str, ...],
) -> str:
    coupling = target.coupling_map
    payload = {
        "provider": target.provider,
        "name": target.name,
        "n_qubits": target.n_qubits,
        "native_gates": list(native_gates),
        "coupling_edges": (
            [] if coupling is None else [list(edge) for edge in coupling.edges]
        ),
        "supports_openqasm": target.supports_openqasm,
        "supports_qcis": target.supports_qcis,
        "supports_dynamic_circuits": target.supports_dynamic_circuits,
        "dynamic_dialect": target.dynamic_dialect,
        "max_classical_bits": target.max_classical_bits,
        "is_simulator": target.is_simulator,
        "emission_profile": emission_profile,
    }
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:  # pragma: no cover - defensive
        raise TargetEmulationError(
            f"target profile payload is not serializable: {error}"
        ) from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _target_snapshot(
    target: CloudBackendProfile,
    *,
    precision: str,
    emission_profile: str,
    native_gates: tuple[str, ...],
    declared_maximum_program_operations: int,
    declared_maximum_shots: int,
) -> TargetCapabilitySnapshot:
    """Combine the observed local probe with the profile's declared facts."""

    # The probe states what the local execution can actually do: the device it
    # will use and the precision it will carry.
    probe = probe_local_cpu_target_capabilities(precision)
    digest = _declaration_digest(
        target, emission_profile=emission_profile, native_gates=native_gates
    )
    source = FactSource(
        kind=EMULATED_TARGET_DECLARATION_SOURCE_KIND,
        ref=f"emulated-target-{digest[:16]}",
    )
    facts = {fact.name: fact for fact in probe.facts}
    # `target.class` is restated because the probe's value names the local
    # runtime while this snapshot describes the emulated target.
    declarations: dict[str, Any] = {
        "target.class": EMULATED_TARGET_CLASS,
        "qubits.logical_capacity": target.n_qubits,
        "gates.native": native_gates,
        "limits.maximum_program_operations": declared_maximum_program_operations,
        "limits.maximum_shots": declared_maximum_shots,
        "measurements.results": _EMULATED_RESULT_FORMATS,
        "artifacts.profiles": (emission_profile,),
    }
    for name, value in declarations.items():
        facts[name] = CapabilityFact(
            name=name,
            value=value,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=source,
        )
    evidence_refs = tuple(probe.evidence_refs) + (
        EvidenceReference(
            evidence_id=source.ref,
            sha256=digest,
            level=EvidenceLevel.OBSERVABLE,
            scope=probe.scope,
        ),
    )
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=f"emulated-{target.provider}-{target.name}",
            target_class=EMULATED_TARGET_CLASS,
            provider=EMULATED_TARGET_PROVIDER,
            provider_version=EMULATED_TARGET_ADAPTER_VERSION,
            target_revision=digest,
            environment_id=probe.target_identity.environment_id,
        ),
        scope=probe.scope,
        captured_at=probe.captured_at,
        valid_until=probe.valid_until,
        facts=tuple(facts.values()),
        evidence_refs=evidence_refs,
    )


def _local_counts(measurement: MeasurementResult, *, shots: int) -> dict[str, int]:
    """Reduce one sample tensor to a logical-wire-ordered histogram.

    The executor reads the sample tensor's last axis in the wire order the
    measurement node carries, so a bitstring is built in that order and then
    placed by logical wire. With no topology legalization the two orders
    coincide, which is why this reproduces the executor's counts encoding
    exactly.
    """

    if measurement.kind != "sample":
        raise TargetEmulationError(
            f"local target emulation expected a samples result, got "
            f"{measurement.kind!r}"
        )
    value = measurement.value
    if not isinstance(value, torch.Tensor):
        raise TargetEmulationError(
            "local target emulation expected a samples tensor, got "
            f"{type(value).__name__}"
        )
    shape = tuple(value.shape)
    if len(shape) != 3 or shape[0] != 1:
        raise TargetEmulationError(
            "local target emulation expects one unbatched samples tensor; "
            f"got shape {shape}"
        )
    wires = tuple(measurement.wires)
    logical = measurement.metadata.get("fq_logical_wires")
    logical_wires = wires if logical is None else tuple(int(item) for item in logical)
    if len(logical_wires) != len(wires):
        raise TargetEmulationError(
            "local target emulation found a measurement whose logical result "
            "projection does not match its measured wires"
        )
    order = sorted(range(len(wires)), key=lambda index: logical_wires[index])
    histogram: Counter[str] = Counter()
    for row in value[0].detach().cpu().tolist():
        histogram["".join(str(int(row[index])) for index in order)] += 1
    if sum(histogram.values()) != shots:
        raise TargetEmulationError(
            f"local execution returned {sum(histogram.values())} samples for a "
            f"{shots}-shot request"
        )
    return dict(sorted(histogram.items()))


def _require_profile_coverage(profile: DeviceNoiseProfile, program: CircuitIR) -> None:
    """Refuse a calibration that cannot time every compiled instruction.

    The noise lowering asks the profile for each gate's duration, and it raises
    a bare lookup error for a gate the profile does not cover. Checking here
    names the gate and the profile instead, because a caller who handed over a
    calibration wants to know which instruction defeated it.
    """

    for instruction in program:
        if instruction.metadata.get("is_channel"):
            continue
        try:
            profile.duration_for(instruction.name, instruction.wires)
        except ValueError as error:
            raise TargetEmulationError(
                f"device profile {profile.source!r} has no duration for "
                f"{instruction.name!r} on wires {instruction.wires}, so the "
                "emulated target cannot be timed"
            ) from error


def _execute(
    program: CircuitIR,
    *,
    precision: str,
    shots: int,
    seed: int | None,
    memory_limit_bytes: int | None,
    allow_approximate: bool,
    noise_model: NoiseModel | None,
    noise_source: str,
) -> EmulatedExecutionRecord:
    result = run_program(
        program,
        options=ExecutionOptions(
            mode="auto",
            device="cpu",
            precision=precision,
            shots=shots,
            seed=seed,
            memory_limit_bytes=memory_limit_bytes,
            allow_approximate=allow_approximate,
        ),
        noise_model=noise_model,
    )
    plan = result.plan
    if not isinstance(plan, ExecutionPlan):
        raise TargetEmulationError(
            "local target emulation expected an execution plan alongside the "
            f"result, got {type(plan).__name__}"
        )
    noisy_plan = plan.noisy_execution_plan
    representation = (
        plan.state_mode if noisy_plan is None else noisy_plan.representation
    )
    if not result.measurements:
        raise TargetEmulationError(
            "local target emulation produced no measurement result"
        )
    return EmulatedExecutionRecord(
        representation=representation,
        exact_channel=(
            noisy_plan is not None and noisy_plan.evolution == "exact_channel"
        ),
        approximate=representation == _APPROXIMATE_REPRESENTATION,
        noise_source=noise_source,
        noise_model_identity=(
            None if noisy_plan is None else noisy_plan.noise_model_identity
        ),
        counts=_local_counts(result.measurements[0], shots=shots),
        shots=shots,
        seed=seed,
    )


def emulate(
    program: object,
    *,
    target: CloudBackendProfile,
    shots: int = 1024,
    seed: int | None = None,
    emission_profile: str | None = None,
    noise_model: NoiseModel | None = None,
    device_profile: DeviceNoiseProfile | None = None,
    coupling_map: CouplingMap | DirectedCouplingMap | None = None,
    routing_strategy: str = "auto",
    initial_layout: tuple[int, ...] | None = None,
    evaluated_at: datetime | None = None,
    memory_limit_bytes: int | None = None,
    allow_approximate: bool = False,
    declared_maximum_program_operations: int = 4096,
    declared_maximum_shots: int = 1_000_000,
) -> LocalTargetEmulation:
    """Compile ``program`` for ``target`` and run it locally.

    The returned diagnostic is the target's own compilation evidence, the
    emitted program is what the target would have received, and the execution
    records are local CPU results. ``coupling_map`` overrides the profile's
    connectivity when it is given, which is how a simulator profile with no
    advertised topology is emulated against an explicit one.

    ``noise_model`` and ``device_profile`` are alternatives, not layers: the
    first is the caller's own channels, the second is a calibration the target
    declares, and the noisy record names which one it carries. Passing neither
    returns only the noiseless record.
    """

    if not isinstance(target, CloudBackendProfile):
        raise TypeError("target must be a CloudBackendProfile")
    if not isinstance(target.provider, str) or not target.provider.strip():
        raise TargetEmulationError("target profile requires a non-empty provider")
    if not isinstance(target.name, str) or not target.name.strip():
        raise TargetEmulationError("target profile requires a non-empty name")
    if type(shots) is not int:
        raise TypeError("shots must be an integer")
    if shots < 1:
        raise ValueError("shots must be >= 1")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a non-negative integer or None")
    if type(allow_approximate) is not bool:
        raise TypeError("allow_approximate must be a boolean")
    if noise_model is not None and not isinstance(noise_model, NoiseModel):
        raise TypeError("noise_model must be a NoiseModel or None")
    if device_profile is not None and not isinstance(
        device_profile, DeviceNoiseProfile
    ):
        raise TypeError("device_profile must be a DeviceNoiseProfile or None")
    if noise_model is not None and device_profile is not None:
        raise TypeError(
            "pass either noise_model or device_profile, not both: the noisy "
            "record states which one produced its channels"
        )
    if coupling_map is not None and not isinstance(
        coupling_map, (CouplingMap, DirectedCouplingMap)
    ):
        raise TypeError(
            "coupling_map must be a CouplingMap, a DirectedCouplingMap or None"
        )
    if type(declared_maximum_program_operations) is not int:
        raise TypeError("declared_maximum_program_operations must be an integer")
    if declared_maximum_program_operations < 1:
        raise ValueError("declared_maximum_program_operations must be >= 1")
    if type(declared_maximum_shots) is not int:
        raise TypeError("declared_maximum_shots must be an integer")
    if shots > declared_maximum_shots:
        raise TargetEmulationError(
            f"target emulation requests {shots} shots, above the declared "
            f"target maximum of {declared_maximum_shots}"
        )

    source_ir = ensure_circuit_ir(program)
    precision = str(source_ir.dtype)
    prepared = _prepare_program(source_ir, shots=shots, seed=seed)
    selected = _select_emission_profile(target, emission_profile)
    native_gates = _declared_native_gates(target)
    snapshot = _target_snapshot(
        target,
        precision=precision,
        emission_profile=selected.name,
        native_gates=native_gates,
        declared_maximum_program_operations=declared_maximum_program_operations,
        declared_maximum_shots=declared_maximum_shots,
    )
    effective_coupling = target.coupling_map if coupling_map is None else coupling_map
    legalization = legalize_circuit_for_target(
        prepared,
        backend=selected.backend,
        snapshot=snapshot,
        evaluated_at=evaluated_at,
        coupling_map=effective_coupling,
        routing_strategy=routing_strategy,
        initial_layout=initial_layout,
    )
    physical_plan = build_physical_circuit_plan(
        legalization, coupling_map=effective_coupling
    )
    emission = emit_legalized_target(legalization, profile=selected.name)
    conformance = verify_target_emission(emission, legalization)
    topology = legalization.topology_legalization
    diagnostic = TargetCompilationDiagnostic(
        backend=legalization.backend,
        emission_profile=selected.name,
        media_type=emission.media_type,
        target_snapshot_id=legalization.target_snapshot_id,
        native_opcodes=legalization.native_gate_legalization.native_opcodes,
        decompositions=legalization.native_gate_legalization.decompositions,
        source_instruction_count=len(prepared.instructions),
        routed_instruction_count=(
            len(prepared.instructions)
            if topology is None
            else topology.routed_instruction_count
        ),
        inserted_swap_count=0 if topology is None else topology.inserted_swap_count,
        routing_strategy=None if topology is None else topology.strategy,
        direction_semantics=(
            "undirected" if topology is None else topology.direction_semantics
        ),
        final_logical_to_physical=physical_plan.final_logical_to_physical,
        logical_result_physical_slots=physical_plan.logical_result_physical_slots,
        allocation_identity=physical_plan.allocation_identity,
        physical_plan_version=physical_plan.version,
        logical_wire_count=physical_plan.logical_wire_count,
        physical_slot_count=physical_plan.physical_slot_count,
        schedule_depth=legalization.schedule.depth,
        maximum_parallel_width=legalization.schedule.maximum_parallel_width,
        schedule_identity=legalization.schedule.schedule_identity,
        target_legalization_identity=legalization.legalization_identity,
        emission_identity=emission.emission_identity,
        conformance_identity=conformance.conformance_identity,
        parsed_operation_count=conformance.parsed_operation_count,
    )
    effective_noise: NoiseModel | None
    if device_profile is not None:
        _require_profile_coverage(device_profile, legalization.program)
        effective_noise = NoiseModel.from_device_profile(device_profile)
        noise_source = NOISE_SOURCE_DEVICE_PROFILE
    elif noise_model is not None:
        effective_noise = noise_model
        noise_source = NOISE_SOURCE_CALLER_MODEL
    else:
        effective_noise = None
        noise_source = NOISE_SOURCE_NONE
    return LocalTargetEmulation(
        target_identity=snapshot.target_identity,
        target_snapshot_id=snapshot.snapshot_id,
        emission_profile=selected.name,
        media_type=emission.media_type,
        emitted_program=emission.text,
        declared_native_gates=native_gates,
        declared_logical_capacity=target.n_qubits,
        precision=precision,
        shots=shots,
        seed=seed,
        diagnostic=diagnostic,
        noiseless=_execute(
            legalization.program,
            precision=precision,
            shots=shots,
            seed=seed,
            memory_limit_bytes=memory_limit_bytes,
            allow_approximate=allow_approximate,
            noise_model=None,
            noise_source=NOISE_SOURCE_NONE,
        ),
        noisy=(
            None
            if effective_noise is None
            else _execute(
                legalization.program,
                precision=precision,
                shots=shots,
                seed=seed,
                memory_limit_bytes=memory_limit_bytes,
                allow_approximate=allow_approximate,
                noise_model=effective_noise,
                noise_source=str(noise_source),
            )
        ),
    )


__all__ = (
    "EMULATED_TARGET_ADAPTER_VERSION",
    "EMULATED_TARGET_CLASS",
    "EMULATED_TARGET_DECLARATION_SOURCE_KIND",
    "EMULATED_TARGET_PROVIDER",
    "LOCAL_TARGET_EMULATION_SCHEMA",
    "NOISE_SOURCE_CALLER_MODEL",
    "NOISE_SOURCE_DEVICE_PROFILE",
    "NOISE_SOURCE_NONE",
    "EmulatedExecutionRecord",
    "LocalTargetEmulation",
    "TargetCompilationDiagnostic",
    "TargetEmulationError",
    "emulate",
)
