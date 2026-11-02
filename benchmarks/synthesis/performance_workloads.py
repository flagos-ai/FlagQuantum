"""Fixed PF-00 workload adapters and independent verification, outside timers."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from flagquantum.compiler._construction.frontend_python import Definition
    from flagquantum.services.synthesis import Program

from .expressions import Case, corpus


class Limits(TypedDict):
    max_qubits: int
    basis: tuple[str, ...] | None
    topology: Literal["line"] | None
    memory_limit_bytes: int


@dataclass(frozen=True)
class Workload:
    case: str
    frontend: Literal["scalar", "qfunc", "qmod"] = "scalar"
    variant: Literal["inline", "helper", "exchanged"] = "inline"
    target: Literal["logical", "basis", "line"] = "logical"
    refused: bool = False

    def __post_init__(self) -> None:
        if (
            self.frontend not in {"scalar", "qfunc", "qmod"}
            or self.variant not in {"inline", "helper", "exchanged"}
            or self.target not in {"logical", "basis", "line"}
            or type(self.refused) is not bool
        ):
            raise ValueError("unknown PF-00 workload profile")
        if self.case not in cases() or (
            self.variant != "inline" and self.case != "energy-2x2"
        ):
            raise ValueError("unknown fixed expression or spelling variant")

    @property
    def identifier(self) -> str:
        return ":".join(
            (
                self.case,
                self.frontend,
                self.variant,
                self.target,
                "refused" if self.refused else "success",
            )
        )


def cases() -> dict[str, Case]:
    from .expressions import EDGES

    result = {case.identifier: case for case in corpus()}
    expression = next(e for e in EDGES if e.name == "constant_product")
    for width in (4, 5):
        case = Case(expression, (width,))
        result[case.identifier] = case
    return result


def workloads() -> tuple[Workload, ...]:
    rows = [Workload(case.identifier) for case in corpus()]
    supplemental = (
        "energy-2x2",
        "energy-8x8",
        "shared_square-8x8",
        "square-2",
        "square-3",
        "constant_product-4",
        "constant_product-5",
        "add-1x3",
        "multiply_add-1x1x1",
    )
    rows.extend(
        Workload(name, frontend)
        for name in supplemental
        for frontend in ("qfunc", "qmod")
    )
    rows.extend(Workload(name) for name in ("constant_product-4", "constant_product-5"))
    rows.extend(
        Workload("energy-2x2", frontend, target=target)
        for frontend in ("scalar", "qfunc", "qmod")
        for target in ("basis", "line")
    )
    rows.extend(
        Workload("energy-2x2", frontend, variant=variant)
        for frontend in ("scalar", "qfunc", "qmod")
        for variant in ("helper", "exchanged")
    )
    rows.extend(
        Workload("energy-2x2", frontend, refused=True)
        for frontend in ("scalar", "qfunc", "qmod")
    )
    return tuple(rows)


class Prepared:
    """Build captured source before E0; retain source bytes/hash with each trial."""

    def __init__(self, workload: Workload) -> None:
        import flagquantum as fq
        from flagquantum import synth

        from .arithmetic_baseline import qmod_source
        from .expressions import exchanged_energy, helper_energy, square_helper

        self.workload = workload
        self.case = cases()[workload.case]
        self.folder = tempfile.TemporaryDirectory(prefix="fq-pf00-source-")
        self.fq = fq
        self.synth = synth
        names = self.case.expression.arguments
        ports = ", ".join(f"{name}: Output[QNum]" for name in (*names, "result"))
        setup = "\n".join(
            f"    allocate({width}, {name})\n    hadamard_transform({name})"
            for name, width in zip(names, self.case.widths, strict=True)
        )
        expr = self.case.expression.text
        if workload.variant == "exchanged":
            expr = "y+x*x"
        helper = ""
        computation = "    result |= " + expr
        if workload.variant == "helper":
            helper = "@synth.qfunc\ndef energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:\n    result |= x*x+y\n\n"
            computation = "    energy(x, y, result)"
        native = (
            "from flagquantum import synth\nfrom flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform\n\n"
            + helper
            + f"@synth.qfunc\ndef main({ports}) -> None:\n{setup}\n{computation}\n"
        )
        self.source = native if workload.frontend == "qfunc" else qmod_source(self.case)
        self.function: Definition | str = self.case.expression.function
        self.helpers: tuple[Definition, ...] = ()
        if workload.frontend == "scalar":
            if workload.variant == "helper":
                self.function = helper_energy
                self.helpers = (square_helper,)
            elif workload.variant == "exchanged":
                self.function = exchanged_energy
            import inspect

            self.source = inspect.getsource(self.function)
        elif workload.frontend == "qmod":
            if workload.variant == "exchanged":
                self.source = self.source.replace("x*x+y", "y+x*x")
            elif workload.variant == "helper":
                self.source = (
                    "qfunc energy(const x:qnum<2>,const y:qnum<2>,output result:qnum) {result=x*x+y;}\n"
                    + self.source.replace("result=x*x+y;", "energy(x,y,result);")
                )
            self.function = self.source
        else:
            path = Path(self.folder.name) / "native.py"
            path.write_text(native)
            spec = importlib.util.spec_from_file_location("fq_pf00_native", path)
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot load fixed native workload")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            from flagquantum.compiler._construction.frontend_python import (
                _QuantumFunction,
            )

            if not isinstance(module.main, _QuantumFunction):
                raise TypeError("generated source did not produce a typed qfunc")
            self.function = module.main
        self.source_sha256 = hashlib.sha256(self.source.encode()).hexdigest()
        self.request = self._make_request()
        self.last_e0_ns = 0

    def _make_request(self) -> partial[Program]:
        from .arithmetic_baseline import BASIS, PLANNING_BYTES

        w = self.workload
        limits: Limits = {
            "max_qubits": 1 if w.refused else 256,
            "basis": None if w.target == "logical" else BASIS,
            "topology": "line" if w.target == "line" else None,
            "memory_limit_bytes": PLANNING_BYTES,
        }
        function = self.function
        if isinstance(function, str):
            return partial(
                self.synth.synthesize, function, source_format="qmod", **limits
            )
        if w.frontend == "scalar":
            return partial(
                self.synth.synthesize,
                function,
                inputs={
                    name: self.synth.UInt(width)
                    for name, width in zip(
                        self.case.expression.arguments, self.case.widths, strict=True
                    )
                },
                input_state=self.synth.InputState.uniform(),
                functions=self.helpers,
                **limits,
            )
        from flagquantum.compiler._construction.frontend_python import _QuantumFunction

        if not isinstance(function, _QuantumFunction):
            raise TypeError("native entry requires typed qfunc")
        return partial(self.synth.synthesize, function, **limits)

    def compile(self) -> Program:
        started = time.perf_counter_ns()
        try:
            return self.request()
        finally:
            self.last_e0_ns = time.perf_counter_ns() - started

    def verify(self, program: Program) -> dict[str, object]:
        from .arithmetic_baseline import MAX_EXECUTION_QUBITS, costs, workspace_of
        from .oracle import expected_amplitudes, output_width
        from .verification import basis_checks

        if len(program.registers["result"]) != output_width(self.case):
            raise AssertionError("PF-00 full-domain output width mismatch")
        if self.workload.frontend == "scalar":
            count, exhaustive = basis_checks(self.case, program)
        else:
            from dataclasses import replace
            from itertools import product

            from .oracle import evaluate
            from .verification import classical_apply, encode

            arithmetic = program.arithmetic
            input_wires = tuple(
                q
                for name in self.case.expression.arguments
                for q in program.registers[name]
            )
            prefix = arithmetic.instructions[: len(input_wires)]
            if (
                len(prefix) != len(input_wires)
                or {gate.wires for gate in prefix} != {(q,) for q in input_wires}
                or any(
                    g.name != "h" or g.params or g.matrix is not None or g.metadata
                    for g in prefix
                )
            ):
                raise AssertionError(
                    "native preparation is not the declared input-only H prefix"
                )
            block = replace(
                arithmetic,
                instructions=arithmetic.instructions[len(input_wires) :],
                shape=(),
            )
            registers = tuple(
                program.registers[name]
                for name in (*self.case.expression.arguments, "result")
            )
            targets = (
                range(1 << len(registers[-1]))
                if self.workload.case == "energy-2x2"
                else (0, 1, (1 << len(registers[-1])) - 1)
            )
            count = 0
            import random

            exhaustive = sum(self.case.widths) <= 6
            rng = random.Random(20261102)
            maxima = tuple((1 << width) - 1 for width in self.case.widths)
            values_to_check = (
                list(product(*(range(limit + 1) for limit in maxima)))
                if exhaustive
                else [tuple(0 for _ in maxima), maxima]
                + [
                    tuple(rng.randrange(limit + 1) for limit in maxima)
                    for _ in range(16)
                ]
            )
            for values in values_to_check:
                for target in targets:
                    initial = encode((*values, target), registers)
                    expected = encode(
                        (*values, target ^ evaluate(self.case.expression.name, values)),
                        registers,
                    )
                    if classical_apply(block, initial) != expected:
                        raise AssertionError("native basis/XOR/cleanup mismatch")
                    count += 1
        # Basis interpretation uses the unprepared logical arithmetic, even for routed targets.
        e1: dict[str, object] = {
            "status": "not_run_state_budget",
            "limit_qubits": MAX_EXECUTION_QUBITS,
        }
        if (
            sum(self.case.widths) <= 6
            and program.circuit.n_wires <= MAX_EXECUTION_QUBITS
        ):
            start = time.perf_counter_ns()
            result = self.fq.run(program)
            run_ns = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            distribution = dict(
                program.probabilities(
                    result, registers=(*self.case.expression.arguments, "result")
                )
            )
            read_ns = time.perf_counter_ns() - start
            from itertools import product

            from .oracle import evaluate

            expected_distribution = {
                (*values, evaluate(self.case.expression.name, values)): 1
                / (1 << sum(self.case.widths))
                for values in product(
                    *(range(1 << width) for width in self.case.widths)
                )
            }
            if distribution.keys() != expected_distribution.keys() or any(
                abs(distribution[key] - value) > 1e-12
                for key, value in expected_distribution.items()
            ):
                raise AssertionError(
                    "PF-00 named result differs from independent mathematics"
                )
            state = result.statevector().reshape(-1)
            selected = program.selection.selected
            if selected is None or selected.target is None:
                raise AssertionError("missing verified target layout")
            physical_registers = {
                name: tuple(selected.target.layout.physical_of(q) for q in ports)
                for name, ports in program.registers.items()
            }
            amplitudes = expected_amplitudes(
                self.case, physical_registers, program.circuit.n_wires
            )
            residual = state.clone()
            for index, amplitude in amplitudes.items():
                residual[index] -= amplitude
            error = float(residual.abs().max())
            if error > 1e-12:
                raise AssertionError(
                    "PF-00 complete amplitude or clean-workspace mismatch"
                )
            e1 = {
                "status": "verified",
                "run_ns": run_ns,
                "named_read_ns": read_ns,
                "maximum_amplitude_error": error,
                "outcomes": len(distribution),
            }
        return {
            "basis_checks": count,
            "basis_domain_exhaustive": exhaustive,
            "nonzero_targets": True,
            "e1": e1,
            "resources": asdict(costs(program.circuit, workspace_of(program))),
        }

    def close(self) -> None:
        self.folder.cleanup()
