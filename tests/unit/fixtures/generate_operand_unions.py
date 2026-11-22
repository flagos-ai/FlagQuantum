"""Record complete unchanged operand union methods and compiler refusals."""

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import subprocess
import tempfile
from itertools import product
from pathlib import Path
from typing import TypedDict


class CompileRefusal(TypedDict):
    case: str
    expression: str
    category: str


class BasisObservation(TypedDict):
    classical_bits: list[int]
    quantum_words: list[int]
    phase_half_turns: list[str]
    lane_counts: list[dict[str, int]]


class GateConsumer(TypedDict):
    control: str
    target: str
    kmx: str
    basis_cases: list[BasisObservation]


class OperandUnionCorpus(TypedDict):
    pin: str
    frozen_blobs: dict[str, str]
    source_rows: list[str]
    compile_refusals: list[CompileRefusal]
    gate_consumers: list[GateConsumer]
    extension_sha256: str


def regenerate(source: Path, inventory: Path, output: Path) -> None:
    expected = {
        row["源文件"]: row["源对象"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
    }
    names = [
        "bit_or_bool.h",
        "bit_or_false.cc",
        "bit_or_false.h",
        "qubit_or_bit.h",
        "qubit_or_bit_or_bool.cc",
        "qubit_or_bit_or_bool.h",
        "qubit_or_false.cc",
        "qubit_or_false.h",
        "qubit_or_minus_state.h",
        "qubit_or_true.h",
        "qubit_or_xbit_or_xbool.cc",
        "qubit_or_xbit_or_xbool.h",
        "qubit_or_xzbit_or_xzbool.cc",
        "qubit_or_xzbit_or_xzbool.h",
        "id_tagging.h",
    ] + [
        name + extension
        for name in ("bit_id", "qubit_id", "register_id", "xbit_id", "xbool")
        for extension in (".cc", ".h")
    ]
    verified = {}
    for name in names:
        relative = "kickmix/src/kickmix/id/" + name
        raw = (source / relative).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[relative]:
            raise ValueError(f"Frozen operand union source differs for {relative}")
        verified[relative] = blob
    for relative in (
        "kickmix/src/kickmix/py/circuit/ops/cx.pybind.cc",
        "kickmix/src/kickmix/py/circuit/ops/cx.cc",
        "kickmix/src/kickmix/build/circuit_builder.cc",
        "kickmix/src/kickmix/build/circuit_builder.h",
    ):
        raw = (source / relative).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[relative]:
            raise ValueError(f"Frozen gate consumer source differs for {relative}")
        verified[relative] = blob
    failures = [
        ("false_restricted", "BitIdOrFalse(true)", "constant expression"),
        ("true_restricted", "QubitOrTrue(false)", "constant expression"),
        ("minus_restricted", "QubitOrMinusState(PLUS_KET)", "constant expression"),
        ("true_no_default", "QubitOrTrue()", "no matching constructor"),
        ("minus_no_default", "QubitOrMinusState()", "no matching constructor"),
    ] + [
        (name, name + "(uint32_t{1})", "deleted function")
        for name in (
            "BitOrBool",
            "QubitOrBit",
            "QubitOrFalse",
            "QubitOrTrue",
            "QubitOrMinusState",
            "QubitOrBitOrBool",
            "QubitOrXBitOrXBool",
            "QubitOrXZBitOrXZBool",
        )
    ]
    refusals: list[CompileRefusal] = []
    with tempfile.TemporaryDirectory(prefix="fq-operand-union-") as folder:
        temporary = Path(folder)
        binary = temporary / "oracle"
        compiler = ["clang++", "-std=c++20", "-I" + str(source / "kickmix/src")]
        subprocess.run(
            [
                *compiler,
                "-O1",
                str(Path(__file__).with_name("operand_unions.cc")),
                *[
                    str(source / path)
                    for path in verified
                    if "/id/" in path and path.endswith(".cc")
                ],
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        rows = subprocess.run(
            [str(binary)], check=True, text=True, capture_output=True, timeout=10
        ).stdout.splitlines()
        probe = temporary / "reject.cc"
        for case, expression, category in failures:
            probe.write_text(
                '#include "kickmix/id/bit_or_false.h"\n'
                '#include "kickmix/id/qubit_or_false.h"\n'
                '#include "kickmix/id/qubit_or_xzbit_or_xzbool.h"\n'
                "using namespace kickmix;\nint main(){ auto value="
                + expression
                + "; }\n"
            )
            result = subprocess.run(
                [*compiler, "-fsyntax-only", str(probe)],
                text=True,
                capture_output=True,
                timeout=15,
            )
            if not result.returncode or category not in result.stderr:
                raise RuntimeError(f"Original compiler refusal differs for {case}")
            refusals.append(
                {"case": case, "expression": expression, "category": category}
            )
    km = importlib.import_module("kickmix")
    extension = Path(inspect.getfile(km.Circuit))
    if (
        km.__version__ != "0.1"
        or not hasattr(km, "Simulator")
        or extension.suffix != ".so"
    ):
        raise RuntimeError(
            "The locked native Kickmix extension and Simulator are required"
        )
    gate_consumers: list[GateConsumer] = []
    for control_name, control in (
        ("false", False),
        ("true", True),
        ("classical", km.b(1)),
        ("classical_alias", km.b(0)),
        ("quantum", km.q(0)),
    ):
        for target_name, target in (
            ("quantum", km.q(1)),
            ("x_classical", km.xb(0)),
            ("plus", km.xbool(False)),
            ("minus", km.xbool(True)),
        ):
            builder = km.CircuitBuilder()
            builder.cx(control, target)
            circuit = builder.finish_circuit()
            basis_cases: list[BasisObservation] = []
            for first, second in product((0, 1), repeat=2):
                simulator = km.Simulator(
                    batch_size=4, counting_operations=True, debug_print=False, seed=1
                )
                simulator.write_across_shots(km.q(0), 12)
                simulator.write_across_shots(km.q(1), 10)
                simulator.write_across_shots(km.b(0), 15 * first)
                simulator.write_across_shots(km.b(1), 15 * second)
                simulator.do(circuit)
                basis_cases.append(
                    {
                        "classical_bits": [first, second],
                        "quantum_words": [
                            int(simulator.read_across_shots(km.q(k), out=int))
                            for k in range(2)
                        ],
                        "phase_half_turns": [
                            str(simulator.read_shot_phase(k)) for k in range(4)
                        ],
                        "lane_counts": [
                            dict(simulator.operation_statistics(k).items())
                            for k in range(4)
                        ],
                    }
                )
            gate_consumers.append(
                {
                    "control": control_name,
                    "target": target_name,
                    "kmx": str(circuit),
                    "basis_cases": basis_cases,
                }
            )
    corpus: OperandUnionCorpus = {
        "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
        "frozen_blobs": verified,
        "source_rows": rows,
        "compile_refusals": refusals,
        "gate_consumers": gate_consumers,
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(corpus, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.source, args.inventory, args.output)
