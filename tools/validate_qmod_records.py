"""Full original flat-record QAOA validation using an independent arithmetic oracle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import resource
import sys
import time
from fractions import Fraction
from operator import add, mul, pow, sub, truediv
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from tools.validate_qmod_sparse_phase import (
    CHUNK,
    independent_terms,
    maximum_error,
    reference_circuit,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
BLOBS = {
    "protein_folding": "f859094258efa802ac945e8376946f5e5c68811e",
    "patch_min_vertex_cover": "068de18b7b1a31e64458a39112399d2dac87ad97",
    "kidney_exchange_problem": "52442933bbe4d0a03bc0caa4cd057da3cb004cd4",
    "max_clique": "cbe2bdbb75c97bfdd4b474e7006f6d7b20c56ea8",
    "max_independent_set": "b214c9d2558cb8584f6dedb6d9901eaa1dadd323",
    "max_k_vertex_cover": "22f58bd00d08b2f920e25ec594e6974b9c1eaa46",
    "minimum_dominating_set": "54614c549ce5db94af2c5d100c8ec4ebbd95fb4d",
    "set_cover": "68acd2dd9ce373119987cf92b28a3ad050f083b6",
    "set_partition": "0fc084450cfaf46053c63cebcea58f923d137ef3",
    "ising_model": "b2e4dca01efeed6856554bc6f2ee6c1f968dc0c2",
}
ARITHMETIC = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.Div: truediv, ast.Pow: pow}


class RecordCost:
    """Parse only the independent oracle's flat bit layout and arithmetic tree.

    Python's AST parser supplies syntax. A closed arithmetic interpreter evaluates
    the cost on classical bits; it calls no source functions or compiler parser.
    """

    def __init__(self, source):
        record = re.search(r"qstruct\s+(\w+)\s*\{([^}]*)\}", source)
        self.fields = []
        self.width = 0
        for field, size in re.findall(r"(\w+)\s*:\s*qbit(?:\[(\d+)\])?\s*;", record[2]):
            width = int(size) if size else 1
            self.fields.append((field, self.width, width, bool(size)))
            self.width += width
        opening = re.search(r"\bphase\s*\(", source).end()
        nesting = 0
        end = opening
        for end in range(opening, len(source)):
            character = source[end]
            if character in "([":
                nesting += 1
            elif character in ")]":
                nesting -= 1
            elif character == "," and nesting == 0:
                break
        self.expression = ast.parse(source[opening:end], mode="eval").body
        self.count = int(re.search(r"params:\s*real\[(\d+)\]", source)[1])
        self.degree = min(self.width, self._degree(self.expression))

    def _degree(self, node):
        if isinstance(node, ast.Constant):
            return 0
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            return 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self._degree(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self._degree(node.left), self._degree(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(left, right)
            if isinstance(node.op, ast.Mult):
                return left + right
            if isinstance(node.op, ast.Div) and right == 0:
                return left
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                return left * node.right.value
        raise ValueError("oracle admits only real arithmetic over declared fields")

    def __call__(self, bits, number=Fraction):
        fields = {
            name: bits[offset : offset + width] if array else bits[offset]
            for name, offset, width, array in self.fields
        }

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return number(str(node.value))
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "v"
            ):
                return fields[node.attr]
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                return evaluate(node.value)[node.slice.value]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -evaluate(node.operand)
            if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
                return ARITHMETIC[type(node.op)](
                    evaluate(node.left), evaluate(node.right)
                )
            raise ValueError("non-arithmetic source is outside the oracle")

        return evaluate(self.expression)


def compare_classical_phase(source, oracle):
    params = [0.0] * oracle.count
    params[0] = 0.19
    program = synth.synthesize(
        source, classical_inputs={"params": params}, memory_limit_bytes=512 * 1024**2
    )
    state = fq.run(program).statevector().reshape(-1)
    size = 1 << oracle.width
    error = 0.0
    for start in range(0, size, CHUNK):
        stop = min(start + CHUNK, size)
        values = torch.arange(start, stop, dtype=torch.int64)
        bits = [((values >> bit) & 1).to(torch.float64) for bit in range(oracle.width)]
        expected = torch.exp(1j * params[0] * oracle(bits, float)) / size**0.5
        error = max(error, float((state[start:stop] - expected).abs().max()))
    if error > 1e-12:
        raise AssertionError(f"full-domain cost phase mismatch {error}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(BLOBS), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-phase", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or not (-0.1 <= args.scale <= 0.1) or not args.scale:
        parser.error("use positive threads and nonzero scale within ±0.1")
    torch.set_num_threads(args.threads)
    source = FIXTURES / f"{args.case}.qmod"
    raw = source.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != BLOBS[args.case]:
        raise ValueError("original source blob mismatch")
    oracle = RecordCost(raw.decode())
    params = [
        args.scale * (index + 1) * (-1 if index % 2 else 1)
        for index in range(oracle.count)
    ]
    started = time.perf_counter()
    checkpoint = started
    program = synth.synthesize(
        source, classical_inputs={"params": params}, memory_limit_bytes=512 * 1024**2
    )
    times = {"synthesis_seconds": time.perf_counter() - checkpoint}
    print(f"{args.case} scale={args.scale} synthesized", flush=True)
    checkpoint = time.perf_counter()
    actual = fq.run(program).statevector().reshape(-1)
    times["execution_seconds"] = time.perf_counter() - checkpoint
    checkpoint = time.perf_counter()
    terms = independent_terms(oracle, oracle.degree, oracle.width)
    midpoint = oracle.count // 2
    interleaved = [
        value
        for pair in zip(params[:midpoint], params[midpoint:], strict=True)
        for value in pair
    ]
    reference = reference_circuit(interleaved, terms, oracle.width)
    times["reference_construction_seconds"] = time.perf_counter() - checkpoint
    checkpoint = time.perf_counter()
    expected = fq.run(reference).statevector().reshape(-1)
    times["reference_execution_seconds"] = time.perf_counter() - checkpoint
    error = maximum_error(actual, expected)
    norm = float(actual.abs().square().sum())
    if abs(norm - 1) > 1e-9:
        raise AssertionError(f"state norm {norm}")
    del actual, expected, reference
    print(f"{args.case} scale={args.scale} full state verified", flush=True)
    phase_error = None
    if args.check_phase:
        checkpoint = time.perf_counter()
        phase_error = compare_classical_phase(source, oracle)
        times["classical_phase_seconds"] = time.perf_counter() - checkpoint
    resources = program.report["resources"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "case": args.case,
        "source_blob": blob,
        "parameters": params,
        "record_fields": oracle.fields,
        "named_registers": dict(program.registers),
        "threads": args.threads,
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": program.report["workspace_qubits"],
        "oracle_terms": len(terms),
        "oracle_degree": oracle.degree,
        "max_abs_state_error": error,
        "max_abs_classical_phase_error": phase_error,
        "state_norm": norm,
        "timings": times,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "validation": "passed",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
