# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Main scenarios adapted from locked Kickmix py/main_test.py.

"""Capture locked file bytes, resource projections and stream edge cases."""

import hashlib
import io
import json
from base64 import b64encode
from pathlib import Path

import kickmix as km

TEXTS = (
    "",
    "X q0\nCX q0 q9",
    "APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER q1 r0\n"
    'APPEND_TO_REGISTER b0 r1\nREGISTER r0 "quantum_reg"\n'
    'REGISTER r1 "classical_reg"\nCCX q0 q1 q2\nCX q0 q1\nX q0\n'
    "HMR q2 b0\nZ_POW q0 0.25 if b0",
    "CCX q0 q1 q2\nCCZ q0 q1 q2 if b0\nZ q0\n"
    "Z_POW q0 0.0\nZ_POW q0 0.5\nZ_POW q0 1.0\nZ_POW q0 1.5\n"
    "Z_POW q0 -0.5 if b0\nZ_POW q0 0.25\nZ_POW q0 0.75\n"
    "Z_POW q0 -0.25 if b0\nZ_POW q0 0.125\nZ_POW q0 0.1 if b1",
    "X q0\nX q0 if b0\nCX q0 q1\nCX q0 q1 if b0\n"
    "PUSH_CONDITION if b0\nCCX q0 q1 q2\nCCZ q0 q1 q2 if b1\n"
    "CSWAP q0 q1 q2\nCSWAP q0 q1 q2 if b1\nZ_POW q0 0.25\n"
    'Z_POW q0 0.125 if b1\nINFO_TAG "hello"\nPOP_CONDITION\n'
    "DEBUG_PRINT\nDEBUG_PRINT q0\nDEBUG_PRINT b0 if b1",
)


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    cases = []
    for text in TEXTS:
        circuit = km.Circuit(text)
        payloads = {}
        for format in ("kmx", "kmb"):
            stream = io.BytesIO()
            circuit.to_file(stream, format=format)
            payloads[format] = b64encode(stream.getvalue()).decode()
        stats = circuit.operation_statistics()
        cases.append(
            {
                "input": text,
                "canonical": str(circuit),
                "payloads": payloads,
                "counts": dict(stats),
                "instructions": stats.num_instructions,
                "toffoli": stats.num_toffoli_like,
                "t": stats.num_t_like,
                "levels": stats.z_pow_clifford_hierarchy_counts,
            }
        )
    probes = []
    for source in (False, True):
        for text in ('INFO_TAG "x#y"', 'REGISTER r0 "x#y"', "X q0\r\n"):
            input_stream = io.StringIO(text) if source else io.BytesIO(text.encode())
            try:
                value = str(km.Circuit.from_file(input_stream))
                probes.append({"text_stream": source, "text": text, "value": value})
            except ValueError:
                probes.append(
                    {"text_stream": source, "text": text, "error": "ValueError"}
                )
    output = Path(__file__).resolve().parents[1] / "data/physical_circuit_files.json"
    output.write_text(json.dumps({"cases": cases, "probes": probes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
