# SPDX-License-Identifier: Apache-2.0
# Frozen source assets copyright 2026 Google LLC.

"""Copy frozen raw circuit assets and observe unchanged source execution."""

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from random import Random

SOURCE_BLOBS = {
    "cmul16_mod_21.kmx": "e4749327b60da053ad213bb175e158bf312e92b5",
    "iadd8_qc_2clean_6dirty.kmx": "4688c9883f30864e6414ecef23bf4bae1982bfbf",
    "inc3.kmx": "d4a8f8544c4f27caf774a7a3504e17a3b85185af",
    "inc8.kmx": "0604339b577f47d5cc63de0a89088863e1e8a935",
    "mul16_mod_21.kmx": "7679d01ef4e443806f530ba2c7ac526ccae0baa8",
    "qft_measurement.kmx": "b00e1636b21e0a2c35e78cb62c60acbaec26852a",
}


def regenerate(source_root: Path, asset_directory: Path, output: Path) -> None:
    import kickmix as km

    if (
        km.__version__ != "0.1"
        or not hasattr(km, "Simulator")
        or not str(km.__file__).endswith(".so")
    ):
        raise ValueError("The locked compiled Kickmix Simulator is required")
    records = []
    asset_directory.mkdir(parents=True, exist_ok=True)
    for name, blob in SOURCE_BLOBS.items():
        raw = (source_root / "testdata" / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != blob:
            raise ValueError(f"Frozen circuit asset mismatch for {name}")
        circuit = km.Circuit(raw.decode())
        canonical = str(circuit)
        packed = BytesIO()
        circuit.to_file(packed, format="kmb")
        width = circuit.num_qubits
        bits = circuit.num_bits
        cases = []
        lines = [
            line
            for line in canonical.splitlines()
            if not line.startswith(("REGISTER ", "APPEND_TO_REGISTER "))
        ]
        for seed in (0, 41):
            rng = Random(127 + seed)
            qubits = [rng.getrandbits(64) for _ in range(width)]
            initial_bits = [rng.getrandbits(64) for _ in range(bits)]
            if name in ("cmul16_mod_21.kmx", "inc3.kmx", "qft_measurement.kmx"):
                qubits = [
                    sum(((lane >> q) & 1) << lane for lane in range(64))
                    for q in range(width)
                ]
            if name == "inc8.kmx":
                for q in range(12, 17):
                    qubits[q] = 0
            if name == "iadd8_qc_2clean_6dirty.kmx":
                qubits[8] = qubits[9] = 0
                for bit in range(8, bits):
                    initial_bits[bit] = 0
            if name == "mul16_mod_21.kmx":
                for q in range(8):
                    qubits[q] = 0
                initial_bits = [0] * bits
            original = km.Simulator(batch_size=64, seed=seed)
            streamed = km.Simulator(batch_size=64, seed=seed)
            for sim in (original, streamed):
                for q, value in enumerate(qubits):
                    sim.write_across_shots(km.q(q), value)
                for bit, value in enumerate(initial_bits):
                    sim.write_across_shots(km.b(bit), value)
            original.do(circuit)
            draws = []
            group: list[str] = []
            for line in lines:
                if group or line.startswith("PUSH_CONDITION"):
                    group.append(line)
                    if line != "POP_CONDITION":
                        continue
                    streamed.do(km.Circuit("\n".join(group)))
                    group.clear()
                    continue
                code, *fields = line.split()
                before = (
                    [Fraction(streamed.read_shot_phase(lane)) for lane in range(64)]
                    if code == "R"
                    else []
                )
                streamed.do(km.Circuit(line))
                if code == "HMR":
                    result = int(fields[1][1:])
                    draws.append(streamed.read_across_shots(km.b(result), out=int))
                elif code == "R":
                    draws.append(
                        sum(
                            int(
                                (
                                    Fraction(streamed.read_shot_phase(lane))
                                    - before[lane]
                                )
                                % 2
                            )
                            << lane
                            for lane in range(64)
                        )
                    )
            assert not group
            final_q = [
                original.read_across_shots(km.q(q), out=int) for q in range(width)
            ]
            final_b = [
                original.read_across_shots(km.b(bit), out=int) for bit in range(bits)
            ]
            phases = [
                int(Fraction(original.read_shot_phase(lane)) * (1 << 127))
                for lane in range(64)
            ]
            assert final_q == [
                streamed.read_across_shots(km.q(q), out=int) for q in range(width)
            ]
            assert final_b == [
                streamed.read_across_shots(km.b(bit), out=int) for bit in range(bits)
            ]
            assert phases == [
                int(Fraction(streamed.read_shot_phase(lane)) * (1 << 127))
                for lane in range(64)
            ]
            cases.append(
                {
                    "seed": seed,
                    "qubits": qubits,
                    "bits": initial_bits,
                    "draws": draws,
                    "final_qubits": final_q,
                    "final_bits": final_b,
                    "phases": phases,
                }
            )
        records.append(
            {
                "name": name,
                "blob": blob,
                "canonical": canonical,
                "kmb_hex": packed.getvalue().hex(),
                "qubits": width,
                "bits": bits,
                "counts": dict(Counter(line.split()[0] for line in lines)),
                "cases": cases,
            }
        )
        (asset_directory / name).write_bytes(raw)
    output.write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.source_root, args.asset_directory, args.output)
    print("Recorded all six frozen circuit assets and exact source branch observations")


if __name__ == "__main__":
    main()
