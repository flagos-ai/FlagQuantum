# SPDX-License-Identifier: Apache-2.0
# Source observations from locked Kickmix circuit_builder_util.pybind.cc.

"""Observe table conversion through its original initialization/erasure callers."""

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from tests.unit.fixtures.generate_lookup_binding import Case, source_records


def conversion_table(label: str, events: list[str]) -> Sequence[object]:
    class Converted:
        def __init__(self, name: str, value: int | None) -> None:
            self.name, self.value = name, value

        def __int__(self) -> int:
            events.append(self.name)
            if self.value is None:
                raise RuntimeError(self.name)
            return self.value

    class BitLengthOverride(int):
        def bit_length(self) -> int:
            events.append("bit_length")
            return 100

    if label == "early-invalid":
        return [-1, Converted("later", None)]
    if label == "later-invalid":
        return [Converted("first", 0), -1, Converted("third", None)]
    if label == "conversion-error":
        return [Converted("first", 0), Converted("second", None)]
    if label == "converted-values":
        return [Converted("first", 0), Converted("second", 1)]
    if label == "subclass-boundary":
        return [BitLengthOverride(1)]
    raise ValueError(label)


def main() -> None:
    source = importlib.import_module("kickmix")
    assert source.__version__ == "0.1" and hasattr(source, "Simulator")
    assert source.__file__ is not None
    assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    rows = []
    widths = (0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)
    for n, width in [(0, w) for w in widths] + [(1, w) for w in (9, 17, 65, 129)]:
        mask = (1 << width) - 1
        for route in ("init", "joined"):
            case: Case = {
                "address_width": n,
                "target_width": width,
                "route": route,
                "table": [mask ^ (index * 0x55 & mask) for index in range(1 << n)],
                "classical_width": 0,
                "reuse_pool": False,
            }
            text = source_records(source, case)
            circuit = source.Circuit(text)
            simulator = source.Simulator(
                batch_size=3, seed=0, debug_print=False, counting_operations=True
            )
            for wire in range(n + width):
                simulator.write_across_shots(source.q(wire), (wire * 5 + 3) & 7)
            simulator.do(circuit)
            rows.append(
                {
                    "input": case,
                    "text": text,
                    "static_counts": dict(
                        Counter(line.split()[0] for line in text.splitlines())
                    ),
                    "values": [
                        simulator.read_across_shots(source.q(wire), out=int)
                        for wire in range(2 * n + width)
                    ],
                    "phases": [
                        str(simulator.read_shot_phase(lane)) for lane in range(3)
                    ],
                    "executed_counts": dict(simulator.operation_statistics()),
                }
            )
    failures = []
    for label in (
        "early-invalid",
        "later-invalid",
        "conversion-error",
        "converted-values",
        "subclass-boundary",
    ):
        events: list[str] = []
        table = conversion_table(label, events)
        builder = source.CircuitBuilder()
        address = builder.create_quantum_register(0 if len(table) == 1 else 1, "a")
        target = builder.create_quantum_register(1, "t")
        outcome: dict[str, str | None]
        try:
            builder.init_lookup(table=table, address=address, target=target)
            outcome = {"error": None, "message": None}
        except (ValueError, RuntimeError) as error:
            outcome = {"error": type(error).__name__, "message": str(error)}
        failures.append({"label": label, "events": events, **outcome})
    output = {"rows": rows, "conversion_observations": failures}
    destination = (
        Path(__file__).resolve().parents[1] / "data/lookup_table_conversion.json"
    )
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
