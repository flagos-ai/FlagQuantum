# SPDX-License-Identifier: Apache-2.0
# Source observations from locked Kickmix circuit_builder_util.pybind.cc.

"""Observe table conversion through its original initialization/erasure callers."""

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, SupportsIndex, TypedDict

from typing_extensions import NotRequired

from tests.unit.fixtures.generate_lookup_binding import Case, source_records


class ConversionObservation(TypedDict):
    label: str
    width: int
    events: list[str]
    text: str
    error: str | None
    message: str | None
    execution: NotRequired[dict[str, object]]
    joined_events: NotRequired[list[str]]
    joined_execution: NotRequired[dict[str, object]]


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

    class IntLength:
        def __int__(self) -> int:
            events.append("length:int")
            return 1

    class IndexLength:
        def __index__(self) -> int:
            events.append("length:index")
            return 1

    class EncodedOverride(int):
        def __int__(self) -> int:
            raise AssertionError("Integer subclasses retain their source identity")

        def bit_length(self) -> Any:
            events.append("bit_length")
            if label == "length-failure":
                raise RuntimeError("length failure")
            if label == "int-length":
                return IntLength()
            if label == "index-length":
                return IndexLength()
            return 0 if label == "zero-width" else 1

        def __lt__(self, other: int) -> bool:
            events.append("less_than")
            return False if label == "negative-override" else int.__lt__(self, other)

        def to_bytes(
            self,
            length: SupportsIndex = 1,
            byteorder: Literal["little", "big"] = "big",
            *,
            signed: bool = False,
        ) -> bytes:
            events.append(f"bytes:{length}:{byteorder}")
            if label == "bytes-failure":
                raise RuntimeError("bytes failure")
            return {
                "encoded-zero": b"\0",
                "encoded-one": b"\1",
                "extra-bytes": b"\1\xff",
                "padding": b"\xff",
                "underreported": b"\2",
                "negative-override": b"\1",
                "zero-width": b"",
                "int-length": b"\1",
                "index-length": b"\1",
            }[label]

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
    if label in (
        "encoded-zero",
        "encoded-one",
        "extra-bytes",
        "padding",
        "underreported",
        "negative-override",
        "zero-width",
        "bytes-failure",
        "length-failure",
        "int-length",
        "index-length",
    ):
        value = (
            -1 if label == "negative-override" else 2 if label == "underreported" else 1
        )
        return [EncodedOverride(value)]
    raise ValueError(label)


def main() -> None:
    from tests.unit.fixtures.generate_physical_circuit_binding import observe

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
    failures: list[ConversionObservation] = []
    for label in (
        "early-invalid",
        "later-invalid",
        "conversion-error",
        "converted-values",
        "subclass-boundary",
        "encoded-zero",
        "encoded-one",
        "extra-bytes",
        "padding",
        "underreported",
        "negative-override",
        "zero-width",
        "bytes-failure",
        "length-failure",
        "int-length",
        "index-length",
    ):
        events: list[str] = []
        table = conversion_table(label, events)
        builder = source.CircuitBuilder()
        address = builder.create_quantum_register(0 if len(table) == 1 else 1, "a")
        width = 0 if label == "zero-width" else 1
        target = builder.create_quantum_register(width, "t")
        outcome: dict[str, str | None]
        try:
            builder.init_lookup(table=table, address=address, target=target)
            outcome = {"error": None, "message": None}
        except (ValueError, RuntimeError) as error:
            outcome = {"error": type(error).__name__, "message": str(error)}
        text = "\n".join(
            line
            for line in str(builder.finish_circuit()).splitlines()
            if not line.startswith(("REGISTER", "APPEND_TO_REGISTER"))
        )
        observation: ConversionObservation = {
            "label": label,
            "width": width,
            "events": events.copy(),
            "text": text,
            "error": outcome["error"],
            "message": outcome["message"],
        }
        if outcome["error"] is None:
            observation["execution"] = observe(source.Circuit(text))
            events.clear()
            builder.del_lookup(table=table, address=address, target=target)
            joined = "\n".join(
                line
                for line in str(builder.finish_circuit()).splitlines()
                if not line.startswith(("REGISTER", "APPEND_TO_REGISTER"))
            )
            observation["joined_events"] = events.copy()
            observation["joined_execution"] = observe(source.Circuit(joined))
        failures.append(observation)
    output = {"rows": rows, "conversion_observations": failures}
    destination = (
        Path(__file__).resolve().parents[1] / "data/lookup_table_conversion.json"
    )
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
