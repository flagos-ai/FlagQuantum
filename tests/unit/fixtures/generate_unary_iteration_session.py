"""Capture locked source context lifecycle and complete physical streams."""

import hashlib
import json
from pathlib import Path

import kickmix as km


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert (
        hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest()
        == "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    cases = []
    for width in (0, 1, 2, 3):
        for mode in ("true", "false", "quantum", "classical"):
            for action in (
                "sweep",
                "range",
                "seek",
                "break",
                "exception",
                "twopass",
                "empty",
            ):
                builder = km.CircuitBuilder()
                address = builder.create_quantum_register(width, name="address")
                control = builder.create_quantum_register(1, name="control")
                target = builder.create_quantum_register(1, name="target")
                guard = (
                    builder.create_classical_register(1, name="guard")
                    if mode == "classical"
                    else None
                )
                enable = {
                    "true": True,
                    "false": False,
                    "quantum": control[0],
                    "classical": None if guard is None else guard[0],
                }[mode]
                values = None
                if action == "range":
                    values = range(-2, (1 << width) + 2)
                elif action == "empty":
                    values = range(0)
                cursor = builder.unary_iteration(address, values, control=enable)
                visits = []
                positions = []
                allocations = []
                for _ in range(2 if action == "twopass" else 1):
                    try:
                        with cursor:
                            positions.append(cursor.cur_address_value)
                            allocations.append(builder.num_allocated_qubits)
                            if action == "seek":
                                for value in (
                                    -(10**30),
                                    0,
                                    1,
                                    (1 << width) - 1,
                                    10**30,
                                    0,
                                ):
                                    cursor.move_to(value)
                                    positions.append(cursor.cur_address_value)
                                    builder.cx(cursor.match_qubit, target[0])
                            else:
                                for value, match in cursor:
                                    visits.append(value)
                                    if value % 3 == 0:
                                        builder.cx(match, target[0])
                                    if action == "break":
                                        break
                                    if action == "exception":
                                        raise RuntimeError("body stop")
                    except RuntimeError as error:
                        assert str(error) == "body stop"
                    allocations.append(builder.num_allocated_qubits)
                cursor.close()
                text = str(builder.finish_circuit())
                records = [
                    line
                    for line in text.splitlines()
                    if not line.startswith(("REGISTER ", "APPEND_TO_REGISTER "))
                ]
                cases.append(
                    {
                        "width": width,
                        "mode": mode,
                        "action": action,
                        "visits": visits,
                        "positions": positions,
                        "allocations": allocations,
                        "records": records,
                    }
                )
    builder = km.CircuitBuilder()
    outer = builder.create_quantum_register(2, name="outer")
    inner = builder.create_quantum_register(2, name="inner")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(outer, range(3)) as first:
        for left, match in first:
            with builder.unary_iteration(inner, range(3), control=match) as second:
                for right, enabled in second:
                    if (left + right) % 2:
                        builder.cx(enabled, target[0])
    nested = [
        line
        for line in str(builder.finish_circuit()).splitlines()
        if not line.startswith(("REGISTER ", "APPEND_TO_REGISTER "))
    ]
    output = Path(__file__).resolve().parents[1] / "data/unary_iteration_session.json"
    output.write_text(json.dumps({"cases": cases, "nested": nested}, indent=2) + "\n")


if __name__ == "__main__":
    main()
