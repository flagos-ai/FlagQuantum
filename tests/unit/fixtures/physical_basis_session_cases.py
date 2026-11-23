# SPDX-License-Identifier: Apache-2.0
# Source observations target locked Kickmix 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Exercise the Python binding boundary through source and native factories."""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest

pytest.importorskip("numpy")
import numpy as np


def observe_sessions(
    simulator: Callable[..., Any],
    circuit: Callable[[str], Any],
    quantum: Callable[[int], Any],
    classical: Callable[[int], Any],
    register: Callable[[int], Any],
) -> dict[str, object]:
    """Normalize external binding observations immediately to JSON values."""
    result: dict[str, object] = {}
    seed = (1 << 255) + 123456789
    for shots in (0, 1, 3, 255, 256, 257, 513):
        sim = simulator(shots, debug_print=False, seed=seed, counting_operations=True)
        quiet = simulator(shots, debug_print=False, seed=seed)
        text = "R q0\nR q1\nCX q0 q2\nCCX q0 q1 q3\nHMR q3 b0\nZ q2\nZ_POW q0 0.125\nNEG if b0"
        for repetition in range(3):
            sim.do(circuit(text))
            quiet.do(circuit(text))
            masks = sim.read_across_shots(
                [quantum(k) for k in range(4)] + [classical(0)], out=int
            )
            phases = [str(sim.read_shot_phase(k)) for k in range(shots)]
            assert masks == quiet.read_across_shots(
                [quantum(k) for k in range(4)] + [classical(0)], out=int
            )
            assert phases == [str(quiet.read_shot_phase(k)) for k in range(shots)]
            result[f"seeded:{shots}:{repetition}"] = [
                masks,
                phases,
                dict(sim.operation_statistics()),
            ]
            if shots:
                result[f"lane:{shots}:{repetition}"] = [
                    dict(sim.operation_statistics(k)) for k in (0, shots - 1)
                ]
        sim.clear_for_shot()
        result[f"clear:{shots}"] = [
            sim.read_across_shots(quantum(0), out=int),
            sim.read_phase_flipped_across_shots(out=int),
            dict(sim.operation_statistics()),
        ]
        sim.clear_operation_statistics()
        result[f"countclear:{shots}"] = dict(sim.operation_statistics())

    sim = simulator(257, debug_print=False, seed=seed, counting_operations=True)
    aliases = circuit(
        'APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b0 r0\nAPPEND_TO_REGISTER q2 r0\nREGISTER r0 "mixed"'
    )
    sim.use_same_registers_as(aliases)
    sim.write_across_shots("mixed", [1, 1 << 256, 3])
    result["aliases"] = [
        sim.read_across_shots(key, out=int)
        for key in ("mixed", register(0), [quantum(0), classical(0), quantum(2)])
    ]
    result["shapes"] = [
        list(sim.read_across_shots(key).shape)
        for key in (quantum(0), "mixed", list[object](), True)
    ]
    result["within"] = [sim.read_within_shot("mixed", k, out=int) for k in (0, 1, 256)]
    result["withinshapes"] = [
        list(sim.read_within_shot(key, 0).shape) for key in (quantum(0), "mixed", True)
    ]
    sim.write_across_shots(classical(0), (1 << 256) | 1)
    sim.do(circuit("PUSH_CONDITION if b0\nX q1"))
    sim.do(circuit("X q1"))
    result["retained"] = [
        sim.read_across_shots(quantum(1), out=int),
        dict(sim.operation_statistics()),
    ]
    sim.do(circuit("POP_CONDITION\nX q1"))
    result["popped"] = [
        sim.read_across_shots(quantum(1), out=int),
        dict(sim.operation_statistics()),
    ]

    sim = simulator(5, debug_print=False, seed=0)
    items = [quantum(0), classical(0), quantum(2)]
    values = np.arange(30).reshape(3, 10) % 3 == 0
    sim.write_across_shots(items, values[::-1, ::-2])
    storage = np.zeros((6, 10), dtype=np.bool_)
    out = storage[::2, ::-2]
    assert sim.read_across_shots(items, out=out) is out
    result["stridedacross"] = storage.tolist()
    line = np.zeros(10, dtype=np.bool_)
    sim.read_across_shots(quantum(0), out=line[::-2])
    result["stridedsingle"] = line.tolist()
    sim.write_within_shot(items, 3, np.array([True, False, True])[::-1])
    within = np.zeros(6, dtype=np.bool_)
    sim.read_within_shot(items, 3, out=within[::-2])
    result["stridedwithin"] = within.tolist()
    sim.write_across_shots([quantum(0), quantum(0)], [3, 5])
    sim.write_within_shot([quantum(0), quantum(0)], 4, 1)

    class IntegerSequence:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, position: int) -> int:
            return (3, 5)[position]

    sim.write_across_shots([quantum(0), classical(0)], IntegerSequence())
    result["customsequence"] = sim.read_across_shots(
        [quantum(0), classical(0)], out=int
    )
    result["duplicates"] = sim.read_across_shots(quantum(0), out=int)
    scalar = np.zeros((), dtype=np.bool_)
    assert sim.read_within_shot(quantum(0), 0, out=scalar) is scalar
    result["scalar"] = bool(scalar)
    overlap = np.zeros(1, dtype=np.bool_)
    overlapping = np.lib.stride_tricks.as_strided(overlap, shape=(3, 5), strides=(0, 0))
    sim.read_across_shots(items, out=overlapping)
    result["overlap"] = overlap.tolist()
    for index, phase in enumerate(
        (0, 1, Fraction(1, 3), -Fraction(1, 8), Fraction(17, 8))
    ):
        sim.write_shot_phase(index, phase)
    result["phases"] = [str(sim.read_shot_phase(k)) for k in range(5)]
    sim.write_shot_phase(2, 0)
    sim.write_shot_phase(3, 1)
    sim.write_shot_phase(4, 0)
    phase_out = np.zeros(10, dtype=np.bool_)
    sim.read_phase_flipped_across_shots(out=phase_out[::-2])
    result["phaseout"] = [
        sim.read_phase_flipped_across_shots(out=int),
        phase_out.tolist(),
    ]
    sim.write_shot_phase(2, Fraction(1, 4))
    partial = np.ones(5, dtype=np.bool_)
    try:
        sim.read_phase_flipped_across_shots(out=partial)
    except ValueError:
        result["partialphase"] = partial.tolist()
    else:
        raise AssertionError("Nonbinary phase accepted")

    sim = simulator(3, debug_print=False, seed=0)
    try:
        sim.write_across_shots([quantum(0), quantum(8)], [5, 8])
    except ValueError:
        result["partialwrite"] = sim.read_across_shots(
            [quantum(0), quantum(8)], out=int
        )
    else:
        raise AssertionError("Out-of-range write accepted")
    result["emptybounds"] = sim.read_within_shot([], 99, out=int)
    sim.write_within_shot([], 99, 0)
    for width in (9, 257, 513, 1000, 2048):
        indices = [quantum(k) for k in range(width)]
        value = (1 << (width - 1)) | 3
        sim.write_within_shot(indices, 2, value)
        result[f"wide:{width}"] = sim.read_within_shot(indices, 2, out=int)

    calls: list[object] = []

    def debug(target: Any, values: Any, predicate: Any, active: Any) -> None:
        def identifier(value: Any) -> str | None:
            if value is None:
                return None
            return (
                str(value)
                if not hasattr(value, "diagnostic_text")
                else value.diagnostic_text
            )

        assert values.dtype == np.bool_ and active.dtype == np.bool_
        calls.append(
            [
                identifier(target),
                values.tolist(),
                identifier(predicate),
                active.tolist(),
            ]
        )

    sim = simulator(300, debug_print=debug, seed=0, counting_operations=True)
    sim.write_across_shots(quantum(0), 5 | (3 << 256))
    sim.write_across_shots(classical(0), 3 | (1 << 256))
    sim.do(
        circuit(
            "DEBUG_PRINT q0\nDEBUG_PRINT b0 if b0\nPUSH_CONDITION if b0\nDEBUG_PRINT if b0\nPOP_CONDITION"
        )
    )
    result["debug"] = calls
    sim.debug_print = False
    sim.do(circuit("DEBUG_PRINT q0"))
    result["debugcount"] = dict(sim.operation_statistics())
    return result
