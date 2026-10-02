"""The OpenQASM gate spellings that FlagQuantum emission and import share.

Two directions of the same translation meet here:

* :data:`EMITTED_GATES` is the one-statement spelling
  ``flagquantum.compiler.openqasm.emit_openqasm`` writes for each opcode the
  target dialect defines directly;
* :data:`FIXED_GATES` and :data:`PARAMETERIZED_GATES` are the spellings
  ``flagquantum.compiler.openqasm_import.import_openqasm`` accepts, and
  ``flagquantum.compiler.target_conformance`` reads the same two tables, so the
  strict round-trip check recognises exactly the spellings import recognises.

Five unitary opcodes are absent from :data:`EMITTED_GATES` because the emitter
writes each of them as a short sequence of these gates rather than as one
statement (``sx``, ``sxdg``, ``rxx``, ``ryy``, ``rzz``), and two more appear in
it under a version-dependent alias (``phase`` and ``cphase``, written as ``u1``
or ``p`` and as ``cu1`` or ``cp``). Those decompositions belong to the emitter;
this module owns only the spellings that are one statement in both directions.
"""

from __future__ import annotations

EMITTED_GATES: dict[str, str] = {
    "i": "id",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "u1": "u1",
    "u2": "u2",
    "u3": "u3",
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
    "ccx": "ccx",
    "cswap": "cswap",
}

FIXED_GATES: dict[str, str] = {
    "id": "i",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "sx": "sx",
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "ccx": "ccx",
    "cswap": "cswap",
}

PARAMETERIZED_GATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "rx": ("rx", ("theta",)),
    "ry": ("ry", ("theta",)),
    "rz": ("rz", ("theta",)),
    "u1": ("u1", ("theta",)),
    "u2": ("u2", ("phi", "lbd")),
    "u3": ("u3", ("theta", "phi", "lbd")),
    "p": ("phase", ("theta",)),
    "cu1": ("cphase", ("theta",)),
    "cp": ("cphase", ("theta",)),
    "crx": ("crx", ("theta",)),
    "cry": ("cry", ("theta",)),
    "crz": ("crz", ("theta",)),
}

__all__ = (
    "EMITTED_GATES",
    "FIXED_GATES",
    "PARAMETERIZED_GATES",
)
