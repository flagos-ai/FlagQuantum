"""Execute every frozen factory cell and its owned declaration counterpart."""

import builtins
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_notebook_factories_graph_and_closed_execution_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display

    root = Path(__file__).resolve().parents[2]
    path = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/phase_estimation/qubitization_qpe.ipynb"
    )
    raw = path.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "b99e4b7c6cf28cb1224c1afc8cf455042d602b6f"
    )
    source_notebook = json.loads(raw)
    native_notebook = json.loads(
        (root / "examples/notebooks/qubitization_phase_estimation.ipynb").read_text()
    )
    assert len(source_notebook["cells"]) == 13
    assert len(native_notebook["cells"]) == 14
    assert [c["cell_type"] for c in source_notebook["cells"]] == [
        c["cell_type"] for c in native_notebook["cells"][:13]
    ]
    assert [
        c["metadata"]["frozen_source_cell"] for c in native_notebook["cells"][:13]
    ] == list(range(13))
    # The unmodified notebook imports marked upstream test helpers. Isolate its
    # original execution from the repository's strict pytest marker registry.
    script = r"""
import json
from hashlib import sha256
from pathlib import Path
import qualtran.drawing as drawing
for name in ("show_bloqs", "show_call_graph", "show_counts_sigma"):
    setattr(drawing, name, lambda *args, **kwargs: None)
ns = {}
for cell in json.loads(Path(__import__("sys").argv[1]).read_text())["cells"]:
    if cell["cell_type"] == "code":
        exec("".join(cell["source"]), ns)
def signature(registers):
    return [dict(name=r.name, kind=type(r.dtype).__name__, width=r.bitsize,
                 bound=getattr(r.dtype, "iteration_length", None),
                 frac=getattr(r.dtype, "num_frac", None),
                 shape=list(r.shape), side=r.side.name) for r in registers]
rows = {}
for name in ("hubbard_model_small", "sparse_chem", "chem_thc", "ising"):
    q = ns["qubitization_qpe_" + name]
    rows[name] = dict(signature=signature(q.signature),
                      reflection=signature(q.walk.reflect.signature),
                      alpha=float(q.walk.block_encoding.alpha), width=q.m_bits)
q = ns["qubitization_qpe_hubbard_model_small"]
counts = {}
for child, count in ns["qubitization_qpe_hubbard_model_small_sigma"].items():
    if child == q.ctrl_state_prep: key = "LuisPerinaState"
    elif child == q.walk: key = "Walk"
    elif child == q.walk.controlled(): key = "ControlledWalk(1)"
    elif child == q.qft_inv: key = "QFT†"
    else:
        assert child.control_val == 0 and child.prepare_gate == q.walk.reflect.prepare_gate
        key = "Reflection(0)"
    counts[key] = int(count)
print(json.dumps(dict(rows=rows, counts=counts, blocking=int(ns["qroam_blocking_factor"]),
    tensors={name:sha256(ns[name].tobytes()).hexdigest() for name in ("tpq", "eris", "t_l", "eta", "zeta")})))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    expected = json.loads(result.stdout)
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError("Native notebook must use owned implementations")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native = execute(native_notebook)
    assert len(displays) == 6
    for name, row in expected["rows"].items():
        operation = native["qubitization_qpe_" + name]
        assert operation.m_bits == row["width"]
        for actual, declared in (
            (operation.signature, row["signature"]),
            (operation.reflection_signature, row["reflection"]),
        ):
            ports = []
            for register in declared:
                match register["kind"]:
                    case "BQUInt":
                        dtype = bounded_unsigned_register_type(
                            register["width"], register["bound"]
                        )
                    case "QAny":
                        dtype = opaque_register_type(register["width"])
                    case "QFxp":
                        dtype = fixed_point_register_type(
                            register["width"], register["frac"]
                        )
                    case _:
                        raise AssertionError(register)
                ports.append(
                    Register(
                        register["name"],
                        dtype,
                        tuple(register["shape"]),
                        RegisterSide[register["side"]],
                    )
                )
            assert {r.name: r for r in actual} == {r.name: r for r in ports}
        with pytest.raises(ValueError, match="PREPARE"):
            width = int(operation.m_bits)
            operation.make_on(
                phase=tuple(range(width)),
                target=tuple(range(width, int(operation.signature.n_qubits()))),
            )
    for actual, name in (
        (native["hubbard_walk"].alpha, "hubbard_model_small"),
        (native["sparse_walk"].alpha, "sparse_chem"),
        (native["thc_data"].normalization, "chem_thc"),
        (native["ising"].table.normalization, "ising"),
    ):
        assert actual == pytest.approx(expected["rows"][name]["alpha"], rel=1e-15)
    from hashlib import sha256

    assert {
        name: sha256(native[name].tobytes()).hexdigest()
        for name in ("tpq", "eris", "t_l", "eta", "zeta")
    } == expected["tensors"]
    assert native["qroam_blocking_factor"] == expected["blocking"] == 32
    graph = native["qubitization_qpe_hubbard_model_small_graph"]
    assert (
        graph.leaf_counts
        == expected["counts"]
        == {
            "LuisPerinaState": 1,
            "ControlledWalk(1)": 1,
            "Reflection(0)": 26,
            "Walk": 16382,
            "QFT†": 1,
        }
    )
    np.testing.assert_array_equal(
        [native["qubitization_qpe_" + name].m_bits for name in expected["rows"]],
        [14, 21, 40, 4],
    )
    with pytest.raises(NotImplementedError, match="Physical QPE child admission"):
        native["unresolved_walk"]({}, (), (), False)
    with pytest.raises(NotImplementedError, match="Physical QPE child admission"):
        native["unresolved_controlled_walk"](0, {}, (), (), False)
