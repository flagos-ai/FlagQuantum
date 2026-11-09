"""Native bit-array views preserve phase, scope, ownership and bounded resources."""

import cmath
import importlib.util
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import (
    CRX,
    CX,
    RZ,
    RZZ,
    H,
    Output,
    QArray,
    QBit,
    S,
    U,
    X,
    allocate,
    qfunc,
)

pytestmark = pytest.mark.integration


@qfunc
def asymmetric(q: Output[QArray[QBit]]) -> None:
    allocate(3, q)
    H(q[0])
    S(q[0])
    X(q[2])
    RZ(0.4, q[1])
    CRX(-0.7, q[2], q[1])
    RZZ(0.3, q[:2])
    U(1, 2, 1.5, 1.1, q[1])


def test_asymmetric_indices_slice_rotation_and_absolute_global_phase() -> None:
    cosine, sine = math.cos(0.5), math.sin(0.5)
    euler = cmath.exp(1.1j) * torch.tensor(
        [
            [cosine, -cmath.exp(1.5j) * sine],
            [cmath.exp(2j) * sine, cmath.exp(3.5j) * cosine],
        ],
        dtype=torch.complex128,
    )
    reference = fq.Circuit(3, dtype=torch.complex128)
    reference.h(2).s(2).x(0).rz(1, theta=0.4).crx(0, 1, theta=-0.7)
    reference.cx(2, 1).rz(1, theta=0.3).cx(2, 1).any(1, matrix=euler)
    program = synth.synthesize(asymmetric)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0
    assert program.registers["q"] == (0, 1, 2)


@qfunc
def couple(left: QArray[QBit], right: QArray[QBit]) -> None:
    H(left[0])
    CX(left[0], right[0])


@qfunc
def nested(q: QArray[QBit]) -> None:
    couple(q[1:2], q[2:3])


@qfunc
def sliced(q: Output[QArray[QBit]]) -> None:
    allocate(4, q)
    nested(q[:3])
    X(q[3])


def test_nested_disjoint_borrows_keep_order_and_return_ownership() -> None:
    reference = fq.Circuit(4, dtype=torch.complex128).h(2).cx(2, 1).x(0)
    torch.testing.assert_close(
        fq.run(synth.synthesize(sliced)).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@qfunc
def decorate(q: QArray[QBit]) -> None:
    for index in range(len(q)):
        S(q[index])


@qfunc
def scoped(q: Output[QArray[QBit]]) -> None:
    allocate(3, q)
    for index in range(2):
        H(q[index])
        decorate(q[index : index + 1])
    X(q[2])


def test_callee_loop_names_and_views_do_not_capture_caller_bindings() -> None:
    reference = fq.Circuit(3, dtype=torch.complex128).h(2).s(2).h(1).s(1).x(0)
    program = synth.synthesize(scoped)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@qfunc
def scalar(q: QBit) -> None:
    H(q)
    S(q)


@qfunc
def apply_scalars(q: Output[QArray[QBit]]) -> None:
    allocate(3, q)
    for index in range(2, -1, -1):
        scalar(q[index])


def test_scalar_formals_accept_individual_array_locations() -> None:
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).s(0).h(1).s(1).h(2).s(2)
    torch.testing.assert_close(
        fq.run(synth.synthesize(apply_scalars)).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def _definition(
    tmp_path: Path,
    body: str,
    prelude: str = "",
    annotation: str = "Output[QArray[QBit]]",
) -> _QuantumFunction:
    path = tmp_path / "native_views.py"
    path.write_text(
        "from flagquantum.synth import *\n"
        + prelude
        + f"\n@qfunc\ndef main(q: {annotation}) -> None:\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("native_views", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main
    assert isinstance(result, _QuantumFunction)
    return result


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(3,q)\nH(q[-1])", "program.view"),
        ("allocate(3,q)\nH(q[3])", "program.view"),
        ("allocate(3,q)\nH(q[True])", "program.loop"),
        ("allocate(3,q)\nH(q[q[0]])", "program.loop"),
        ("allocate(3,q)\nH(q[0][0])", "program.view"),
        ("allocate(3,q)\nhadamard_transform(q[:4])", "program.view"),
        ("allocate(3,q)\nhadamard_transform(q[1:1])", "program.view"),
        ("allocate(3,q)\nhadamard_transform(q[::2])", "program.view"),
        ("allocate(3,q)\nCX(q[1],q[1])", "register.gate"),
        ("allocate(3,q)\nRZZ(1,q[:3])", "program.width"),
        ("H(q[0])", "program.width"),
        ("q |= 3", "program.expression"),
        ("allocate(3,q)\nfor q in range(2):\n    H(q[0])", "program.loop"),
        (
            "allocate(3,q)\nfor i in range(2):\n    for i in range(2):\n        H(q[0])",
            "program.loop",
        ),
        ("allocate(3,q)\nfor i in range(257):\n    H(q[0])", "program.loop"),
        ("allocate(3,q)\nfor i in range(0,2,0):\n    H(q[0])", "program.loop"),
        ("allocate(3,q)\nfor i in range(q[0]):\n    H(q[0])", "program.loop"),
        ("allocate(3,q)\nfor i in range(2):\n    break", "program.loop"),
        ("allocate(3,q)\nfor i in range(0):\n    return", "program.loop"),
        (
            "allocate(3,q)\nfor i in range(2):\n    H(q[0])\nelse:\n    H(q[1])",
            "program.loop",
        ),
        ("allocate(3,q)\nfor i in range(2):\n    H(q[i])\nH(q[i])", "program.loop"),
        (
            'allocate(1,q)\nfor i in range(256):\n    for j in range(17):\n        "no effect"',
            "budget",
        ),
    ],
)
def test_view_and_loop_refusals(tmp_path: Path, body: str, diagnostic: str) -> None:
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(_definition(tmp_path, body))


@pytest.mark.parametrize(
    ("prelude", "body", "diagnostic"),
    [
        (
            "@qfunc\ndef pair(a: QArray[QBit], b: QArray[QBit]):\n    CX(a[0],b[0])\n",
            "allocate(3,q)\npair(q[:2],q[1:])",
            "program.borrow",
        ),
        (
            "@qfunc\ndef init(a: QArray[QBit]):\n    allocate(1,a)\n",
            "allocate(1,q)\ninit(q)",
            "program.borrow",
        ),
        (
            "@qfunc\ndef init(a: QArray[QBit]):\n    prepare_state([1,0],0,a)\n",
            "allocate(1,q)\ninit(q)",
            "program.borrow",
        ),
        (
            "@qfunc\ndef fresh(a: Output[QArray[QBit]]):\n    allocate(1,a)\n@qfunc\ndef escape(a: QArray[QBit]):\n    fresh(a)\n",
            "allocate(1,q)\nescape(q)",
            "program.borrow",
        ),
        (
            "@qfunc\ndef recursive(a: QArray[QBit]):\n    recursive(a)\n",
            "allocate(1,q)\nrecursive(q)",
            "program.recursion",
        ),
        (
            "@qfunc\ndef bit(a: QBit):\n    H(a)\n",
            "allocate(2,q)\nbit(q)",
            "program.width",
        ),
        (
            "@qfunc\ndef whole(a: QArray[QBit]):\n    H(a[0])\n",
            "allocate(1,q)\nfor i in range(256):\n    whole(q)\nwhole(q)",
            "budget",
        ),
        (
            "@qfunc\ndef whole(a: QArray[QBit]):\n    H(a[0])\n@qfunc\ndef readonly(a: Const[QArray[QBit]]):\n    whole(a)\n",
            "allocate(1,q)\nreadonly(q)",
            "program.borrow",
        ),
        (
            "@qfunc\ndef wrong(a: Const[QBit], b: QArray[QBit]):\n    H(b[0])\n",
            "allocate(2,q)\nwrong(q[0],q[1:])",
            "program.borrow",
        ),
    ],
)
def test_mutable_helper_ownership_and_expansion_limits(
    tmp_path: Path, prelude: str, body: str, diagnostic: str
) -> None:
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(_definition(tmp_path, body, prelude))


@pytest.mark.parametrize(
    "annotation",
    ["QArray[QBit]", "Output[QArray[QNum]]", "Output[QArray[QArray[QBit]]]"],
)
def test_array_types_and_root_roles_are_explicit(
    tmp_path: Path, annotation: str
) -> None:
    with pytest.raises(HybridVerificationError, match="program.entry|annotation"):
        synth.synthesize(_definition(tmp_path, "allocate(1,q)", annotation=annotation))


@pytest.mark.parametrize("name", ["range", "len"])
def test_shadowed_builtins_never_execute(tmp_path: Path, name: str) -> None:
    marker = tmp_path / "host-call"
    prelude = f'def {name}(*args):\n    open({str(marker)!r}, "w").write("called")\n    return 1\n'
    body = "allocate(2,q)\nfor i in range(len(q)):\n    H(q[i])"
    with pytest.raises(HybridVerificationError, match="program.loop"):
        synth.synthesize(_definition(tmp_path, body, prelude))
    assert not marker.exists()


def test_array_storage_is_not_a_host_container() -> None:
    with pytest.raises(TypeError, match="qfunc ports"):
        QArray[QBit]()


def test_final_resource_limits_remain_independent() -> None:
    for limits in (
        {"max_qubits": 2},
        {"max_operations": 1},
        {"max_depth": 1},
        {"memory_limit_bytes": 1},
    ):
        with pytest.raises(ValueError):
            synth.synthesize(
                asymmetric,
                max_qubits=limits.get("max_qubits", 256),
                max_operations=limits.get("max_operations", 100000),
                max_depth=limits.get("max_depth"),
                memory_limit_bytes=limits.get("memory_limit_bytes"),
            )


@pytest.mark.parametrize("name", ["range", "len"])
def test_none_is_also_a_shadowed_builtin(tmp_path: Path, name: str) -> None:
    with pytest.raises(HybridVerificationError, match="program.loop"):
        synth.synthesize(
            _definition(
                tmp_path,
                "allocate(2,q)\nfor i in range(len(q)):\n    H(q[i])",
                f"{name} = None\n",
            )
        )


def test_mutable_helper_adjoint_retains_absolute_coherent_state(tmp_path: Path) -> None:
    entry = _definition(
        tmp_path,
        "allocate(1,q)\nH(q[0])\nS(q[0])\nwith adjoint():\n    change(q)",
        "@qfunc\ndef change(a:QArray[QBit]):\n    H(a[0])\n    S(a[0])\n",
    )
    expected = fq.Circuit(1, dtype=torch.complex128).h(0).s(0).gate("sdg", 0).h(0)
    torch.testing.assert_close(
        fq.run(synth.synthesize(entry)).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )
