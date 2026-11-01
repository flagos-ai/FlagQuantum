"""Native comparison normalization and additive upstream provenance."""

import hashlib
import json
import operator
from pathlib import Path

import pytest

from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem._qmod.adapter import (
    SUPPORTED_SOURCE_VERSION,
    QmodImport,
    parse_qmod,
)

pytestmark = pytest.mark.unit
RELATIONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
CORPUS = Path(__file__).resolve().parents[2] / "fixtures/classiq_qmod"
SOURCE_SHA = "1290eacc5d2d6ab66e7ea8f89e0110d588c949666dfba54cddc589acf7257528"
BLOB_SHA = "76c6b3ce28f48901a4982fc3d4251deeb1777d88"
LICENSE_SHA = "241f68c9f35357f97e8a2a195e447d4f80098c140f0b3b3c770a11eed96f58e0"


def parse(source: str) -> QmodImport:
    return parse_qmod(
        source, filename="compare.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("left,right", [(0, 0), (0, 1), (2, 1)])
def test_literal_comparisons_fold_to_exact_classical_bits(
    relation: str, left: int, right: int
) -> None:
    imported = parse(
        f"qfunc main(output flag:qnum) {{ flag={left}{relation}{right}; }}"
    )
    assert not imported.functions
    initializer = next(
        op
        for op in imported.program.body.blocks[0].operations
        if op.name == "qreg.initialize"
    )
    assert initializer.attributes["value"] == int(RELATIONS[relation](left, right))
    assert imported.inferred_widths == {"flag": 1}


@pytest.mark.parametrize("relation", RELATIONS)
def test_quantum_comparisons_reuse_owned_kernels_and_one_bit_width(
    relation: str,
) -> None:
    imported = parse(
        f"qfunc compare(const x:qnum<2>,const y:qnum<3>,output flag:qbit) {{ flag=x{relation}y; }} qfunc main(output x:qnum<2>,output y:qnum<3>,output flag:qbit) {{ x=0; y=0; compare(x,y,flag); }}"
    )
    graph = next(f for f in imported.functions if ".expr" in f.name)
    block = graph.body.blocks[0]
    assert block.operations[-1].operands[0].type.parameters == (1,)
    assert {op.name for op in block.operations} <= {
        "qint.equal",
        "qint.less",
        "arith.constant",
        "qint.return",
    }
    comparison = block.operations[0]
    expected = block.arguments[::-1] if relation in {">", "<="} else block.arguments
    assert comparison.operands == expected
    assert len(
        [op for op in block.operations if op.name in {"qint.equal", "qint.less"}]
    ) == (2 if relation in {"!=", "<=", ">="} else 1)


@pytest.mark.parametrize("expression", ["x<y<2", "x==y==0", "x>=y!=0"])
def test_unparenthesized_chains_are_rejected(expression: str) -> None:
    with pytest.raises(HybridVerificationError, match="qmod.operator"):
        parse(
            f"qfunc main(output x:qnum<2>,output y:qnum<2>,output flag:qbit) {{ x=0; y=0; flag={expression}; }}"
        )


@pytest.mark.parametrize("expression", ["x<y", "(x==y)*x", "x*(1<2)"])
def test_phase_arguments_keep_the_polynomial_source_boundary(expression: str) -> None:
    with pytest.raises(HybridVerificationError, match="qmod.phase"):
        parse(
            f"qfunc main(output x:qbit,output y:qbit) {{ allocate(x); allocate(y); phase({expression},pi); }}"
        )


def test_arithmetic_precedence_and_explicit_nested_comparisons() -> None:
    imported = parse(
        "qfunc main(output x:qnum<2>,output y:qnum<2>,output flag:qbit) { x=1; y=2; flag=x+1<y*2; }"
    )
    assert [op.name for op in imported.functions[0].body.blocks[0].operations] == [
        "arith.constant",
        "qint.add",
        "arith.constant",
        "qint.multiply",
        "qint.less",
        "qint.return",
    ]
    imported = parse(
        "qfunc main(output x:qnum<2>,output y:qnum<2>,output flag:qbit) { x=1; y=2; flag=(x<y)<2; }"
    )
    assert [op.name for op in imported.functions[0].body.blocks[0].operations].count(
        "qint.less"
    ) == 2


def verify_comparison_bytes(payload: bytes, license_bytes: bytes) -> None:
    blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
    if (
        hashlib.sha1(blob).hexdigest() != BLOB_SHA
        or hashlib.sha256(payload).hexdigest() != SOURCE_SHA
    ):
        raise ValueError("comparison source integrity mismatch")
    if hashlib.sha256(license_bytes).hexdigest() != LICENSE_SHA:
        raise ValueError("comparison license integrity mismatch")


def test_added_original_source_and_shared_license_have_pinned_identities() -> None:
    manifest = json.loads((CORPUS / "comparison_sources.json").read_text())
    assert manifest["repository"] == "https://github.com/Classiq/classiq-library"
    assert manifest["commit"] == SUPPORTED_SOURCE_VERSION
    assert (
        manifest["license_file"] == "LICENSE.txt"
        and manifest["license_sha256"] == LICENSE_SHA
    )
    assert manifest["fixtures"] == [
        {
            "file": "comparator_integer_example.qmod",
            "upstream_path": "functions/function_usage_examples/arithmetic/comparator/comparator_integer_example.qmod",
            "git_blob_sha1": BLOB_SHA,
            "sha256": SOURCE_SHA,
        }
    ]
    verify_comparison_bytes(
        (CORPUS / "comparator_integer_example.qmod").read_bytes(),
        (CORPUS / "LICENSE.txt").read_bytes(),
    )
    assert len(json.loads((CORPUS / "manifest.json").read_text())["fixtures"]) == 3


@pytest.mark.parametrize("source", [False, True])
def test_corrupted_comparison_source_or_license_is_refused(source: bool) -> None:
    payload = (CORPUS / "comparator_integer_example.qmod").read_bytes()
    license_bytes = (CORPUS / "LICENSE.txt").read_bytes()
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_comparison_bytes(
            payload + b"\n" if source else payload,
            license_bytes if source else license_bytes + b"\n",
        )
