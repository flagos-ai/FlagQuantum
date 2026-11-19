"""Frozen source witnesses for logical graphs with non-executable descriptors."""

from hashlib import sha1
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
sympy = pytest.importorskip("sympy")


def test_original_multityped_builder_contract(pytestconfig: pytest.Config) -> None:
    from qualtran._infra import composite_bloq

    pytestconfig.addinivalue_line("markers", "notebook: upstream notebook witness")
    from qualtran._infra import composite_bloq_test

    for module, blob in (
        (composite_bloq, "b0317f675c96f9cce1581fcbb7030c48ab32a529"),
        (composite_bloq_test, "83dfa234bc95c8310f1d3d8526fbb2aa9b538a7f"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob
    composite_bloq_test.test_type_error()


@pytest.mark.parametrize("shape", [(), (2,), (0,)])
@pytest.mark.parametrize("kind", ["opaque", "uint", "signed", "bounded", "fixed"])
def test_source_symbolic_width_identity_keeps_concrete_elements(
    kind: str, shape: tuple[int, ...]
) -> None:
    from qualtran import (
        BloqBuilder,
        BloqError,
        BQUInt,
        QAny,
        QFxp,
        QInt,
        QUInt,
        Register,
        Signature,
    )

    width = sympy.Symbol("n", integer=True, positive=True)
    dtype = {
        "opaque": QAny(width),
        "uint": QUInt(width),
        "signed": QInt(width),
        "bounded": BQUInt(width, 2),
        "fixed": QFxp(width, 1),
    }[kind]
    signature = Signature((Register("q", dtype, shape=shape),))
    builder, inputs = BloqBuilder.from_signature(signature)
    result = builder.finalize(q=inputs["q"])
    assert result.signature == signature
    count = int(np.prod(shape)) if shape else 1
    assert len(result.connections) == count
    if count:
        copied = result.copy()
        assert copied == result
    else:
        # The frozen copy path drops empty output mappings.
        with pytest.raises(BloqError, match="Finalizing requires"):
            result.copy()
