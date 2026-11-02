"""UP-00 keeps future program syntax outside current integer source admission."""

import pytest

from flagquantum.compiler._construction.frontend_python import capture, qfunc
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "source",
    [
        "def main(q: int) -> float:\n    return 1.9*(q-0.5)\n",
        "def main(q: 'Output[QBit]'):\n    allocate(q)\n",
        "def main(q: 'Const[QNum[2, False, 0]]') -> int:\n    return q\n",
        "def main(q: 'Input[QBit]') -> int:\n    return q\n",
    ],
)
def test_program_annotations_fail_before_effect_capture(source: str) -> None:
    function = _cell(source, filename="<up00-program-annotation>")
    for construct in (lambda: capture(function, {"q": 1}), lambda: qfunc(function)):
        with pytest.raises(
            HybridVerificationError, match="construction.python.annotation"
        ):
            construct()


@pytest.mark.parametrize("expression", ["1.9*q", "0.5", "q-0.5"])
def test_unannotated_float_data_cannot_bypass_integer_capture(expression: str) -> None:
    function = _cell(f"def main(q: int):\n    return {expression}\n")
    with pytest.raises(HybridVerificationError):
        capture(function, {"q": 1})


@pytest.mark.parametrize(
    "body",
    [
        "if q:\n        q=q+1\n    return q",
        "control(q, lambda: phase(0.3))\n    return q",
        "free(q)\n    return q",
        "drop(q)\n    return q",
    ],
)
def test_effectful_body_is_not_reinterpreted_as_pure_arithmetic(body: str) -> None:
    function = _cell(f"def main(q: int) -> int:\n    {body}\n")
    with pytest.raises(HybridVerificationError, match="construction.python.statement"):
        capture(function, {"q": 1})
