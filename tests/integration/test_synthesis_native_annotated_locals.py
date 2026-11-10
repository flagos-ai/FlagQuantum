"""Standard local width metadata shares the existing checked lifetime rules."""

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "annotation", ["QNum[1]", "Annotated[QNum, 1]", "typing.Annotated[QNum, 1]"]
)
def test_annotation_spellings_preserve_phase_and_clean_workspace(
    annotation: str,
) -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[2]]",
            "allocate(x)\nhadamard_transform(x)\nphase(0.13*x,1)\n"
            f"with within():\n    mark: {annotation} = x < 1\n"
            "with apply():\n    phase(0.37*mark,1)",
            "import typing\nfrom typing import Annotated\n",
        )
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    reference.phase(0, theta=0.26).phase(1, theta=0.13)
    reference.x(0).x(1).cphase(0, 1, theta=0.37).x(0).x(1)
    compare_states(program, reference, ("x",))


@pytest.mark.parametrize(
    "annotation",
    [
        "Annotated[QNum, 0]",
        "Annotated[QNum, 257]",
        "Annotated[QNum, True]",
        "Annotated[QNum, -1]",
        "Annotated[QNum, (1, False, 0)]",
        "Annotated[QNum, 1, 2]",
        "Annotated[QBit, 1]",
        "Annotated[int, 1]",
        "Annotated[QNum, 1 + 1]",
        "Annotated[QNum, width]",
    ],
)
def test_invalid_local_annotations_fail_before_lowering(annotation: str) -> None:
    with pytest.raises(HybridVerificationError, match="local|annotation"):
        synth.synthesize(
            definition(
                "x:Output[QNum[2]]",
                "allocate(x)\nwith within():\n"
                f"    mark: {annotation} = x < 1\nwith apply():\n    phase(mark,0.37)",
                "from typing import Annotated\nwidth = 1\n",
            )
        )


def test_annotated_local_does_not_allow_stale_cleanup() -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "x:Output[QNum[2]]",
                "allocate(x)\nwith within():\n"
                "    mark: Annotated[QNum, 1] = x < 1\nwith apply():\n    X(x)",
                "from typing import Annotated\n",
            )
        )
