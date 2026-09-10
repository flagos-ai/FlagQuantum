from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(
    r"^```python\n(.*?)^```",
    flags=re.MULTILINE | re.DOTALL,
)


@pytest.mark.integration
def test_readme_golden_path_executes() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = PYTHON_FENCE.search(text)

    assert match is not None
    namespace = {"__name__": "__readme_golden_path__"}
    exec(compile(match.group(1), "README.md:first-python-block", "exec"), namespace)

    assert namespace["training"].completed_steps == 10
    assert namespace["result"].plan is not None
    assert namespace["training"].final_loss < namespace["training"].losses[0]
    # RY(theta) followed by CX preserves <Z_0> = cos(theta).
    angle = next(namespace["model"].parameters()).detach().squeeze()
    torch.testing.assert_close(
        namespace["result"].expectation().squeeze(),
        angle.cos(),
        atol=1e-6,
        rtol=1e-6,
    )
