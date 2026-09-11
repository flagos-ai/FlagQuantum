from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVICE_PATHS = (ROOT / "flagquantum" / "services" / "preflight.py",)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return tuple(imported)


def test_services_have_no_protocol_llm_kernel_or_vendor_imports() -> None:
    forbidden_roots = {
        "anthropic",
        "braket",
        "compute",
        "cuda",
        "fastmcp",
        "langchain",
        "llama_index",
        "mcp",
        "openai",
        "qiskit_ibm_runtime",
        "quafu",
        "remote",
        "simulation",
        "torch_fl",
        "transformers",
    }
    violations: list[str] = []
    for path in SERVICE_PATHS:
        for module in _imports(path):
            if forbidden_roots.intersection(module.split(".")):
                violations.append(f"{path.relative_to(ROOT).as_posix()}: {module}")

    assert violations == []


def test_local_sdk_and_services_run_without_mcp_sdk() -> None:
    script = r"""
import importlib.abc
import sys

class RejectMCP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"mcp", "fastmcp"}:
            raise ModuleNotFoundError(f"blocked optional protocol SDK: {fullname}")
        return None

sys.meta_path.insert(0, RejectMCP())

import flagquantum as fq
from flagquantum.services import preflight_execution

circuit = fq.Circuit(1).h(0)
preflight = preflight_execution(circuit)
result = fq.run(circuit)

assert preflight.executable
assert result.state is not None
assert not any(name == "mcp" or name.startswith("mcp.") for name in sys.modules)
assert not any(name == "fastmcp" or name.startswith("fastmcp.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
