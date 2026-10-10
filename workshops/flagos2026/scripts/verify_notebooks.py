"""Execute workshop notebooks locally; refuse enabled remote-action switches."""

import argparse
import ast
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REMOTE_SWITCHES = {
    "SUBMIT",
    "CANCEL",
    "submit_job",
    "check_job",
    "cancel_job",
    "submit_to_hardware",
    "submit_trained_circuit",
    "submit_sweep",
    "compile_for_hardware",
    "collect",
}
SENSITIVE_NAMES = {
    "QUAFU_API_KEY",
    "QUAFU_API_TOKEN",
    "JIUDING_AK",
    "JIUDING_SK",
    "QBOSON_SDK_CODE",
}
BANNED_CODE = ("export QUAFU_", "pip install flagquantum")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="flagquantum-workshop")
    parser.add_argument(
        "--notebook", help="One notebook filename; defaults to all labs"
    )
    args = parser.parse_args()
    workshop = Path(__file__).resolve().parents[1]
    paths = sorted((workshop / "notebooks").rglob("*.ipynb"), key=lambda p: p.name)
    if not paths:
        parser.error("No notebooks found")
    if args.notebook:
        paths = [p for p in paths if p.name == args.notebook]
        if not paths:
            parser.error("No notebook matches that filename")
    prepared = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.execution_count is not None or cell.outputs:
                raise ValueError(f"{path.name}: clear saved execution state")
            lowered = cell.source.lower()
            for snippet in BANNED_CODE:
                if snippet.lower() in lowered:
                    raise ValueError(f"{path.name}: remove `{snippet}` from code cells")
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id in SENSITIVE_NAMES
                            and isinstance(node.value, ast.Constant)
                            and node.value.value
                        ):
                            raise ValueError(
                                f"{path.name}: do not assign credentials in a notebook"
                            )
                        if (
                            isinstance(target, ast.Name)
                            and target.id in REMOTE_SWITCHES
                        ):
                            if (
                                not isinstance(node.value, ast.Constant)
                                or node.value.value is not False
                            ):
                                raise ValueError(
                                    f"{path.name}: set {target.id} to False before validation"
                                )
        prepared.append((path, notebook))
    output = workshop / "outputs" / "validation"
    output.mkdir(parents=True, exist_ok=True)
    for path, notebook in prepared:
        NotebookClient(
            notebook,
            timeout=90,
            kernel_name=args.kernel,
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        nbformat.write(notebook, output / path.name)
        print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
