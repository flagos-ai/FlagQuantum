"""Execute workshop notebooks locally; refuse enabled remote-action switches."""

import argparse
import ast
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager
from nbclient import NotebookClient

REMOTE_SWITCHES = {
    "SUBMIT",
    "CANCEL",
    "submit_job",
    "check_job",
    "cancel_job",
    "submit_to_hardware",
    "submit_to_cloud",
    "submit_batch",
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
    parser.add_argument("--language", choices=("en", "cn", "both"), default="both")
    parser.add_argument(
        "--notebook", help="One notebook filename; defaults to all labs"
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="Also validate optional notebooks in reference_notebooks/",
    )
    args = parser.parse_args()
    workshop = Path(__file__).resolve().parents[1]
    directories = {
        "en": ("notebooks",),
        "cn": ("notebooks_cn",),
        "both": ("notebooks", "notebooks_cn"),
    }[args.language]
    paths = [
        path
        for directory in directories
        for path in sorted((workshop / directory).glob("*.ipynb"))
    ]
    if args.include_references:
        paths += sorted((workshop / "reference_notebooks").rglob("*.ipynb"))
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
            parsed = ast.parse(TransformerManager().transform_cell(cell.source))
            for node in ast.walk(parsed):
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
        drawing_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and '.draw(format="mpl"' in cell.source
        ]
        for cell in drawing_cells:
            pngs = [
                output.get("data", {}).get("image/png")
                for output in cell.outputs
                if "image/png" in output.get("data", {})
            ]
            if not pngs or any(
                not isinstance(png, str) or len(png) < 100 for png in pngs
            ):
                raise RuntimeError(f"{path.name}: circuit drawing has no inline PNG")
        destination = output / path.parent.name / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        print(
            f"PASS {path.parent.name}/{path.name}: all cells; "
            f"{len(drawing_cells)} circuit drawing cells with inline PNG",
            flush=True,
        )


if __name__ == "__main__":
    main()
