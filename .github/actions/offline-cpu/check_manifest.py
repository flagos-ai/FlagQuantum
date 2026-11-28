"""Check that a checkout can use the dependencies in a locked CPU image."""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the locked CPU image.
    import tomli as tomllib


def offline_manifest(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    # These fields only affect distribution contents and release-wheel testing.
    config["project"].pop("license-files", None)
    tools = config.get("tool", {})
    tools.pop("cibuildwheel", None)
    # Lint configuration does not change the installed dependency environment.
    tools.pop("ruff", None)
    # Test selection and measurement are read from the checkout, not installed
    # into the image. Project/build requirements and unknown tools stay checked.
    tools.pop("pytest", None)
    tools.pop("coverage", None)
    if not tools:
        config.pop("tool", None)
    return config


if __name__ == "__main__":
    checkout, image = map(Path, sys.argv[1:])
    if offline_manifest(checkout) != offline_manifest(image):
        sys.exit("The checkout manifest differs from the locked CPU image manifest.")
