"""Read the declared build and runtime requirements for an image profile."""

import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

root = Path(__file__).resolve().parent
config = tomllib.loads((root / "pyproject.toml").read_text())
profile = json.loads((root / "profiles.json").read_text())[sys.argv[1]]
project = config["project"]
# CI builds against the preinstalled runtime. The older Torch constraint
# in build-system.requires is for the isolated release build.
requirements = [
    requirement
    for requirement in config["build-system"]["requires"]
    if not requirement.startswith("torch>=")
]
requirements.extend(project["dependencies"])
for extra in profile:
    requirements.extend(project["optional-dependencies"][extra])
print(*requirements, sep="\n")
