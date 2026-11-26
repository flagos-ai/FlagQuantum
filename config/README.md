# Repository configuration

Architecture boundaries and team ownership are maintained here. Packaging metadata
remains at the repository root, as required by its existing build entry points.
The dependency, maturity and evidence-origin policies keep their current root
locations until their #1084 consumers can move atomically.

From the repository root, use:

```sh
conda env create -f config/environments/environment.yml
python tools/check_architecture.py
python tools/check_team_scope.py --validate
```

The editable pip requirement in the environment file is relative to that file's
directory. This preserves the same repository checkout and extras after relocation;
see [Conda's pip installer](https://github.com/conda/conda/blob/main/conda/env/installers/pip.py).
No dependency versions are changed by the move.
