# Repository configuration

Architecture boundaries and team ownership are maintained here. Packaging metadata
remains at the repository root, as required by its existing build entry points.
Capability maturity is read from `config/capability-maturity.toml` and evidence
origins from `config/evidence-revision-origins.toml`. Their evidence and example
paths remain relative to the repository root. The dependency policy retains its
current root location pending its own integrated consumer closure.

From the repository root, use:

```sh
conda env create -f config/environments/environment.yml
python tools/architecture/check_architecture.py
python tools/architecture/check_team_scope.py --validate
python tools/validation/check_capability_maturity.py
python tools/release/check_evidence_revisions.py
```

The editable pip requirement in the environment file is relative to that file's
directory. This preserves the same repository checkout and extras after relocation;
see [Conda's pip installer](https://github.com/conda/conda/blob/main/conda/env/installers/pip.py).
No dependency versions are changed by the move.
