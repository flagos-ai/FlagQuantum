# Development tools

Owns the local development-environment setup script. Product behavior belongs in
FlagQuantum; architecture rules, CI source transport and release verification
belong in their existing tool directories. Incoming #1084 development programs
are not present or claimed implemented by this move.

From the repository root, inspect the unchanged supported options:

```sh
bash tools/development/setup_flagquantum_dev_env.sh --help
```

The script retains the selected `PYTHON_BIN`, explicit `VENV_DIR`, repository-root
`.venv` default and original editable-install behavior. It locates the same
repository from a foreign working directory. For a typical small change, edit
this script, check `bash -n tools/development/setup_flagquantum_dev_env.sh`, then
verify its help, invalid-option and missing-interpreter cases before installing
anything. Follow [contributor installation](../../CONTRIBUTING.md) for the
existing environment prerequisites.
