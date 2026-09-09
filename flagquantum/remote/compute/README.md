# Remote compute

This directory owns experimental adapters to external CPU/GPU job systems.
Jiuding currently supports one CPU or single-GPU instance, automatic workspace context
discovery, create-and-start submission, status, waiting, shared JSON results,
and stopping active jobs. It uses the standard library and direct HTTPS calls.
It does not own circuit execution, numerical backend selection, gradients,
distributed launch, or the QPU shots/counts contracts. No Stable Core exports
are added. These adapter-specific interfaces are not frozen.

Start with `JiudingClient` in `jiuding.py` and
`examples/remote/jiuding_submit.py`. A user script defines `main()` returning a
JSON-serializable value; `_worker.py` calls it and writes a run-bound artifact.
Complex functions and local imports can live in that script's shared project.
The caller must provision the shared code and a compatible image first.

Read `docs/guides/JIUDING.md` for the supported journey and limits. Run
`python -m pytest tests/team/remote/test_jiuding.py -q` for offline behavior
tests. No test in that file creates real tasks. The Bell example provides a
small CPU numerical check for live acceptance; `jiuding_bell_gpu.py` checks
CUDA execution with one visible GPU. A new provider-wide contract,
root export or distributed claim requires a separate reviewed change.
