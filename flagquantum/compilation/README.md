# Transitional compilation planning

This directory retains the existing execution-plan model, serialization,
resource estimates, backend/mode selection, and noise-planning code while those
responsibilities are separated between Runtime and Core.

Program optimization, instruction scheduling, backend lowering, coupling maps,
and topology routing are authoritative in `flagquantum/compiler`. Do not add a
second implementation here. Start in `planner.py` for the remaining stable
planning path and run the CPU vertical-slice and execution-plan contract tests.
