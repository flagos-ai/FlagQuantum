# Synthesis compiler diagnostics

These contributor tools construct internal compiler records to test dependency
contracts and resource schedules. They are not user-facing native programs.

User examples live in `examples/synthesis/` and use public FlagQuantum APIs.
Run diagnostics from the repository root, for example:

```sh
python -m tools.synthesis_compiler_examples.local_registers.main
```

The corresponding compiler and integration tests preserve their acceptance checks.
