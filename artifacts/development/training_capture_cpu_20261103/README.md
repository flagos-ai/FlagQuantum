# Conditional FX binding code generation

Module builder capture uses Torch FX to preserve symbolic quantum topology and
the classical binding program. The FX graph already has executable code when
`make_fx` returns. If the subsequent dead-code pass removes nodes, Compiler
regenerates code for the reduced graph. When the pass leaves the graph intact,
the existing code is reused.

The comparison uses ordinary `fq.train` with Adam, the frozen cases 0, 3 and
5, the same initial values and changing inputs, and five pairs of fresh
processes per shape. Process order alternates by trial. Both legs run the
current Compiler source. The baseline harness reports a changed graph at
the Compiler's cleanup call to reproduce its previous unconditional
regeneration. It runs the same cleanup pass and records the original return
value. Calls made inside Torch tracing are passed through unchanged.

The harness imports Torch before the worker's training timer so it can
instrument capture. First-update time includes capture and the public
update, with interpreter/framework imports excluded. Capture duration is
measured inside the existing Compiler entry. The final checkpoint replay
runs in both legs. A separate batch-shape probe measures fresh capture on
the same Module with another batch size.

```bash
D=artifacts/development/training_capture_cpu_20261103
PYTHONPATH=. python "$D/run.py" --output /tmp/fq936-fx-capture.json
```

The raw receipts pin the Compiler and harness SHA-256, the existing worker
digest, source commit, timing samples, trajectory hashes and recovery
errors. Each trajectory hash covers all 60 losses, parameters, gradients and
consumed IR hashes.

## Results

All 30 workers passed. All 15 pairs produced identical trajectory hashes,
and each route replayed its checkpoint with zero final parameter error.
All observed Compiler cleanup calls left their graph intact.

| Case (width/depth/batch) | First update, before → after | Initial capture, before → after | First changed batch, before → after |
| --- | ---: | ---: | ---: |
| 0 (3/2/1) | 67.52 → 70.63 ms | 51.01 → 52.17 ms | 18.35 → 17.40 ms |
| 3 (5/4/64) | 103.42 → 109.67 ms | 81.60 → 88.03 ms | 109.12 → 85.14 ms |
| 5 (7/4/1) | 130.75 → 117.04 ms | 109.28 → 97.43 ms | 79.28 → 71.73 ms |

Values are five-process medians. Initial capture and first-update
measurements are mixed, while the changed-batch medians fell in this run.
The change removes one unnecessary code-generation pass and the timing
campaign does not establish a general first-update speedup. It does not
satisfy the default compiled benefit required by #936 or the public
performance acceptance of #918.

The Module capture/replacement, training state, versioned checkpoint
contract, and training compilation suite passed all 148 focused tests.
Compiler strict typing, Ruff and the evidence privacy check passed.
