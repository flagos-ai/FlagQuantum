# Public declared-batch CPU comparison v1

Frozen before the repeated measurement campaign. Development evidence only.

- Widths 2, 8, 12; batches 1, 32; depth 2. Three fresh processes per shape.
- Identical RY/RZ layers and nearest-neighbor CX chain; deterministic linspace
  parameters and features; exact complex128/float64; CPU, one thread.
- Mean squared normalized Z sum against 0.3; shared parameter gradients;
  fixed features remain runtime arguments. SGD learning rate 0.01.
- Public FlagQuantum default Torch and explicitly selected JAX statevector/TN.
- TensorCircuit exact Circuit with dense state diagonal reduction and bond-two
  Z-sum MPO, official Torch interface with NumPy and DLPack conversion.
- Separate whole-JAX comparators compile quantum loss, gradient and SGD update.
  They are a stronger execution category; FlagQuantum private JAX kernels are
  excluded from primary results. These candidates do not exhaust all possible
  TensorCircuit implementations or optimizers.
- Setup plus first forward/backward recorded separately. Three warmup updates,
  five blocks of twenty updates; blocking synchronization. Rotate candidate
  order between fresh processes. Preserve all blocks and first-call costs.
- Check initial loss/parameter gradients and retain all final parameter vectors
  to verify 103-update trajectories; retain errors and 180-second process
  deadlines. Peak RSS is cumulative process RSS through each candidate, not
  isolated simulator allocation or capacity evidence.
- Record source commit, producer hash, working-tree diff hash, versions and
  platform. Raw original output remains separate from sanitized publication.

Command: `.venv/bin/python benchmarks/batched_training_comparison.py --output
<raw-output.json> --repeats 3`. No performance release threshold or universal
fastest claim follows from this finite, single-host development matrix.
