# Experiment files

`fq.save` writes a **FlagQuantum Experiment** (`.fqx`); `fq.load` reads one.
An experiment is a portable program and optional recorded numeric result. Saving
and loading never compile, execute, contact a provider, or draw new samples.

```python
import flagquantum as fq

program = fq.compile(fq.Circuit(2).h(0).cx(0, 1))
result = fq.run(
    program,
    outputs=fq.counts(),
    options=fq.ExecutionOptions(device="cpu", backend="pytorch", shots=128, seed=42),
)
fq.save("bell.fqx", program, result=result)
experiment = fq.load("bell.fqx")
assert experiment.result.counts == result.counts
assert sum(experiment.result.counts[0].values()) == 128
```

See [the runnable example](../../examples/save_experiment.py). An existing path
raises `FileExistsError`; pass `overwrite=True` to explicitly replace it. Writers
publish a complete archive atomically and remove temporary files on failure.
Paths are local strings or `os.PathLike[str]`; saving requires the `.fqx` suffix,
case-insensitively. Loading recognizes the content even if the file was renamed.

## File responsibilities

| File | Purpose |
| --- | --- |
| `.fqx` | Portable experiment: circuit, optional numeric results and execution record |
| `.py` | Editable Python source |
| `.ipynb` | Notebook document, cells and outputs |
| `.qasm` | OpenQASM circuit interchange |
| `.json`, `.csv` | Standalone structured data or tabular result export |
| Existing checkpoint formats | Training, optimizer or process recovery |
| Existing deployment formats | Deployment packages and provider-specific delivery |

Only experiment bundles use `.fqx`. This API introduces no `.fqc` or `.fqr`
formats and does not change existing checkpoint, program-artifact or deployment
contracts. It introduces no Web dependency or automatic browser publication.

## Program and result binding

`program` accepts `Circuit`, `CircuitIR`, a static `ExecutionPlan`, or an
`Experiment` returned by `load`. Results must retain a static CircuitIR execution
plan (plan version `1.0`). Planless, opaque native, dynamic, Ising and dynamics
results are outside this first format version and fail explicitly.

The provided program must match the result's executed IR. Only runtime config
metadata and Runtime-added measurements (when the input had no measurements)
are ignored. No recompilation guesses equivalence. If optimization or scheduling
changed the representation, archive the actual plan instead:

```python
fq.save("executed.fqx", result.plan, result=result)
```

Loaded arrays are detached, owned CPU tensors with the same dtype and shape;
measurement ordering, qubit ordering, shots, batch values, accuracy, provenance
and numeric diagnostics are preserved. Saving does not materialize an absent
backend state. Opaque objects, custom classes and native handles are rejected.
This is not an autograd, optimizer, distributed-process or provider-session
checkpoint.

`experiment.program` is the existing `CircuitIR`. `experiment.result` is an
optional `ExecutionResult`; its `plan` is `None`. The recorded plan is available
as a fresh dictionary in `experiment.execution_plan`, for inspection rather than
execution authority. Live plan validation remains unchanged. Explicitly calling
`fq.run(experiment.program, ...)` creates a new execution with current settings.
To preserve historical execution data without replay, save the whole record:

```python
fq.save("copy.fqx", experiment)
```

Do not pass a separate `result` when saving an `Experiment`.

## Version 1.0 wire format

The container is an ordinary ZIP with schema `flagquantum.experiment`, version
`1.0`. The proposed media type is `application/vnd.flagquantum.experiment+zip`;
this is not a claim of IANA registration.

- `manifest.json`: schema, version, SDK producer, program content hash, and a
  complete table of member byte sizes and SHA-256 digests.
- `circuit.json`: unchanged CircuitIR `1.0` JSON.
- Optional `result.json`: tagged data tree, historical static plan and compiler
  identity. Mapping items preserve string versus integer keys; tuples and lists
  remain distinct.
- Optional `arrays/0000.bin`, etc.: tensor bytes referenced from the data tree by
  member, dtype and shape. Bytes are little-endian; storage is contiguous.

Supported tensors: bool, uint8, int8/16/32/64, float16, bfloat16, float32/64,
complex64/128, including empty and scalar tensors. The codec requires no NumPy,
pickle or executable object deserializer. Metadata must consist of supported
JSON primitives, mappings, sequences and tensors, with finite JSON numbers.

Readers reject unsupported versions, mismatched identities, unknown or duplicate
members, duplicate JSON keys, invalid tensor descriptors, unreferenced arrays,
corruption and truncated archives. No archive paths are extracted. Digests detect
corruption; they do not authenticate the sender.

Limits are 256 ZIP members, 16 MiB per JSON member, 64 MiB per tensor and 256 MiB
of total uncompressed data. Archive size and ZIP directory size are checked before
member enumeration. ZIP64, encrypted members, archive comments, unsupported
compression and big-endian tensor hosts are not supported. Files outside these
bounds fail with `SerializationError`. OS access errors remain `OSError`.

The [candidate contract](../../contracts/experiment-file-v1-candidate.json) and
[API proposal](../api-changes/FQ-EXPERIMENT-FILES-20261029.md) define this additive
surface; existing public serialization formats remain authoritative.
