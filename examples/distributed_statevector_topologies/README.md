# Reproducible distributed statevector topologies

This example reproduces the 31-qubit, 8-layer differentiable Linear, Ring, or
Brickwork workload used in the distributed statevector evaluation. It uses the same runner as the
measurement artifacts and enables the optimized path by default in the runtime.

## Run on one machine

From the repository's `FlagQuantum/` directory:

For the CUDA-optimized path install the optional extra once:

```bash
pip install -e '.[cuda]'
```

If Triton is unavailable, FlagQuantum automatically uses its PyTorch fallback
and reports that mode in the runtime summary; correctness is unchanged.

Most users should leave the runtime in its default `auto` mode. For a
conservative non-A800 run use `FQ_SV_RUNTIME_MODE=portable`; `max_perf` is for
known, profiled topologies. The many `FQ_SV_*` variables are advanced overrides
and are not required for normal use.

```bash
NPROC=8 ./examples/distributed_statevector_topologies/run.sh
```

Select the topology with one variable:

```bash
ENTANGLEMENT=linear NPROC=8 ./examples/distributed_statevector_topologies/run.sh
ENTANGLEMENT=ring NPROC=8 ./examples/distributed_statevector_topologies/run.sh
ENTANGLEMENT=brickwork NPROC=8 ./examples/distributed_statevector_topologies/run.sh
```

Set `NPROC=2` or `NPROC=4` for smaller machines. The script checks that the
requested number of CUDA devices is visible, prints the effective runtime
configuration, and writes a JSON payload containing every timing sample,
communication counters, memory, and correctness metadata.

Every topology uses 31 qubits, 8 layers, 248 trainable parameters, complex64, and
full value plus reversible-adjoint gradient. No cuQuantum dependency is used.

To compare the safe baseline explicitly:

```bash
FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT=0 NPROC=2 ./examples/distributed_statevector_topologies/run.sh
```
