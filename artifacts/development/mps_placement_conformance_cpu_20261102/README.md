# Reproducible sharded/unsharded MPS placement conformance

This is the corrected reproduction record for the placement leg of #899.
The nine rank payloads and their logs were generated from committed producer
`71003d31521f6106feaacfe83fd46ef97fac9d44`, based on main
`ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62`. Environment: Python 3.12.14,
PyTorch 2.14.1, macOS arm64, CPU, Gloo, one host, one thread per process,
complex128 amplitudes and float64 parameters. This is development evidence;
it makes no performance, capacity, accelerator or scalability claim.

## Reproduce

From a checkout with the project dependencies installed, use the following
commands. To replay the historical producer exactly, check out the full source
commit above first. The full command, source commit, seed, parameters,
precision, rendezvous, interface and comparison tolerances are also in each
rank's JSON. Install `torch==2.14.1` to reproduce the recorded arithmetic.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
# The recorded macOS run explicitly used its loopback interface.
export GLOO_SOCKET_IFNAME=lo0
# On Linux use the actual interface name, for example lo for local loopback.
for world in 2 3 4; do
  python -m benchmarks.mps_placement_conformance \
    --local-world-size "$world" --timeout-seconds 90 \
    --output "artifacts/development/mps_placement_conformance_cpu_20261102/world-$world.json"
done
python -m pytest tests/distributed/test_mps_placement_conformance.py -q
```

The built-in launcher creates a fresh, initially nonexistent file store and one
worker process per rank. Gloo still needs permission to open local TCP sockets;
the file store avoids hostname-based rendezvous but does not eliminate network
transport. A restricted environment that denies `uv_bind` cannot run this test.
There is no silent launcher or backend fallback. For an existing launcher,
`torchrun --standalone --nproc-per-node=2 benchmarks/mps_placement_conformance.py
--output /tmp/placement.json` remains available and records environment rendezvous.

Each worker logs initialization, arm start and arm completion beside its JSON.
The parent reports rank exit status while waiting, stops after the declared
deadline, terminates remaining workers after a timeout or rank failure, waits
for them, and closes the rank logs. The process-group deadline also bounds
collectives. Failure/timeout cleanup is exercised by
`tests/unit/test_conformance_launcher.py`.

## Matched program and result

Each world size holds the circuit, parameters, observable and bond cap fixed
between the rank-owned reverse executor and the unsharded `run_mps` reference.
Caps 64 and 2 respectively measure an untruncated and a truncating arm. The
truncating arm explicitly authorizes the retained-subspace approximation on
both sides; agreement across placements does not make that rule exact.

| World | Width | Cap | Value relative error | Gradient relative L2 error |
| --- | --- | --- | --- | --- |
| 2 | 6 | 64 | 0 | 2.20108e-16 |
| 2 | 6 | 2 | 1.11761e-16 | 3.65321e-16 |
| 3 | 8 | 64 | 1.11735e-16 | 4.68537e-16 |
| 3 | 8 | 2 | 5.58967e-16 | 5.98073e-16 |
| 4 | 10 | 64 | 5.58677e-16 | 4.74956e-16 |
| 4 | 10 | 2 | 2.23688e-16 | 3.83593e-16 |

All six arms meet their declared `1e-12` value and gradient bounds. Cosine
deficit also remains below `1e-12`. Every rank retains site ownership, parameter
ownership, saved tensor bytes, communication counters, backward status,
fallback/reconstruction flags and the full executor summary. Process peak RSS
is an OS high-water mark, not peak live tensor memory or a scalability result.

The measured scope remains worlds 2/3/4, widths 6/8/10, one seed, the two caps,
one observable and CPU first-order reverse execution. State fidelity across
placements, optimizer steps, checkpoint recovery, multi-node transport and
accelerator behavior are not measured. State fidelity and multiple observables
on the single-device representations are covered by the separate corrected
representation corpus. The rank evidence checks are in
`tests/integration/test_representation_conformance_evidence.py` in the revision
containing this README.
