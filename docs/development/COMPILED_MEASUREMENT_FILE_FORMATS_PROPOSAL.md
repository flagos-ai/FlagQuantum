# Compiled measurement file formats

Status: approved implementation candidate, pending new-head acceptance and delivery.
Primary owner: Simulation; Integration coordinates the additive contract and
installation probes. Classification: `single_device_fast_path`.
Approval SHA-256: `97b2ea69aa68534a40f68fd55144f1103f3c7c0fa8d2d12b4a4102425596281a`.

## Public delta and reuse

`CompiledMeasurementSampler.sample_write(shots, *, filepath, format)` accepts
`01`, `b8`, `r8`, `hits`, `dets`, and `ptb64`, with no new default, return value,
root export, Circuit type, or numerical backend. Its original b8 contract and
history remain unchanged. The additive contract records this authorized delta.
The existing `fq.Circuit.from_ir` path and DynamicCircuit inherit the same method.

The private extension dispatches to the existing Stim 1.16.0 batch writer.
All 96 upstream files and the source lock remain byte-identical; there is no
copied Python codec, new dependency, build change, or upstream closure expansion.
The private b8 entry remains callable. The SDK and embedded implementations
serve the same existing consumer boundary.

## Formats, limits, and failures

Let S be positive shots, M explicit measurement occurrences, and d the decimal
width of M-1 (one for M=1). The preflight byte bounds are:

| Format | Upper bound | Encoding |
| --- | --- | --- |
| 01 | S*(M+1), exact | ASCII bits followed by newline |
| b8 | S*ceil(M/8), exact | LSB-first per-shot bytes |
| r8 | S*(M+1), conservative | Zero runs plus terminating virtual true bit |
| hits | S*M*(d+1), conservative | Comma-separated set measurement indices |
| dets | S*(5+M*(d+2)), conservative | `shot` followed by `M`-prefixed indices |
| ptb64 | (S/64)*M*8, exact | 64-shot groups, eight bytes per measurement |

ptb64 refuses S not divisible by 64 before the writer or staging is touched.
All formats retain the 1 GiB final-file cap and check observed free space for
the bound. Sparse output may be refused even when its realized size would fit.
Free space is not reserved and is not a filesystem quota. Existing payload and
principal frame estimate limits remain 64 MiB, with 4096 qubits, 65536 records
and 100000 lowered operations. These estimates do not bound process RSS.
The upstream 1024-shot batch has a maximal principal frame estimate of 9 MiB;
reference, transpose, allocator and library storage are additional. Upstream
streaming scratch may use anonymous temporary files; the final-file cap is not
an aggregate scratch-space guarantee.

Validation streams 64 KiB chunks and keeps scalar parse state, with no full
shot table or per-record token list. Fixed-size output checks exact size;
variable-size output checks complete shot framing, indices and termination.
The parser working-storage budget is 1 MiB. Existing destination/symlink refusal,
atomic no-replace publication and owned staging cleanup remain unchanged.
Writer, validation and publication failures invalidate the sampler and publish
no incomplete file. Preflight refusal consumes no sampling state. T, conditional
H and unsupported controls/noise remain refused before engine preparation.

## User workflow and independent references

Run `python examples/qec/compiled_measurement_sampler.py`. Its existing feedback
example is retained, followed by a deterministic repeated-slot example. Two
measurements of slot 0 retain independent columns `(1, 0)`. For 64 shots the
new output lines are `01 192`, `b8 64`, `r8 128`, `hits 128`, `dets 512`,
and `ptb64 16`.

Tests compare literal byte witnesses with the pinned SDK and use independent
readers to reconstruct columns. They cover empty sparse shots, multi-digit
indices, cross-byte records, 254/255/256 zero runs, 64/128-shot groups, latest
slot and inverted X feedback. Random draws are checked by invariants, without
assuming identical streams across separate calls or architectures.

## Acceptance and scope

The approved scope contains exactly 16 paths, recorded in the reviewed proposal.
The original sampler contract, vendor sources, CI, setup and dependency files
are excluded. Run the three existing sampler/profile test files, Ruff, Black,
strict Mypy, existing executable-docstring tests, and seeded pr-default and
pr-runtime tiers, with explicit timeouts. Then build one offline macOS arm64
embedded wheel using `FQ_BUILD_EMBEDDED_STIM=1 python -m build --wheel --no-isolation`.
In a qualified disposable installed prefix without the Stim distribution,
run `tools/verify_embedded_stabilizer_install.py` and
`tools/qec_installed_smoke.py core` from outside the source tree with PYTHONPATH
unset. Verify installed package/extension provenance and actual native writer
calls. Missing dependencies or any new failed acceptance stop the run; retain
all logs without automatic retries. Final-head example and installed receipts
must precede PR publication.

No research, benchmark, GPU recovery or old CI rerun is part of this delivery.
Detector file output, file conversion and streaming-decoder composition remain
separate gaps. Full Stim/PyMatching parity and FTS-04/R03/R08 remain partial;
FTS-02/R04/M2 do not change.
