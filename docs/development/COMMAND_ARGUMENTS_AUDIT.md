# Complete command argument source audit

## Interface and ownership before implementation

Frozen Kickmix `util/arg_parse.cc`, `.h` and `.test.cc` are K-0284 through
K-0286 at `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0.
All five source/test-helper blobs are verified and all 11 unchanged original
tests pass in an isolated temporary directory.

Ecosystem owns private command scanning, typed flag conversion, enum reference
selection and edge-owned file acquisition. It consumes plain native strings,
scalars and streams. Core and Simulation own no CLI state. Compiler's existing
binary output boundary retains complete-result writes. The existing KMX command
entry points use argparse for command orchestration and help. Numeric callbacks
can reuse the source integer grammar through its private authoritative owner.
Complete command entry source parents remain separate from this helper group.

Native scanning takes argv including the program name and returns the first
matching flag. It preserves empty/missing distinction, equals syntax, -- and
the source's differing negative-value rules for discovery and unknown checks.
Integer parsing retains sign-only zero and underscore admission after a nonzero
accumulator. Float32 and float64 use the host C runtime strtof/strtod directly,
matching the source runtime instead of double-rounding through Python float.
ctypes is a standard-library dependency. C objects remain private to this
boundary and native functions return only float values. Calls use owned buffers
and retain the C runtime's parse endpoint. No process locale is changed.

Enum lookup returns the original mapped object. A missing default key maps
source out_of_range to KeyError. File acquisition is a context manager that
closes a newly opened handle and preserves caller-owned defaults. Source fopen
binary modes map to Python binary modes. Text/newline behavior and unsupported
C mode strings need explicit comparison boundaries. No quantum primitives,
measurement branches, work bits or quantum resource model belong to parsing.

## Definition disposition and evidence

| Source definition | Native owner and observed behavior |
| --- | --- |
| find_argument and require_find_argument | find_argument/require_argument retain first match, duplicates, terminator, empty values, prefix matching and C-string truncation |
| check_for_unknown_arguments | check_unknown_arguments retains known/deprecated loop order, mode skip, negative-token distinction and sorted diagnostics |
| parse_int64 and find_int64_argument | _command_numbers plus int64_argument retain uint64 intermediate arithmetic, signed bounds, sign-only zero and underscore grammar |
| find_float_argument and find_double_argument | float_argument selects direct strtof/strtod, exact scalar-overload defaults/bounds and signed zero; failure diagnostics match |
| find_bool_argument | bool_argument distinguishes absent, bare and nonempty flags |
| find_enum_argument template | enum_argument retains mapped-object identity and missing-default KeyError corresponding to original out_of_range |
| find_open_file_argument | open_file_argument preserves defaults and standard host-POSIX byte modes with explicit scoped ownership; nonstandard modes remain open |
| Existing KMX commands | sample, spin and sampled-counts use the shared integer callback through their actual main entry points |

Five frozen blobs are checked before compiling the unchanged source. All eleven
unchanged original tests pass in an isolated temporary working directory.
The 2,498-record corpus independently exercises source values, exact diagnostics,
floating-point bits, enum results, file contents/default identity and refusals.
It regenerates byte-for-byte, with hex-encoded observations and a quoted wire
format preserving tabs, newlines and NUL bytes. It includes all standard file
mode spellings exercised here and exclusive-open success/failure. Native tests
also prove default-stream identity, handle closure on success and consumer
failure, enum object identity and actual CLI callback replacement.

Concentrated regression passes 353 tests, and 55 local register-execution,
padding and phase-overlap integration tests pass. Strict mypy passes all seven
changed Python files. Black, Ruff, Ecosystem scope, architecture, internal IR,
protected API baseline and whitespace checks pass. The two private scalar
owners have 99% combined statement/branch coverage. The unreachable defensive
C-result type check is unexecuted. Dependency policy reports 720 existing
whole-repository findings and zero findings in touched paths.

## Source defects and remaining boundaries

The original uint64 overflow predicate accepts `18446744073709551616` as zero
and the next few unsigned wrap aliases. Native preserves the exact predicate,
including rejection of `18446744073709551620`. These are original observations,
not an amended reference. Ordinary signed-bound failures remain distinct.

The float32 witness `1.0000000596046447753906250000000001` rounds directly to
`0x1.000002p+0`, while conversion through Python double then float32 gives one.
The native direct-C route matches the original. NaN explicit values fail,
whereas a NaN default can return under the original comparisons.

On the measured macOS host, original fopen accepts the nonstandard `re` mode.
Native explicitly refuses it. A source corpus record and an asserted native
refusal retain this mismatch. **K-0284 remains open** for host-specific C mode
semantics, Windows text translation and unverified C-runtime portability.
K-0285 and K-0286 have complete declaration and original-test disposition for
the verified native routes. Missing default enum keys follow actual source
out_of_range behavior despite the source header's invalid_argument prose.

Argparse continues to own existing command orchestration/help. Full main_*.cc
scope is separately inventoried. Scoped binary streams adapt original caller
fclose ownership. These helpers are classical edge operations with zero emitted
quantum primitives, measurements or work qubits. There is no declared source
quantum resource model. Accepted full-complex-amplitude, measurement-branch and
clean-work register evidence is reused by the local integration suite.

## Reproduction

```bash
python tests/unit/fixtures/generate_command_arguments.py \
  --source-root /path/to/frozen/kickmix/src \
  --output /tmp/command_arguments.jsonl
cmp tests/unit/data/command_arguments.jsonl /tmp/command_arguments.jsonl
python -m pytest tests/unit/test_command_arguments.py \
  tests/unit/test_kmx_sample_command.py tests/unit/test_kmx_spin_command.py \
  tests/unit/test_kmx_sampled_counts.py -q
python -m flagquantum.ecosystem.kmx_sample --in input.kmx --out output.txt --shots 1__2
```

The last command takes twelve samples using the original signed-integer grammar.
Raw source-run, regression, integration and environment receipts are retained in
`ftqc-reference-env/g2-command-arguments-*20261122.*` beside the main ledgers.
