# CI stack watchdog

Coverage workers can exit with SIGSEGV while the native delayed stack dumper is
printing diagnostics. CI now requests a Python stack snapshot after 300 seconds
of one test instead of arming `faulthandler.dump_traceback_later`. Test selection,
assertions, coverage collection, resource quotas, failure propagation and job
deadlines are unchanged. Native fatal-signal handling remains enabled by pytest.

This is an integration-owned diagnostic change. It does not modify the compiler,
the FlagQuantum public API or SDKBench acceptance rules.

## Evidence and attribution limits

The [evidence record](evidence/ci-stack-watchdog.json) retains observations,
reproduction sources, native stack symbols, selected CI excerpts and complete-log
hashes. The tested compiler revision is
`9456bc97e35599d884613145fe01657dae36a907`; this change starts from main
`6db1fbda8a168599e5c0f155b9fb14052cf01342`.

In PR #1276, coverage job `137005195254` printed a 300-second stack dump, stopped
at `_polar_angle` with line number zero and reported worker exit status `-11`.
PR #1280 job `137010237032` also reported `-11` during a delayed dump, while
running a different synthesis integration test. These logs establish a signal
and its timing, not the native instruction that caused either Linux failure.

The same Rainbow reference test completed in PR #1279 job `137005234132` and
PR #1280 job `137010230235`, including successful 300-second stack dumps. Their
test-call durations were 387.98 and 378.60 seconds. The checked-in duration hint
is 537.46 seconds. A stack dump at 300 seconds is consequently not evidence that
this test is stuck, and the diagnostic setting was never a 300-second test limit.

On macOS arm64 with Python 3.12.14, the complete reference test passed without
coverage, with sysmon coverage, and with sysmon coverage plus the replacement
watchdog. The replacement was triggered after one second to exercise its output
without lengthening the test. All original reference, native, full-state,
workspace, readout-budget and reduced-density assertions were retained.

A separate perturbation used repeated native dumps every millisecond. It caused
SIGSEGV both during the reference-test pytest invocation and in a plain pytest
integer-arithmetic test that imported neither FlagQuantum nor coverage. The
macOS crash report for the former places the fault in:

```text
dump_frame
dump_traceback
_Py_DumpTracebackThreads
faulthandler_thread
```

The plain pytest control passed without the repeated native dumper. A separate
three-second pure-Python code-object churn control passed with and without it;
the shorter scalar probes also did not reproduce the fault. Thus native delayed
traversal demonstrably can crash independently of FlagQuantum on the local
interpreter, but neither every workload nor every dump reproduces it.

CPython has related upstream reports:
[python/cpython#116008](https://github.com/python/cpython/issues/116008) and
[python/cpython#158200](https://github.com/python/cpython/issues/158200).
They describe lifetime hazards when native traceback traversal reads frames or
code metadata concurrently. They are supporting investigation references, not
proof that either explains these Linux worker exits. No Linux core stack was
obtained. The remote host's exact locked image was unavailable without registry
authorization; no remote reproduction, interpreter upgrade or reboot is claimed.

## Replacement behavior

`tools.pytest_stack_watchdog` wraps each test's existing setup, call and teardown
protocol. A daemon timer requests one snapshot at the configured interval.
`sys._current_frames()` supplies owned Python frame references, and ordinary
Python traceback formatting prints at most 100 frames per thread. References
are released after printing. The plugin duplicates stderr before per-test
capture, including the xdist stderr-proxy case.

Each test cancels and joins its timer before completing. If the timer cannot
stop within five seconds, pytest exits unsuccessfully. Invalid intervals and
simultaneously enabled native delayed dumps are rejected before tests run. A
zero interval disables this plugin. The two coverage commands use:

```bash
python -m pytest ... \
  -p tools.pytest_shard --cov=flagquantum --cov-report= \
  -p tools.pytest_stack_watchdog --fq-stack-dump-after=300 \
  -o faulthandler_timeout=0
```

This snapshot needs the interpreter to schedule Python code. It cannot diagnose
a native operation that holds the GIL indefinitely. The external CI job deadline
remains necessary and unchanged. A snapshot is diagnostic output, not a timeout
verdict or evidence of scientific correctness. The change does not justify
ignoring worker crashes, blindly rerunning failed gates or promoting a compiler
PR whose exact-head checks have not passed.

## Reproduction and validation

Run the watchdog workflow from a normal development environment:

```bash
python -m pytest tests/integration/test_ci_stack_watchdog.py -q
```

Expected result: 11 passes. Cases cover a slow pass, a slow assertion failure,
fast completion, setup failure, per-test timer cancellation, invalid intervals,
conflicting native timers and two real xdist workers. Assertions verify both
diagnostic visibility and retained pytest outcomes; subprocesses have explicit
30-second limits. All 11 cases passed with pytest 7.4.4 and 9.2.0. The plugin
accepts the string-valued zero timeout override used by older pytest versions;
the isolated test runner explicitly fixes its root directory so node IDs remain
comparable. The local run also passed with two outer xdist workers, sysmon
coverage and the replacement watchdog itself active. Its limited instrumentation
reported 47% line coverage of this tooling module; this is not a claim of complete
coverage or a substitute for the behavioral cases.

The production workflow protected by the complete reference check remains:

```bash
python -m examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.comparisons.circuit
```

It checks the native complete quantum state against the independent Circuit
reference. The catalog's frozen-source check additionally executes the original
QMOD path and retains all three amplification counts. For the exact failing
catalog case, run:

```bash
python -m pytest 'tests/integration/test_synthesis_native_catalog.py::test_original_source_reference_remains_a_test_only_check[tests/fixtures/classiq_qmod/example_references/applications/finance/rainbow_options/rainbow_options_integration_method/comparisons/circuit.py]' \
  -p pytest_cov --cov=flagquantum --cov-report= \
  -p tools.pytest_stack_watchdog --fq-stack-dump-after=1 \
  -o faulthandler_timeout=0 -s
```

The local complete check passed and printed the requested snapshot. Its elapsed
time is recorded only for diagnostic provenance; the shared macOS host and
Linux CI are not interchangeable performance environments.

Black, Ruff and focused mypy passed for both new Python files. The sparse
checkout's default tier passed 79 selected unit tests after restoring unmodified
policy/chemistry fixtures omitted by sparse checkout; the failed collection and
fixture attempts remain in the evidence record. This is not full-repository
default-tier certification. Full Linux CI remains required before integration.

## CI environment isolation and subsequent failures

After synchronization with main `02bacc099cdcd1ac08510c8cf944df6e018c94c9`,
the isolated child fixture clears `PYTEST_ADDOPTS`. CPU-runtime jobs inherited
parent worker and report flags while child plugin autoload was disabled, so
argument parsing rejected `-n` and `--dist` before child tests executed. This
failure was reproduced locally with two outer workers. The slow pass and failure
scenarios now inject parent worker/report options and verify that the child does
not write the parent's report. All 11 cases then pass with two outer workers on
pytest 7.4.4 and 9.2.0. The parent CI command, worker count, reports and test
assertions remain unchanged.

Coverage job `137020948855` instead failed before pytest: removing a previous
container exceeded the existing 20-second Docker cleanup limit. A subsequent
read-only inspection on `jp-mc550-122` found the container absent and Docker
29.5.3 responsive on kernel 5.15.0-119-generic. This establishes later state,
not the timeout's root cause or a resolution of cleanup reliability. No service
or container was restarted or modified. Exact-head Linux CI remains required.

Main synchronization uses the already declared `source_commit` evidence field
inside its own record, preserving the repository revision-field vocabulary.

A Git-object walk of all 1,001 tracked JSON evidence artifacts, substituting the
changed working artifact, matches all 20 declared revision field names. This
checks vocabulary only; full provenance and exact-head CI remain separate gates.


## Integration with isolated source staging

The branch synchronizes main `d14bd25c647e7d2ec6f82b087cb32f8fe4fa4277`,
including the isolated source-pack producer fix. The unchanged watchdog and
source-artifact regression files pass together: 38 tests, including the two
real xdist workers and the full/incremental source round trips. No watchdog
implementation, diagnostic interval or scientific acceptance changes in this
synchronization.

The first local invocation found the source-artifact test missing from sparse
checkout and collected nothing. After restoring that tracked file, 37 tests
passed and the xdist child failed because this local environment lacked its
xdist import path. Supplying the existing cached xdist and execnet dependencies
produced the 38-pass run. No test was skipped or changed to mask those setup
failures. Full Linux exact-head CI is still required.
