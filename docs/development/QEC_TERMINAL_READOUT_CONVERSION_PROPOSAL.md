# Terminal readout from a native Circuit to QEC events

Status: approved API contract and implementation candidate.
Proposed title: `feat(qec): compile terminal readout to detection events`.
Primary domain QEC, with Simulation owning the numerical boundary.
Classification: `single_device_fast_path`.

## User outcome and admission prerequisite

Use the same `fq.Circuit` for exact local simulation and for interpreting
externally recorded terminal Z-basis measurements. No new public circuit type.
Existing `fq.Circuit`, `Parameter`, `bind_parameters`, `fq.compile`,
`fq.run`, `MeasurementMap`, and the compiler's `Layout` are reused.

#1096 was merged from reviewed head
`986a171e0de8c2879d4f6db64addec173a234ca0` into main at
`7cddcfec838cfdc95f7a69ff42e8c77a6a608812`. Its complete tree equals
the reviewed tree. Actual-main CI, smoke, installed QEC packaging, and local
isolation passed. The main GPU workflow stopped at HTTPS source checkout
before GPU tests started. The user directed that this outstanding GPU
acceptance does not gate the next proposal. The original failure is retained
in the project delivery evidence, and the prior recovery proposal is closed
without a request. Cut the implementation branch from then-current main and
recheck the conversion and layout contracts before editing.


## Exact proposed API

In `flagquantum.qec.execution.measurement_conversion`:

```python
def compile_m2d_converter(
    program: Circuit | CircuitIR,
    *,
    measurement_qubits: tuple[int, ...],
    detector_map: MeasurementMap,
    observable_map: MeasurementMap,
) -> CompiledMeasurementsToDetectionEventsConverter: ...

class CompiledMeasurementsToDetectionEventsConverter:
    @property
    def measurement_qubits(self) -> tuple[int, ...]: ...

    def convert(
        self,
        *,
        measurements: torch.Tensor,
        separate_observables: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

The caller supplies logical qubits in the order their records use. Each
entry identifies one terminal Z measurement. Maps index that ordered
record, so column zero need not be qubit zero. Compile snapshots the program
and maps and prepares one reference and converter. Each convert accepts a
two-dimensional CPU bool tensor and returns owned CPU bool detector and
logical tensors in map order. Zero shots retain output widths. At least
one detector is required and zero logical observables are allowed. The
public handle is compiler-created and nonserializable. There is no public
constructor taking an SDK object. Add only the two module opt-in exports.

This slice covers terminal readout, dense Torch records and separate outputs.
Text conversion stays available through the #1096 entry. A prepared Stim
text/NumPy/file/packed interface, native mid-circuit measurement, feedback,
noise, remote execution, and DynamicCircuit builder convergence remain
separate proposals. No new root exports, dependency, CI workflow or
serialization contract is proposed.

## Reuse and execution boundary

- Share the #1096 validation and result ownership rules, bounded Stim parser,
  noiseless deterministic-reference check and converter preparation. Extract
  a private prepared conversion helper inside Simulation and have the #1096
  one-shot path and the new prepared path use it. Keep SDK objects private
  to Simulation. Reuse existing annotation emission and terminal-measurement
  order, rather than implementing another Clifford simulator.
- Convert a supported, bound, static native program to a private Clifford
  execution view. First admission is the existing exact Clifford lowering
  plus the representative binary64 `RX(math.pi)` case, equivalent to X up
  to global phase for this measurement task. Admit only the exact supported
  constant and a static scalar. Do not snap a nearby angle to it or alter
  the caller's IR or the ordinary compilation contract. Unbound parameters
  and unsupported bound rotations refuse.
- For an original or ordinary optimized IR, use logical wire labels. For
  a supported compiler `apply_layout` artifact, validate its complete
  logical/physical assignment and inverse using the existing Layout rules,
  then translate the logical readout tuple once. Keep raw columns in the
  caller's logical order. The test must inspect the actual emitted
  measurement sequence and use a distinguishable reference, in addition
  to checking the public logical tuple. This slice makes no claim about
  metadata authenticity after caller tampering or stripped layout evidence.
  Refuse routing artifacts, mixed/unknown layout conventions and conflicting
  output measurement declarations until their contracts are admitted.
- Reject duplicate/out-of-range readouts, mismatched declared record widths,
  malformed record dtype/device/rank/width, existing mid-circuit measurement
  and feedback, unsupported channels, and nondeterministic declared parities.
  A random raw bit is permitted when the declared parity is deterministic.
  Report an actionable reason and retain the original error cause.
- The existing engine is 1,244 lines against a 1,250-line ceiling. Move
  conversion-specific preparation and buffer validation into the private
  helper where needed. Verify the ceiling without changing it. Keep the
  existing three-function Stim bridge and #1096 dense signature intact.

## User example and independent acceptance

CPU, Python 3.12, Torch and the existing Stim optional extra. The ordinary
simulation below uses existing API. The converter block becomes runnable
after implementation. Run the final example from outside the checkout at
the reviewed head and retain stdout and environment evidence.

```python
import math

import flagquantum as fq
import torch
from flagquantum.compiler import Layout, apply_layout
from flagquantum.qec.execution.context import MeasurementMap
from flagquantum.qec.execution.measurement_conversion import compile_m2d_converter

theta = fq.Parameter("theta")
circuit = (
    fq.Circuit(3, dtype=torch.complex128)
    .h(0).cx(0, 1)
    .rx(2, theta=theta)
    .x(1).x(1)
)

bound = circuit.bind_parameters({theta: math.pi})
compiled = fq.compile(bound)
layout = Layout((2, 0, 1))
placed = apply_layout(compiled, layout)
options = fq.ExecutionOptions(mode="auto", device="cpu", allow_approximate=False)

def probabilities(program):
    return fq.run(program, options=options, outputs=fq.probabilities()).probabilities

def z(program, qubit):
    return fq.run(
        program, options=options, outputs=fq.expectation(fq.Z(qubit))
    ).expectation().item()

torch.testing.assert_close(
    probabilities(bound), probabilities(compiled), atol=1e-12, rtol=0
)
assert math.isclose(z(bound, 2), -1, abs_tol=1e-12, rel_tol=0)
assert math.isclose(z(compiled, 2), -1, abs_tol=1e-12, rel_tol=0)
assert math.isclose(z(placed, 1), -1, abs_tol=1e-12, rel_tol=0)
assert math.isclose(
    z(circuit.bind_parameters({theta: 0.0}), 2), 1, abs_tol=1e-12, rel_tol=0
)

readout = (2, 0, 1)
detector_map = MeasurementMap(3, ((1, 2),))
observable_map = MeasurementMap(3, ((0,),))
raw = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)

for program in (bound, compiled, placed):
    converter = compile_m2d_converter(
        program,
        measurement_qubits=readout,
        detector_map=detector_map,
        observable_map=observable_map,
    )
    assert converter.measurement_qubits == readout
    dets, obs = converter.convert(measurements=raw, separate_observables=True)
    assert dets.tolist() == [[False], [True], [False]]
    assert obs.tolist() == [[False], [False], [True]]

print("binding, terminal readout and QEC events agree")
```

Independent oracle: D0 is column 1 XOR column 2. L0 is column 0 XOR 1.
The second record flips D0 alone. The third flips L0 alone. The three
programs must match all eight fixed 3-bit rows against this oracle.
Separately assert logical (2,0,1) emits physical (1,2,0) after placement.
Use a negative witness that drops or misapplies the placement mapping and
shows the reference/column check catches the failure. An identity mapping
echo and equality between routes sharing one SDK are insufficient evidence.

Read-only preflight already verified that the existing compile call reduces
five instructions to three, retains the binary64 pi parameter, and maps
logical readout (2,0,1) to physical (1,2,0) through Layout((2,0,1)).
The placed program yields Z(1) = -0.9999999999999998. Two independently
authored fixed Stim texts for the original and placed variants both matched
the three expected detector/logical records through the current #1096 API.
This proves the fixture and the independent reference, not the proposed
native conversion implementation or its full eight-row acceptance.

## Additional actual-main fixture preflight

On the archived, verified `7cddcfec` tree, the existing ordinary simulation
API reproduced the parameter-binding and placement invariants. The existing
record converter matched the independent XOR oracle for all eight input rows
using both independently written Stim texts. A deliberately misordered
physical measurement sequence was rejected for nondeterministic declared
parities. The proposed native converter and its prepared-state reuse still
require implementation and acceptance. The preflight command and output are
retained beside this proposal.

## Exact path scope

1. `flagquantum/qec/execution/measurement_conversion.py`
2. `flagquantum/qec/execution/README.md`
3. `flagquantum/simulation/stabilizer/engine.py`
4. `flagquantum/simulation/stabilizer/_record_conversion.py` (new private helper)
5. `flagquantum/simulation/stabilizer/README.md`
6. `contracts/qec-terminal-measurement-converter-v1.json` (new)
7. `docs/development/QEC_TERMINAL_READOUT_CONVERSION_PROPOSAL.md` (new)
8. `docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md`
9. `tests/qec/test_terminal_measurement_conversion.py` (new SDK-free contract and replacement fake)
10. `tests/team/simulation/test_terminal_readout_conversion_conformance.py` (new)
11. `tests/team/simulation/test_stim_record_conversion_conformance.py` (retain old cases, adapt private helper import if moved)
12. `tests/unit/test_qec_installed_profiles.py`
13. `tools/qec_installed_smoke.py`
14. `examples/qec/terminal_measurement_conversion.py` (new)
15. `examples/README.md`
16. `tests/api_contract/test_public_docstring_examples.py`

Register the proposed public converter example in the existing doctest list.
The #1096 follow-up established that a module being covered does not itself
execute a second entry's example. Validate parsing and the guarded example
with Stim and in the core-only environment. Keep the #1096 entry intact.

Commit the approved API contract and replacement/conformance tests first in
the same PR, then implement. The private fake supplies the same prepared
handle to the unchanged consumer. Verify that the second batch reuses the
same preparation and that invalid input never produces partial results.
Use the existing packaging hook and do not duplicate the #1096 baseline.
No workflow changes, frozen fixture changes or hardware runs.

## Verification commands

```bash
python tools/check_team_scope.py --team integration --files <the-16-listed-paths>
python -m pytest -q \
  tests/qec/test_terminal_measurement_conversion.py \
  tests/team/simulation/test_terminal_readout_conversion_conformance.py \
  tests/qec/test_stim_measurement_conversion.py \
  tests/team/simulation/test_stim_record_conversion_conformance.py \
  tests/qec/test_stim_experiment_bridge.py \
  tests/unit/test_qec_installed_profiles.py \
  tests/api_contract/test_public_docstring_examples.py \
  tests/test_algorithm_examples.py::test_examples_readme_names_every_example_that_does_not_use_the_root_alias
python -m examples.qec.terminal_measurement_conversion
python tools/check_architecture.py
python tools/check_dependency_policy.py
python tools/check_docs_links.py
python tools/docs_source_of_truth.py --check
python tools/check_qec_installed_profiles.py --dist-dir dist --report-dir /tmp/qec-installed-reports
```

Use Black, minimum Ruff and strict Mypy on changed Python paths. Also run the
SDK-free fake and core profile without optional SDKs, and confirm marker
reachability through existing CI selection. Existing real wheel/sdist
acceptance is the installed gate when implementation is approved. Preserve
the first failure and stop. No CI waiting, recovery or research is part of
this proposal review.

The acceptance advances FTS-04 on the bounded terminal measurement slice.
General dynamic feedback, full Stim parity, fault-tolerant compiler selection
and milestone closure need separate evidence.
