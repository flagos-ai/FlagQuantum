# Matching graph interoperability: complete capability-bundle API proposal

Status: owner approved13 paths; implementation candidate with local validation.
The example-index14th-path amendment is separately pending approval. Fresh
installed-candidate and actual-main acceptance remain pending. This supersedes method-sized proposals for this gap.
Proposed title: feat(qec): support matrix-defined weighted matching workflows.
Classification: single_device_fast_path, primary domain QEC.

## Product boundary

One delivery covers matrix/edge input, validation, inspection, explicit boundary
and column identities, preparation, logical/correction output, failure and real
installed acceptance. This is a bounded weighted-graph capability bundle, not
complete Stim/PyMatching parity. Existing consolidated PR1087 remains separate
and pending actual-main admission. No feature branch is treated as accepted main.

The existing DecodingGraph is an independent-fault probability description. Its
weights are derived from probabilities <=0.5. General optimization graphs have
independent signed costs and optional noise probabilities. They cannot be put
into that type without changing protected meaning. New matching graph records
are justified by those verified requirements, with two actual inputs: explicit
edges and parity-check matrices. They must become the one SDK graph preparation
input internally, replacing duplicated private translation rather than creating
another unrelated registry or probability model.

## Proposed canonical public names and exact fields

No root exports or released serialization schema are added. Data lives in
flagquantum.qec.decoders.matching_graph; SDK code remains in the adapter.

    @dataclass(frozen=True)
    class MatchingEdge:
        detectors: tuple[int, ...]
        weight: float
        fault_ids: tuple[int, ...] = ()
        error_probability: float | None = None

    @dataclass(frozen=True)
    class MatchingGraph:
        num_nodes: int
        num_fault_ids: int
        edges: tuple[MatchingEdge, ...] = ()
        boundary_nodes: tuple[int, ...] = ()

    @dataclass(frozen=True)
    class MatchingSolution:
        fault_ids: tuple[int, ...]
        edge_indices: tuple[int, ...]
        weight: float
        sdk_weight: float

MatchingEdge.detectors is one node (virtual boundary) or two distinct nodes.
A two-node edge may meet an explicitly declared physical boundary node. No public
-1 sentinel is used. Fault IDs are independent of matrix column or edge index.
Weights are finite signed floats within the pinned SDK absolute limit2**24-1;
bool, nonfinite and excess magnitudes fail before SDK creation, never warn-and-drop.
Optional error_probability is independently in[0,1]; no logistic roundtrip.
Graph shape explicitly preserves isolated nodes and unused fault-ID slots.
Edges retain caller order, so edge_indices index that immutable tuple. Constructors
normalize endpoint/fault-ID ordering, but do not silently merge graph edges.
Direct edge construction refuses parallel endpoint pairs; matrix import exposes
its merge choice explicitly. Core types own Python scalars/tuples only.

MatchingSolution.weight is the sum of supplied edge costs over edge_indices,
including selected negative-cost edges. sdk_weight is the SDK's reported value;
the fields are distinct. The new record is required because the old complete
record holds probability-owned DecodingGraphEdge objects, which cannot represent
this domain. Do not change MatchingDecodeResult or any existing result field.
No path-endpoint or probability inference is hidden in the new record.

## Proposed import/export entries

    MatchingGraph.from_check_matrix(
        check_matrix: torch.Tensor,
        *,
        weights: float | torch.Tensor = 1.0,
        faults_matrix: torch.Tensor | None = None,
        error_probabilities: float | torch.Tensor | None = None,
        repetitions: int = 1,
        timelike_weights: float | torch.Tensor = 1.0,
        measurement_error_probabilities: float | torch.Tensor | None = None,
        merge_strategy: Literal[
            "disallow", "independent", "smallest-weight", "keep-original", "replace"
        ] = "disallow",
        use_virtual_boundary_node: bool = True,
    ) -> tuple[MatchingGraph, tuple[int | None, ...]]

    MatchingGraph.to_check_matrices(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

The import tuple is(graph, source_column_to_edge). Column mapping is ordinary
user data for diagnosing merges, not provenance machinery. It identifies the
final spatial edge for each source fault column, or None for an upstream-ignored
zero-support column. With repetitions, the mapping refers to the first spatial
band; subsequent spatial edge positions follow a documented round-major formula,
with time-like edges appended in detector-major order. Tests must pin this order.
Mapping must reflect replacement and independent merges instead of pretending
one original column always remains one independent edge. Do not use DemError.error_id.

H has detector/check rows and fault columns, one or two ones in an active column;
F has fault-ID rows and the same columns. Default F is the identity. Support CPU
dense, coalesced COO and canonical CSR binary bool/integer Tensors. Duplicate
sparse coordinates are rejected before arithmetic; do not silently apply XOR or
sum entries. Shape and zero-support columns survive in the explicit mapping.
More than two selected check rows is a capability refusal; do not decompose faults.
Scalar weights/probabilities broadcast; vectors must have exactly the declared
column/check width. Weights are copied as supplied, not derived from probability.
The current code-capacity CSS constructor is reused only as a reference for its
own domain and is not relabeled this import.

All five merge choices follow pinned PyMatching2.4.0 semantics. In particular,
independent merging uses its upstream weight-combination rule and original fault
label retention; preserve those potentially lossy label choices visibly in the
column mapping and documentation. Do not substitute the existing DEM merge rule
without proving it represents the same operation. The core graph contains the
post-policy edges; constructor input contains no implicit merge strategy.

The default disallow and virtual-boundary=True are deliberate safe FlagQuantum
defaults, unlike upstream's smallest-weight/physical-boundary defaults. State this
prominently; conformance calls must always pass the intended choices explicitly.
This is functional mapping, not an assertion of signature/default compatibility.

Export yields owned CPU detector-node incidence H, fault-ID incidence F and
float64 weights in graph-edge order. Boundary endpoints are encoded by the
explicit graph metadata rather than lost; reimport must be paired with the same
boundary/shape declaration through the explicit-edge constructor when required.
It is not a lossless serialization of optional probabilities or repetition origin.
Inspect graph.edges/boundary_nodes for those fields. Do not imply an exported
matrix triple alone roundtrips every graph field.

## SDK adapter and replacement

New canonical SDK class in flagquantum.qec.adapters.pymatching_graph:

    PyMatchingGraphDecoder(graph: MatchingGraph)
    .decode(detection_events: Iterable[int]) -> MatchingSolution
    .decode_batch(
        detection_events: torch.Tensor,
        *, bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]
    .decode_to_matched_detectors(
        detection_events: Iterable[int],
    ) -> tuple[tuple[int, int | None], ...]

The class stores graph with exactly the new graph type; existing
PyMatchingDecoder.graph stays DecodingGraph. No union widening or unsafe
object.__new__ production construction. No generic provider manager is added.
Private SDK construction must handle node/fault-ID shape through the pinned
public matrix constructor, not fabricate edges merely to force shape inference.
Record preparation may attach private edge-ID labels so one SDK solution carries
both fault labels and edge selection. Never combine logicals, costs and selected
edges from separate tied solutions. Batch output is logical predictions and SDK
weights, not full per-shot edge records. Inputs are CPU, layout is LSB-first when
packed, exact declared widths and zero unused padding; every output owns storage.
Dense batches have graph.num_nodes columns; declared physical boundary columns
must be zero. Sparse events must name only nonboundary nodes. Empty/zero-logical
cases must preserve signed-cost semantics: do not return an assumed zero optimum
for an empty syndrome when negative edges/cycles can contribute a solution.

Matched endpoint output keeps None for virtual or physical boundary matches.
The pinned upstream endpoint operation requires nonnegative weights, so this
operation refuses signed graphs before SDK invocation. Other graph decoding
operations must not inherit that restriction accidentally. Noncanonical packed
padding, correlated decoding and arbitrary upstream object carriers remain
explicit separate gaps. No silent decoder fallback or graph substitution.

The existing DEM adapter converts its probability graph into MatchingGraph for
one shared private SDK builder, while retaining old shape limits, exception order,
refusals, edge mapping and weight/result semantics. This must be demonstrated by
old contract tests plus replacement: the same old consumer runs with the shared
builder replaced by a controlled fake, without consumer edits or an SDK object
leaking. Do not route old methods through new signed-domain result validation.

## Thirteen-path limit

1. flagquantum/qec/decoders/matching_graph.py (new): three immutable records,
   matrix import/export, deterministic merge/mapping and validation.
2. flagquantum/qec/adapters/pymatching_graph.py (new): SDK class and one private
   shared preparation boundary.
3. flagquantum/qec/adapters/pymatching.py: delegate preparation through that
   boundary, preserving original public surface and all old refusals.
4. flagquantum/qec/decoders/README.md: authoritative domain and matrix route.
5. flagquantum/qec/adapters/README.md: SDK boundary and compatibility route.
6. tests/qec/test_matching_graph.py (new): core unit, independent enumeration,
   round/matrix/column/boundary rules; no optional SDK required.
7. tests/qec/test_pymatching_graph.py (new): marked optional SDK integration,
   all views and independent optimum/reference comparisons.
8. tests/qec/test_adapters_pymatching.py: existing-consumer replacement and
   unchanged old contract coverage; no ignore/threshold relaxation.
9. tests/unit/test_qec_installed_profiles.py: core absence, pure boundary and
   profile-source integration; preserve original skip predicates.
10. tools/qec_installed_smoke.py: real PyMatching/both wheel and sdist route,
    preserving all earlier sample/report/output cells present on actual main.
11. examples/qec/check_matrix_matching.py (new): executable ten-minute user path.
12. docs/development/QEC_MATCHING_GRAPH_API_PROPOSAL.md (new): reviewed contract,
    compatibility, limitations and golden-path commands.
13. docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md: whole-bundle evidence rows.

No __init__ exports, registry, dependency, CI workflow, runner, baseline contract
or serialization file change. If a checker demands an additional protected path,
stop and prepare a concrete amendment instead of rewriting a snapshot to pass.
Inside this one PR, commit the approved contract/fake/conformance slice before
SDK/domain implementation. This is one owner-reviewed capability delivery, not
parallel work over an unproven interface. Read ownership rules before edits.

## Planned user example (not executable in current FlagQuantum)

```python
import torch
from flagquantum.qec.decoders.matching_graph import MatchingGraph
from flagquantum.qec.adapters.pymatching_graph import PyMatchingGraphDecoder

checks = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
faults = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.uint8)
graph, columns = MatchingGraph.from_check_matrix(
    checks, faults_matrix=faults,
    weights=torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64),
    error_probabilities=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
    merge_strategy="disallow", use_virtual_boundary_node=True,
)
print(columns)
print(tuple(edge.weight for edge in graph.edges))
solution = PyMatchingGraphDecoder(graph).decode((0, 1))
print(solution.fault_ids, solution.edge_indices, solution.weight)
```

Expected:

    (0, 1, 2)
    (2.0, 3.0, 1.0)
    () (2,) 1.0

The pair edge costs1 versus two boundary edges totaling5. Its fault-ID support
is empty, so it predicts no logical flip; column2 is the independent minimum.
Probabilities0.1/0.2/0.3 do not determine these supplied costs. This is an exact
integer-cost oracle, not a stochastic or performance measurement. CPU and the
existing PyMatching extra suffice; no Stim or hardware required.

## Acceptance and stop conditions

Enumerate all edge subsets on small positive/zero/negative graphs and all small
syndromes. Check parity at nonboundary nodes and minimum total supplied cost;
compare sets of optima for ties. Include empty syndrome negative-cost solutions,
large finite representable costs, isolated nodes, zero-support columns, unused
fault IDs, duplicate policies, physical/virtual boundary equivalence, explicit
probability independence, repetitions/time-like edges, CSR/COO/dense equality,
column mapping and export/reimport with declared boundary metadata.

Compare matching preparation/decoding directly with pinned upstream matrix calls
with every policy/boundary/repetition option explicit. Compare SDK weights in
SDK precision, independent original costs in original precision, preserving all
existing comparison bounds. Add malformed input/output, alias/mutation, one-call
coherence, missing extra, atomic failure and old API/pickle/signature regression
cases. Validate native matrix/merge logic without SDK and swap the private builder
in a consumer-unchanged replacement test. No performance certification follows.

Planned commands include:

    python -m pytest -q tests/qec/test_matching_graph.py
    python -m pytest -q tests/qec/test_pymatching_graph.py tests/qec/test_adapters_pymatching.py tests/unit/test_qec_installed_profiles.py
    python examples/qec/check_matrix_matching.py
    python tools/check_dependency_policy.py

Minimum Ruff0.15, Black24.1 and strict Mypy2/Python3.12 check new production and
new test code using existing environments. Owner/architecture/language/hygiene/
docs/diff checks remain required, as do actual installed wheel/sdist evidence
before a true delivery receipt. The final PR description carries executed public
code, prerequisites, command, stdout and exact reviewed head. Required version
qualification and actual-main gates are not waived by current CI-follow pause.

First failed check preserves logs and stops, without retry, ignore, extra skip,
dependency/runner/threshold change or alternate numerical path. No research
probe, worker, full matrix, package release or dependency installation is part
of this proposal. Implementation/publication requires explicit approval of this
larger protected scope; previous output-view approvals are not sufficient.
