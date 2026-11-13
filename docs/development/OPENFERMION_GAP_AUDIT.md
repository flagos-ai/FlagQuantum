# OpenFermion workflow gap audit and delivery plan

## Decision and baseline

The selected native migration, EXT01-EXT08, molecular input/training follow-ups,
symmetry discovery and bounded graph training are delivered. Do not treat that
completion as full OpenFermion compatibility. The next recommended scientific
slice is **fermionic one-body density measurement and energy reconstruction**,
followed by two-body density measurement only after the smaller path is proven.
Do not start another model example merely to expand the migration count.

This audit examines FlagQuantum main
`b58fc8a01d9395972042abd1aaa507c9769b73da` and the already pinned OpenFermion
revision `34adf3196cd566f4ca53893f868753786797ae2a`. It makes no claim about a
newest release. The upstream pin matches
[`PROVENANCE.toml`](../../third_party/openfermion/PROVENANCE.toml).
The [migration record](OPENFERMION_NATIVE_MIGRATION.md),
[graph encoding record](FERMION_HUBBARD_GRAPH_ENCODINGS.md) and authoritative
[`capability-maturity.toml`](../../capability-maturity.toml) retain their roles.
This audit proposes work; it changes no contract, capability or acceptance level.
The independent [graph closeout PR #1177](https://github.com/flagos-ai/FlagQuantum/pull/1177)
updates the matrix and recorded baseline. Both changes start from the accepted
main above; this audit adds only this document and its source inventory, so it
does not require the closeout branch to run or merge first.

## Method and reproducible source inventory

The [source inventory](openfermion_upstream_inventory.json) records upstream
source hashes, Python file counts by package, declared runtime requirements and
344 names directly imported by the upstream root module. These are AST imports,
not 344 verified APIs, callables or independent capabilities. Dynamic, nested
and transitive exports are excluded; `resource_estimates` is inspected separately
even though it is not imported by that root module. File counts include package
initializers and do not measure migration progress. No compatibility percentage
is inferred from names, files or source lines.

Reproduce the root inventory without importing or installing upstream:

```bash
git clone https://github.com/quantumlib/OpenFermion.git /tmp/openfermion-audit
git -C /tmp/openfermion-audit checkout --detach 34adf3196cd566f4ca53893f868753786797ae2a
python - /tmp/openfermion-audit <<'PY'
import ast
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
record = json.loads(Path("docs/development/openfermion_upstream_inventory.json").read_text())
for name, digest in record["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
imports = {}
tree = ast.parse((root / "src/openfermion/__init__.py").read_text())
for node in tree.body:
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("openfermion."):
        imports.setdefault(node.module, []).extend(alias.asname or alias.name for alias in node.names)
assert imports == record["root_imports"]
counts = {}
for path in sorted((root / "src/openfermion").rglob("*.py")):
    parts = path.relative_to(root / "src/openfermion").parts
    family = parts[0] if len(parts) > 1 else "(root)"
    values = counts.setdefault(family, {"implementation_files": 0, "test_files": 0})
    values["test_files" if path.name.endswith("_test.py") else "implementation_files"] += 1
assert counts == record["python_files_by_directory"]
requirements = (root / "dev_tools/requirements/deps/runtime.txt").read_text().splitlines()
assert [line for line in requirements if line.strip() and not line.startswith("#")] == record["runtime_requirement_lines"]
print(sum(map(len, imports.values())), "direct root import names verified")
PY
```

The executable verification above requires network only for clone. The audit
itself used a bounded shallow fetch of the same commit. Relative upstream paths
below start at `src/openfermion/`; native paths start at `flagquantum/`.
The classification is conservative: existing native behavior is inspected and
linked to representative evidence, while unverified upstream equivalence stays
a gap. Neither this document nor the AST inventory reruns upstream's test suite.

## Workflow coverage and remaining boundaries

| Family and upstream source | Native authority and evidence | Status at the audited baseline | Remaining requirement and owner |
| --- | --- | --- | --- |
| Fermion/Pauli algebra: `ops/operators`, `transforms/opconversions/term_reordering.py` | `observables/fermion.py`, `core/algebra/pauli.py`; `tests/unit/fermion/test_encoding.py`, native boundary tests | Selected canonical static algebra is delivered; upstream storage and signatures are not retained | Majorana, Ising, quadrature and general PolynomialTensor representations are not certified equivalents. Core owns domain values; do not add parallel upstream-shaped types merely to mirror exports. |
| JW, parity, BK: `transforms/opconversions` | `algorithms/fermion/encoding.py`; encoding tests and three-encoding H2/graph workflows | Delivered for explicit full register widths and static coefficients | BinaryCode/nonlinear decoders, segmented/checksum codes, BKSF and Verstraete-Cirac are not delivered by the three encodings. Algorithms owns transforms; require a workload that existing encodings cannot satisfy before expanding. |
| Molecular data and Hamiltonian access: `chem/molecular_data.py` | `algorithms/chemistry_domain`, `ecosystem/chemistry`; FCIDUMP, HDF5, active-space and cached-object tests | Static real closed-shell molecular input is delivered; external arrays stop at adapter boundaries | Complex/open-shell integrals, geometry/SCF derivatives, general MolecularData persistence and cached correlated results remain separate gaps. Ecosystem owns foreign/file input, Algorithms owns validated science. |
| Model Hamiltonians: `hamiltonians/hubbard.py`, `general_hubbard.py`, `jellium.py`, `plane_wave_hamiltonian.py` | `algorithms/fermion/hubbard.py`; chain/ring/spinless/graph tests | Selected chains/rings and uniform spinful simple graphs delivered, with explicit signs and static parameters | Site/bond-dependent terms, flux, multiband models, jellium, plane waves and d-wave models are not covered. Algorithms owns symbolic assembly; no graph framework or solver belongs here. |
| Gaussian preparation: `ops/representations/quadratic_hamiltonian.py`, `circuits/primitives/state_preparation.py` | `algorithms/fermion/preparation.py`, `decomposition.py`, `lowering.py`; preparation tests | Static CPU Slater states and Gaussian vacua delivered | Full excited Gaussian basis and general quadratic-Hamiltonian workflows are not certified. Algorithms owns mathematical plans and circuit construction; Simulation owns execution. |
| Fourier/orbital transforms: `circuits/primitives/ffft.py`, `transforms/repconversions/fourier_transforms.py` | `algorithms/fermion/fourier.py`, `lowering.py`; Fourier tests | Power-of-two adjacent circuit synthesis delivered | General widths and upstream grid/plane-wave operator transformations remain gaps; existing O(n^2) factors do not establish FFT acceleration. Algorithms owns the transform. |
| Symmetry/tapering: `transforms/repconversions/operator_tapering.py`, `qubit_tapering_from_stabilizer.py` | `algorithms/fermion/symmetry.py`, `stabilizer.py`; discovery/reduction tests | Explicit eigenvalue reduction, diagonal/general commuting discovery and bounded selected-sector examples delivered | Automatic sector choice, general matched ansatz transformation and tapering an arbitrary training circuit remain gaps. Discovery must not infer ground sectors from particle parity. Algorithms owns reduction; protected circuit/IR changes require an Integration proposal first. |
| Excitation training: `circuits/unitary_cc.py`, `circuits/primitives/bogoliubov_transform.py` | `algorithms/chemistry.py`, existing native circuit/autograd; UCCSD tests and training examples | Existing native excitation circuits are reused; H2/graph basis replacement is manually specified | General encoded/tapered UCCSD and arbitrary Bogoliubov circuit transformation are not provided by the fixed-pair examples. Reuse excitation algebra instead of adding a fermion optimizer. |
| Product formulas: `circuits/trotter`, `circuits/low_rank.py` | `algorithms/fermion/low_rank.py`, `evolution.py`, existing `algorithms/trotter.py`; low-rank/controlled evolution tests | Selected real spatial low-rank order-1/2 evolution and scalar-time gradients delivered; generic Pauli formulas already exist | Broader split-operator families, analytic chemistry-specific error estimators and upstream numerical convention equivalence remain unqualified. Reuse the existing product-formula boundary. |
| Sparse and sector numerics: `linalg/sparse_tools.py`, `davidson.py` | `algorithms/fermion/numerics.py` provides bounded dense inspection; native runtimes execute circuits | Dense reference inspection is available; no upstream sparse-solver replacement certified | Number/spin-preserving sparse matrices, Davidson and classical sector ground-state solvers are not provided by parity reduction. Algorithms may own bounded reference routines; never bypass Runtime or add a fourth simulation core. |
| Fermionic density observables: `ops/representations/interaction_rdm.py`, `measurements/get_interaction_rdm.py` | Existing `create`, `annihilate`, encoding and expectation already express density monomials; no dedicated measured fermionic RDM workflow located | Primitives exist, but a reusable 1-/2-body density and energy reconstruction path is missing | Algorithms should compose native observables and tensors. `simulation/density_matrix.py` is a qubit subsystem partial trace, not a fermionic one-/two-body RDM replacement. Preserve this distinction. |
| Measurement constraints/reconstruction: `measurements/rdm_equality_constraints.py`, `equality_constraint_projection.py`, `linalg/rdm_reconstruction.py` | Native expectation is authoritative for execution; no matching fermionic constraint/reconstruction workflow located | Not delivered in the selected migration | N-representability constraints, Valdemoro reconstruction and noise projection need their own approximate/failure semantics. Algorithms owns postprocessing; do not silently modify raw measurements. |
| Bosons/quadratures: `ops/operators/boson_operator.py`, `quad_operator.py`, `hamiltonians/hubbard.py` | Existing `observables/boson.py`; boson algebra, truncation and Lindblad consumer tests | Native bosonic algebra and explicitly truncated matrices already exist independently of this migration | Upstream BosonOperator/QuadOperator interop, Bose-Hubbard and phase-space transformations are not certified. Reuse native values and established dynamics; do not claim bosonic support is absent or create duplicate algebra. |
| LCU/resource estimates: `circuits/lcu_util.py`, `resource_estimates` | Existing `algorithms/primitives/linear_combination.py`, `block_encoding.py`, `compiler/resource_estimation.py`, `algorithms/logical_resources.py`; their unit tests | Native LCU/walk and static/logical accounting already exist; no chemistry-resource parity certified | Single/double factorization, THC, periodic resource tables and assumptions require workload-specific evidence. Compiler owns program accounting; Algorithms owns chemistry estimates. No second generic block-encoding framework. |
| Utility, functionals, contrib, testing: corresponding upstream package initializers | Existing native utilities/test oracles and module-local validation | No blanket equivalence claim | Contextuality/norm functionals, lattice helpers, projection utilities, random fixtures and legacy convenience exports are not an automatic migration backlog. Reuse only when a verified workflow requires them; upstream test helpers remain test-only. |
| Foreign interoperability: upstream operator/data types and Cirq circuits | `ecosystem/openfermion.py`, `ecosystem/chemistry/molecular_data.py`; conversion/replacement tests | Fermion/Qubit operator import/export, InteractionOperator import and cached MolecularData import delivered narrowly | No general circuit import/export, reverse integral tensor export or complete type compatibility. Foreign classes stay in Ecosystem; native workflows do not depend on OpenFermion or Cirq. |

## Dependency decision

The pinned upstream runtime requirements include Cirq, SciPy, SymPy, NetworkX,
NumPy, h5py, requests, PubChemPy and deprecation; the exact requirement lines are
in the inventory. They are upstream packaging facts, not requested FlagQuantum
dependencies. No upstream package was installed for this audit.

Native algebra, encoding, preparation and circuit training must remain runnable
without OpenFermion/Cirq. Optional chemistry/file/foreign adapters keep their
existing boundaries. An upstream algorithm that imports Cirq is a mathematical
reference; emit existing FlagQuantum gates/IR instead of translating upstream
gate classes into core. Native BosonOperator, block encodings, product formulas,
resource accounting and qubit density matrices must be inspected before any
proposal adds a corresponding abstraction. Existing types are not proof that
every specialized chemistry workflow is already solved.

## Proposed next delivery sequence

This is a proposed sequence, not authorization to implement or change APIs.
Each implementation starts from main after its prerequisites land. Each row
must deliver one complete input/validation/execution/result/failure/evidence
path. Do not stack implementation PRs against an unmerged custom contract.

| Step | Delivery and domain | Acceptance and exit condition |
| --- | --- | --- |
| A: this audit and graph closeout | Integration documentation. Closeout records #1176; this independent audit pins upstream, distinguishes reused capabilities from gaps and proposes priorities. | Source inventory hashes/root imports reproduce, local links resolve and existing docs/capability checks pass. Neither PR changes numerical code or maturity. |
| B: one-body density measurement scope | Algorithms design first. Specify gamma[p,q]=<a-dagger[p] a[q]>, register/encoding ordering, output dtype/shape, complex handling and static/differentiable boundary using existing tensors and observables. | Show why existing values suffice; choose one real CPU fixed-particle workflow and an explicit dense-reference budget. Review whether any protected boundary changes; if so land a separate approved Integration contract/fake/conformance PR before implementation. No new RDM class or registry is presumed necessary. |
| C: one-body density vertical slice | Algorithms implementation using existing ladder algebra, encoder and native expectation. Separate real/imaginary Hermitian parts of off-diagonal monomials when needed. Runtime remains unmodified. | Vacuum, occupied determinant, complex coherent one-particle state and correlated fixed-particle state agree with independent occupation-bit action; Hermiticity, trace N and orbital-basis changes hold. Reconstruct a one-body Hamiltonian's energy. Replace a direct occupation oracle with encoded JW/parity/BK measurement without changing its consumer. Reject mismatched width/encoding and unsupported inputs before execution. Include a short CPU example and honest memory/term-count limits. |
| D: one-body registration and docs | Integration, after C's scientific acceptance and actual-main verification. | Register only the executed scope. Constructor/measurement parameter gradients must be stated separately. Preserve fail-closed hardware/precision/encoding limits and link exact evidence. |
| E: two-body density and molecular energy | Algorithms, only after C/D. Specify Gamma[p,q,r,s]=<a-dagger[p] a-dagger[q] a[r] a[s]> and the Hamiltonian coefficient convention before implementation. Reuse static molecular coefficients and measured tensors. | Independent signs/permutations, antisymmetry, Hermiticity, trace N(N-1), one-body contraction with the declared index convention, and energy reconstruction match direct expectations on a bounded molecule. Test zero/one-electron failures/degeneracies explicitly. Publish O(n^4) output/measurement costs; do not infer scalable chemistry. Separate later registration. |

The sequence B-E is recommended because it adds a scientific result users can
consume and builds on existing authoritative algebra/measurement boundaries.
It avoids making general circuit transformation or a new solver a prerequisite.
The measured-density path must not allocate a full-state dense oracle in normal
execution; independent dense references belong to bounded tests only.

## Deferred branches and admission criteria

- **General encoded/tapered ansatz:** valuable for extending UCCSD, but the
  full circuit legality, basis transformation and gradient semantics must be
  designed first. Fixed determinant examples do not establish a general
  transform. Admit a minimal full-register encoding replacement before sector
  tapering, without changing consumer training code.
- **Weighted/complex Hubbard:** admit only with a concrete lattice/flux use
  case. Keep edge validation and symbolic assembly in the model module; verify
  Hermiticity, orientation conjugation and gauge-invariant loop spectra
  independently. Do not broaden the current constructor without API review.
- **Quadratic excited Gaussian states:** requires a declared occupation/vacuum
  convention and independent covariance/phase evidence. Reuse preparation and
  decomposition rather than add a separate execution layer.
- **Sparse reference solver:** requires an explicit size budget and a current
  scientific need beyond small dense tests. It must not become a backend or
  bypass one of the three owned numerical cores.
- **RDM constraint projection, bosonic interop, resource-estimate families and
  additional encodings:** remain separate proposals. Their downstream need,
  dependency review and replacement/conformance evidence must precede code.

## Review and verification boundary

No implementation breadth is earned merely by matching a root import list.
For every admitted slice, name the primary domain, existing types reused,
protected boundaries, normal execution cost, expected errors, independent oracle
and ten-minute contributor path. A new dependency or contract needs its own
justification; ordinary features should not require edits across four domains.

Existing executable user workflow supported by this audit:

```bash
python -m examples.algorithms.fermion_hubbard_graph --encoding parity
python -m examples.algorithms.fermion_hubbard_graph --encoding bravyi_kitaev
```

At the audited baseline both optimize the same fixed pair from energy
0.0339385184 to -1.5997840500 with spin populations 1/1 and gradient residual
below 1e-9. Their completion is evidence for reuse, not implementation of the
proposed density APIs. The [graph acceptance record](FERMION_HUBBARD_GRAPH_ENCODINGS.md)
records the already executed science and exact-main verification. New proposed
functions are intentionally not presented as runnable APIs.

## Audit validation

The embedded reproduction script passed against the detached upstream pin:
all 15 recorded source hashes, direct root imports, per-directory Python file
counts and runtime requirement lines matched. All 21 explicitly named upstream
source paths in the coverage table resolve at that pin. The audit branch passed
37 existing capability/documentation tests, generated documentation consistency,
the 1,424-file local-link/anchor audit, Integration scope and whitespace checks.
These checks validate the audit artifact and existing documentation boundaries;
they do not execute or certify any of the proposed future density workflows.
