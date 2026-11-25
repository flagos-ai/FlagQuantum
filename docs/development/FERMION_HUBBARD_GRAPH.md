# Spinful Hubbard model on an explicit graph

## Scope before implementation

This proposed Algorithms-owned extension starts from accepted main
`4b1fbd7edcb2f24f00a74ff5ab1e921b1bcf89cb`. The initial native migration and
H2 encoding replacement are complete; this is a separate model extension.

Add the experimental concrete function `fermi_hubbard_graph(n_sites, edges,
*, t=1.0, u=0.0, mu=0.0)` in the existing `algorithms.fermion.hubbard` module.
It returns the existing `FermionOperator`. Sites retain interleaved up/down
modes, uniform minus-t hopping, unshifted onsite U and minus-mu population.
`edges` is a finite sequence of endpoint pairs describing an undirected simple
graph. Each edge is counted once; either orientation and any order give the same
operator. Refuse self edges, duplicate/reversed edges, out-of-range endpoints,
noninteger endpoints and malformed pairs. Empty graphs and isolated sites are
valid. Coefficients remain finite static real numbers.

Model assembly owns no graph library, solver, encoding, circuit preparation or
runtime policy. Do not change the existing chain/ring signatures or validation,
Stable Core exports, dependency manifests, CI or capability registry. Reuse their
existing authoritative algebra and scalar/count validation. A later Integration
change may register the new experimental boundary after implementation evidence;
existing chain/ring capability entries must not claim arbitrary graphs.

## Acceptance before implementation

- Compare independent occupation-bit matrices on branching and disconnected
  graphs, including isolated sites, Hermiticity and separate spin conservation.
- Replace chain/ring producers with explicit path/cycle edge lists without
  changing their encoding and measurement consumers. Check all three encodings.
- Check edge orientation/order invariance, zero-operator width retention, empty graphs,
  onsite terms and failures before encoding or execution.
- Use an explicit spectator on a non-neighbor hop to check the fermionic sign
  independently; ordinary native expectations, angle gradients and an SGD
  update must match an analytic two-determinant reference.
- Supply a small executable CPU example with no dense eigenstate preparation
  or external science SDK; include its CLI in the existing example inventory.

This is bounded CPU `single_device_fast_path` correctness, not a performance,
accelerator, distributed or scalability claim. Weighted/complex edges, flux,
spin-dependent coefficients, trainable Hamiltonian parameters and general
encoded ansatz transformation remain outside this scope.

## Implemented contributor path

Scope commit `d72636c7a` preceded implementation. The review branch adds only
the concrete model function, independent scientific tests and an executable
example in existing Algorithms-owned paths. It introduces no additional model
representation or public root export. Existing chain/ring implementations are
reused rather than replaced, so their validation remains authoritative.

```bash
python -m examples.algorithms.fermion_hubbard_graph
python -m pytest tests/unit/fermion/test_hubbard_graph.py tests/integration/test_fermion_hubbard_graph.py -q
```

The four-site star has edges (0,1), (0,2), (0,3), t=1.2, U=4 and mu=0.2.
Its fixed pair is cos(theta/2)|00010010> + sin(theta/2)|10010000> in
interleaved occupation order. The down-spin spectator at mode 3 reverses the
mode-0/mode-6 hopping sign, giving E(theta)=1.2*sin(theta)-0.4 and derivative
1.2*cos(theta). Onsite U vanishes on this pair; independent complete occupation
matrices test U on all states, including double occupancy. The pair minimum
-1.6 is not the graph's ground energy. Forty SGD updates from theta=0.37 lower
the energy below -1.59 and retain one particle of each spin. No dense eigenstate
is used; fresh-process tests block external science SDKs and dense eigensolvers.

To change a hopping convention, first run the unit tests. Branching and
disconnected graphs are compared against direct occupation-bit action, then
against JW/parity/BK using independent bit permutations. Path/cycle graphs
replace existing constructors with the same training consumer. To change the
example, rerun the integration and CLI inventory tests. Invalid inputs must
remain errors even when a coefficient makes their contribution zero.

## Local review evidence

The review branch passed 266 related Hubbard/spinless scientific cases,
including 53 graph cases, and 11 selected CLI/inventory cases. The 53 graph
cases passed again after final orientation/order and zero-hopping rejection
assertions and the test tuple annotation were added. Strict mypy on the model,
example and two scientific test modules, Ruff, Black 24.1 and 26.5, Stable Core
API baseline, capability validation, documentation consistency, the 1,422-file
local-link audit, Algorithms ownership and whitespace checks passed.

The recorded example energy is 0.0339385184 initially and -1.5997840500 after
40 updates; finite-difference gradient residual is 2.451e-11 and both spin
populations are 1.000000. Last digits may vary by platform. These focused local
checks are not full-suite or actual-main acceptance. Broad Linux checks remain
assigned to PR CI, and no new capability maturity is registered by this branch.


## Merged implementation acceptance

Implementation [#1174](https://github.com/flagos-ai/FlagQuantum/pull/1174) merged
as `a8fc1fc5494f68a262cb24a293e00c5c53a9874a`. Its tree
`ebdfdc592954aa08e022735e143cd64265231b79` is identical to reviewed head
`b223f5e98240382da34f0c7d341e4dba804efd60`. PR CI run
[43375308679](https://github.com/flagos-ai/FlagQuantum/actions/runs/43375308679)
passed on attempt 2: 239 checks succeeded and three conditional checks skipped.
Attempt 1 failed while downloading the source artifact with ECONNRESET before
one coverage partition could execute; the failed partition and downstream
coverage/final gates passed on rerun without source changes.

The exact merged main checkout independently passed 54 graph unit, integration
and CLI cases. Its
[Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43383069790)
also passed. The full PR CI result applies to the identical merged tree; no
second full-main CI run is claimed. These records qualify this revision's
bounded CPU behavior only.

A separate Integration-owned registration now proposes an `experimental`
`fermion_hubbard_graph` entry, generated catalog and known limitations. It does
not promote chain/ring entries or make the model coefficients differentiable.
The constructor has no gradient support; ordinary native execution in the
example differentiates its circuit angle.
