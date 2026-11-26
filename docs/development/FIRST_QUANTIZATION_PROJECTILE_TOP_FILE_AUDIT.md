# Projectile first-quantization tutorial and top-level composition audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization/projectile`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0374 | `__init__.py` | `47a2b869976eab2cea2c4e62df7308c509f0fd11` |
| Q-0381 | `projectile.ipynb` | `3184e60fe47204d63a5851ebdd2d674b8c65303d` |
| Q-0382 | `select_and_prepare.py` | `dde8e7eb6c1802cf71d67a3338d5cc4214e6e963` |
| Q-0383 | `select_and_prepare_test.py` | `078897c69fdf25ac6aa0ebecee673819bf5d7252` |

The native 18-cell tutorial covers both original examples with their
ordered, typed port signatures and four diagrams. The SELECT graph
retains all widths and multiplicities of the allocation and release
nodes. Those nodes carry no physical gate charge. Both frozen and
native notebooks ran in independent kernels. The other original
test modes gave eight passes and four skips.

Q-0374 accounts for the two original package re-exports and the complete
introduction. The
[native tutorial](../../examples/notebooks/register_first_quantization_projectile.ipynb)
retains the source introduction verbatim, including the three projectile
terms, separate momentum grid, and Gaussian wavepacket assumption. It
constructs `ProjectilePrepareDeclaration` and
`ProjectileSelectDeclaration` from the native resource module. Tests
verify the source re-export identities and grouped ports and resource
models for the tutorial parameters and a distinct precision setup. The
source PREPARE declares no L1 norm. The corresponding native normalization
query is unimplemented.

## Register contracts and resources

The native top-level register owner preserves the four Hamiltonian
flag elements, the electron and projectile arrays, and independent
bounded index domains. PREPARE and SELECT declare different widths
for both coordinate selectors. PREPARE's separate term-selection leaf
operates on nine through wires. The top-level four PREPARE stages
preserve `plus_t`, bind the three ν axes in order, and connect
`succ_nu` to the nested success port.

The original PREPARE adjoint has the expected outer reversed
schedule, but its recursive width query fails at the μ flag in
nested ν. The native adjoint estimate composes the corrected UV
model and retains the original gate categories. The four call names,
order and bystanders are checked against the outer source adjoint.

| Tutorial model at p=6, n=8, eta=10 | Logical accounting | Peak |
| --- | --- | --- |
| PREPARE forward | 733 Toffoli, 25 And, 25 measurements | 104 |
| SELECT forward | 707 Toffoli, 768 CSwap, 76 And, 76 measurements | 345 |
| PREPARE corrected native adjoint | 102 Toffoli | 89, native estimate |

The dedicated controlled exchange remains executable through
`circuit.append(op)`. Its independent two-particle fixture emits
46 T/T† and one measurement. Earlier full-amplitude and workspace
comparisons cover its small measurement branches and noncontiguous
operands. The locked source's `cvs=0` display disagrees with its
execution polarity. The source-compatible path preserves its
observed execution.

The native `control_values` argument activates on an explicit bit pattern
in the bound control order. With the argument omitted, the operation
retains the locked all-one activation. Each zero bit adds one X before
the positive-control traversal and one after it. The logical model
charges these Clifford gates separately from the source model. The
traversal, controlled swaps, measurement feedback and scratch allocation
are identical to the source-compatible path. The source child graph
comparison covers both directions and noncontiguous wiring. The
independent permutation checks `(0,)`, `(0, 1)`, `(1, 0)` and `(0, 0)`
on coherent inputs across both addresses and every control word, with
all small measurement histories, tail preservation, and scratch returned
to zero. These checks distinguish the native polarity extension from the
locked source behavior.

End-to-end physical leaves and the locked recursive adjoint failure
remain open under Q-0382. Q-0381 and
Q-0383 close the complete tutorial and original test coverage.

The subsequent [constructor dependency audit](RESOLVER_PROJECTILE_ROOT_AUDIT.md)
binds the complete finite root file and its direct identifiers to typed Programs.
It retains PREPARE's four calls, SELECT's 26 allocation/child/free calls and the
source-exact reshape wrappers. All-one activation despite cvs labels and the
source empty-control conversion failure remain explicit. Whole-root primitive
execution is still unavailable because its opaque leaves have no source body.
