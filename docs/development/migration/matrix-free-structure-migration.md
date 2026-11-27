# Matrix-free numerical structure migration

The four existing `simulation/matrix_free_*.py` implementations belong to
`simulation/numerics/matrix_free/`. Their algorithms and defaults remain
unchanged. Whole-file relocation entries preserve historical evidence
identifiers while operational source imports use the actual owners.

The complete audited #1084 tree is `11fd9fa27ec31e74f03277010762e8d25d17c105`.
None of these four sources is modified in its base-to-head diff, and the
verified incoming changed-source texts contain no reference to their names.
Repeat the target/consumer check if that input advances before integration.

## Required serialized-plan boundary

The installed pre-move wheel constructs a public LindbladPlan from a Pauli
Hamiltonian descriptor. Its 3,431-byte pickle contains the original numerical
module name and both concrete types. The fixture records those actual bytes
and the SHA-256 of the source file, which matches the checked-out pre-move source.
After relocation the fixture restores and executes through the public run
interface with the same expectation values. Therefore the old Hamiltonian
module retains exactly PauliSum and PauliSumTerm as serialization locators.
There is no generic alias, reverse owner import, numerical reexport or private
compatibility fallback. Type modules and field annotations stay unchanged.

The other three old files are deleted. Numerical consumers and tests use the
canonical owners. The retained two-type boundary is itemized in the existing
public compatibility inventory; removing it requires an approved public-plan
serialization migration, not an arbitrary internal cleanup date.

## Preview validation

The historical-plan execution, retired-import refusals, matrix-free Hamiltonian
behavior, batching/gradients and integrator suites pass: 143 tests. Relocation
and retired-import checks report no errors. The original numerical assertions
are unchanged. Further branch checks and real CI are required before delivery;
this source preview is not a completed PR or a performance qualification.

## Incoming head recheck

The subsequent #1084 increment through
`dcd7a8cf37436416492595a5c1061a624cf562fe` contains seven commits and eight
Cirq implementation, test and audit-document files. All eight blobs were
SHA-verified and contain no old example, matrix-free or physical numerical
module references. This supplements the complete 4,376 changed-text audit at
`11fd9fa27ec31e74f03277010762e8d25d17c105`; it does not merge that PR or
qualify its still-open construction and packaging changes. Recheck again if
the incoming head changes before final integration.

The next verified incoming increment, from `dcd7a8cf` through
`7bbffeed70da51252e6b9f36fd05b1d85509cd6f`, adds six commits and thirteen
format-factory, test and audit-document files. All thirteen blobs are
SHA-verified; none intersects the migration sources or targets or contains an
old operational path reference. The actual retired-import guard also passes
on these files, including relative and parent-package imports.
