# Complete physical hardware parameters

Q-0922 maps the full `physical_parameters.py` file at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`08d00e221fd0292fc4b0f19d47b790398fe04d36`, to the existing domain-owned
`qec/resources/physical_parameters.py`.

The two fields, defaults and both literature constructors are implemented.
Beverland superconducting, ion and Majorana assumptions are compared for both
error-rate options. The four-gate/two-measurement cycle formula and Majorana's
special error rates match exactly. Both Gidney-Fowler options match exactly.
Direct construction, frozen values and hashing remain native dataclass behavior.
Previously accepted real physical-model and notebook consumers reuse this type.
No new type, adapter or production dependency is needed.

Native admission is deliberately stricter than the unvalidated source record.
Physical error must be strictly between zero and 0.01, and cycle duration must
be positive and finite. Source records can hold zero, negative, threshold,
NaN or infinite values. Native rejection for these cases is explicitly tested.
The complete-file mapping therefore claims valid physical-domain formula parity,
with this admission boundary retained. It does not claim source permissiveness,
attrs repr text or subclass-return identity. Native literature constructors use
`cls`, whereas the source returns its base class. Existing protected constructors
and validation are unchanged.

These fields are physical-model inputs, separate from primitive emissions and
logical gate costs. They introduce no circuit, measurement or workspace behavior.

All 444 concentrated resource regressions pass. Both record-file tests pass
strict mypy, Black and Ruff. Core scope, architecture, IR and protected API
checks pass. The source file has no companion original test file.
