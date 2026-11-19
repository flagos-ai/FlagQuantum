# Complete physical-cost summary

Q-0921 maps the complete `physical_cost_summary.py` file at Qualtran v0.7.0
revision `bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`fa6806ae0a1b7dd8cc63d19f0cb6635101aa807d`, to the existing
`qec/resources/physical_cost.py::PhysicalResourceEstimate`.

The source failure probability, footprint and duration map to the native
first-order error sum, data-plus-factory footprint and duration in hours.
Qubit-hours is exactly their product. The native frozen result additionally
retains the authoritative cycle count and data/factory contributions. A duplicate
three-field wrapper is unnecessary. Zero-demand, nontrivial split contributions
and first-order errors above one are preserved. Frozen value semantics and
hashing are checked. The native summary remains the actual shared physical
model's result and the existing search-cost input. Candidate qubit-hours and
ordering are independently compared with source summaries.

Existing complete physical-model and search tests prove all actual consumers,
including the 48-case source matrix and the original THC search. Those accepted
evidences are reused. Summary construction emits no primitive gates or new
physical assumptions. It aggregates existing model outputs. Statevector,
measurement-feedback and workspace tests belong to the emitting components.

All 444 concentrated resource regressions pass. Both record-file tests pass
strict mypy, Black and Ruff. Core scope, architecture, IR and protected API
checks pass. The source file has no companion original test file.
