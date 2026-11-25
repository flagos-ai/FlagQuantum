# Complete rectangular and Kaiser window constructor disposition

## Frozen source and native ownership

Complete phase_estimation/qpe_window_state.py and kaiser_window_state.py,
their tests and direct allocation/OnEach dependencies remain frozen at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. The intake verifies
seven exact Git blobs and actual locked Kickmix 0.1 Simulator availability.

| Source definition | Native owner |
|---|---|
| QPEWindowStateBase m_bits/m_qdtype/m_register | Existing Algorithms PhaseEstimationWindow owns the RIGHT-only unsigned fractional qpe_reg output signature. |
| RectangularWindowState fields and build_composite_bloq | Ecosystem _phase_window_component_library.py reuses rectangular_window_program, its allocated output and existing OnEach/H helper closure. |
| Rectangular precision/delta and standard-deviation constructors | Existing Algorithms RectangularWindowState owns both original concrete and symbolic expressions. |
| KaiserWindowState fields and signature | Ecosystem retains the exact source declaration with a RIGHT-only output and no gate body. |
| Kaiser coefficients and my_tensors | Existing Simulation kaiser_window_coefficients and Algorithms KaiserWindowState retain periodic normalized SciPy coefficients and tensor order. |
| Kaiser from_precision_and_delta | Existing Algorithms owner retains the complete ln/ln-ln expression and original failures. |
| Kaiser build_call_graph | Existing owner retains the separately declared rotation preparation child, original precision, concrete coefficient snapshots and symbolic length. |
| Four examples and DocSpecs | Original small and symbolic examples resolve with unchanged signature metadata. Existing catalog/drawing owners are reused. |

Interop conversion imports no source package into production and introduces no
numerical kernel. Concrete rectangular output wires are allocated from the
existing clean pool and transferred to the caller. They are logical outputs,
so cleanup never releases them as hidden workspace. Symbolic rectangular
Programs retain their allocation/OnEach graph and an unresolved helper body.
Physical execution still requires concrete sizes.

## Source execution, accounting and boundaries

Both constructors preserve the full source register declaration. Rectangular
stored depths zero, one and two retain the existing native Program. Source
producer, operand and output traces match, including symbolic width. Concrete
noncontiguous output bindings match the accepted Program and full complex
preparation amplitudes match the immutable source tensor. Existing spectator,
inverse, normal append, drawing and resource evidence is reused.

A width-n rectangular preparation emits exactly n H and uses n logical output
wires, no private work, measurement or feedback. Its source model separately
reports n Clifford and zero T. The single empty measurement history applies.
Original zero-width rectangular allocation/OnEach graphs remain representable,
while original tensor contraction raises ValueError. At initial acceptance,
native physical zero-bit fixed-point storage retained its refusal. Approved
Proposal 116 now admits the unsigned empty word; with a live borrowed carrier
the stored zero-width Program returns that empty output and emits no gates.
Standalone binding still requires at least one physical wire. The source/native
H/window/T context's full operator is verified in
[the current test acceptance](WINDOW_SOURCE_TEST_ACCEPTANCE.md).

Kaiser has no original decomposition. The constructor remains a bodyless
source declaration, with explicit physical binding refusal. Exact coefficients
and tensor evidence belong to the existing numerical owner. Its optional
quantized rotation child remains an independently declared approximation.
No primitive count or gate stream is attributed to the source Kaiser root.
Small-precision model failures, coefficient overflow/NaN and symbolic physical
limits remain recorded by existing complete-source evidence.

## Acceptance receipts

The concentrated window, numerical, graph and Luis-Perina dependency
regression passes 168 cases. Local integration passes 145 cases. Unchanged
original tests pass five cases with five source skips and one expected
source failure. Strict checking covers both changed Python files. Black,
Ruff, Ecosystem scope, repository language, architecture, multilevel IR and
public API checks pass. Dependency checking retains exactly the existing
720 findings. Receipts use prefix g2-window-constructor and date 20261123.
