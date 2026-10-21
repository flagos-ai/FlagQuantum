# Explicit hybrid layers, readout semantics and readable circuit panels

## Authorization and scope

On 2026-10-21 the user authorized these three draw_hybrid improvements, an
updated executable CNN/quantum Notebook, compiled and visually reviewed PDFs,
tests and a new PR. This decision authorizes the additive drawer namespace type
and keyword arguments below and removal of invented default feature counts.
Owner: compiler/drawer. No Core IR, model execution or root exports change.

## Interface and compatibility

`ClassicalLayer(label, input_shape, output_shape, trainable=False)` is immutable
presentation data. Shapes contain positive, batch-free integer dimensions.
Existing IR and nn.Module types do not describe classical diagram labels and
shapes; this record describes presentation only and cannot execute a model.

`draw_hybrid` keeps accepting string encoder/head labels, or accepts nonempty
sequences of ClassicalLayer. It validates adjacent shapes, the encoder's final
vector against encoder_features, and readout_features against the head input.
Detailed heads require an explicit readout label and feature count. Shape checks
verify the supplied description; they do not infer layer mathematics or trace
arbitrary PyTorch code. The Notebook separately checks actual layer outputs.

The encoder_features/head_outputs defaults become None: detailed layers derive
them; string stages display unspecified counts. Existing explicitly supplied
counts and labels retain their meaning. Previously omitted counts displayed
fictional example values (32 and 10); callers wanting those must supply them.

New keywords: readout, readout_features, encoder_trainable, quantum_trainable,
head_trainable, quantum_view='auto'|'overview'|'detail', gates_per_panel=8.
Unspecified readout remains visibly unspecified with a data arrow; an explicit
readout uses a double line. Declared readout is not an inferred measurement.
Frozen stages have no optimizer update arrow; stage flags override layer flags.
The drawing never runs a forward pass, computes gradients or mutates parameters.

Short circuits can appear inline at natural font size. Auto selects separate
ordered panels for larger circuits; overview omits circuit detail, and detail
always shows panels. Panel boundaries continue the same registers, not reset
states. Final observable/readout annotations appear only in the last panel.
No entire circuit is shrunk to fit an architecture inset. No new dependency,
resource license change, protected snapshot regeneration or executable IR exists.

## Acceptance

Preserve old explicit-call behavior, reject shape and semantic conflicts, check
frozen-stage feedback, test immutable input ownership and all layout modes,
compile real PDFs for detailed CNN and long-circuit layouts, execute the Notebook
including actual shapes and gradients, render every final PDF page and inspect
labels, routing, font sizes, margins and continuation semantics. Test source-only
work without external tools and run repository quality/default/runtime gates.
