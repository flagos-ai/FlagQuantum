# Reference class, module and TOC source closure

## Frozen source and native owners

Q-0074, Q-0075 and Q-0076 cover the complete original render_major_class.py,
render_module.py and render_toc.py at Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. All three unmodified Git blobs are
verified by the source differential tests. Copyright and Apache-2.0 license
text remain in the native files.

Tools owns reference_class_pages.py, reference_module_pages.py and
reference_toc.py. Each keeps the original file's rendering responsibility.
Accepted page records, RenderContext, alias formatter, LinkingWriter, signature
renderer, docstring renderer and WriteIfDifferent supply the dependency closure.
The signature/docstring boundary now explicitly accepts Griffe Alias values,
matching the original alias forwarding behavior. Class aliases retain their own
visible method names while delegating their parameters and documentation.

## Complete comparisons

Full class-page output compares byte for byte with the source, including
summary, overview, aliases, normal and aliased append methods, and omission of
private/special/initializer/attribute members. Every class-member writer call
also compares directly. Equal-content second rendering preserves the native
output timestamp through the existing file owner.

Module-page comparisons cover empty members, minor members alone, modules
alone, major classes alone and all three categories together. The minor class
case includes documented nested members and actual Griffe-resolved inherited
members. The module collection and symbolic base reference provide a real
inheritance relationship. Source set ordering for documented submembers remains
preserved. Major/minor member summaries, function signatures and complete linked
page bytes compare using the same static Griffe objects.

TOC comparisons cover section filtering, sort order, empty module omission,
major-class inclusion, generated paths, rendered file bytes and empty-section
ValueError. Context-provided callbacks retain ownership of section grouping and
page ordering. The package walker remains the consumer responsible for policy.

The original TOC opens its output without an exception-safe close. A bounded
failure test retains the opened stream to verify that it remains open after the
source error. Native render_toc uses the file context manager and closes on the
same error. The test explicitly closes the reference stream afterward. Native
TOC still writes directly to the destination, preserving original partial-output
semantics. The other two renderers retain accepted WriteIfDifferent behavior,
including its documented buffer write when a context body raises.

## Acceptance and remaining scope

Run test_reference_page_renderers.py with accepted signature/docstring,
page/link/context, notebook, SDK helper, discovery, metadata and graph tests.
Strict mypy covers the three new renderers, two shared Alias annotations and
the source differential tests. Black, Ruff, Integration scope, architecture,
IR, protected API and diff checks complete the local batch gates.

This batch renders classical documentation and reuses the accepted real
circuit.append(op), full complex-amplitude, workspace and independent observed
primitive/model evidence. No new quantum decomposition or resource estimate is
introduced. Griffe and Markdown optional versions and license ownership remain
as recorded in REFERENCE_PAGE_LINKS_FILE_AUDIT.md.

Q-0071 package walking/configuration and Q-0065/0066/0073 package export files
remain open until the complete pipeline is connected and source-validated.
Q-0080 explicit notebook catalog and Q-0086 older TensorFlow-docs pipeline stay
separate. Pending Proposals 116, 117 and 118 remain unimplemented.
