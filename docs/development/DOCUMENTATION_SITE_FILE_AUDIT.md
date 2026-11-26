# Complete documentation site configuration and assets

This batch covers the complete Qualtran v0.7.0 `docs/conf.py`, both stylesheets,
both templates and five branding images at frozen revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Google LLC source attribution and
Apache-2.0 notices remain on adapted configuration, styles and templates.
The reference tests verify all eleven source blobs, including the separately
open root index. The source hosted-build profile has its own scope.

## Native ownership and source disposition

| Complete source files | Native owner and accepted responsibility |
| --- | --- |
| Q-0134 docs/conf.py | Native docs/conf.py retains all source assignments. Project identity, repository URL, sidebar routes, style/template filenames and branding locations follow the native documentation structure. Every other assignment is equal to the frozen source. |
| Q-0124 fixes.css | Native docs/_static/layout.css retains all three original table and navbar rules. It adds the native footer's dimensions and a white-logo treatment for the dark theme. |
| Q-0125 qai-theme.css | Native docs/_static/theme.css retains every source selector, declaration and color, using FlagQuantum-owned custom-property names. The source's attribute-qualified hover selectors remain unchanged in behavior. |
| Q-0130 qai_logo.html | Native project-logo.html retains the linked footer-image role, with project identity and a local asset. |
| Q-0131 qualtran-sidebar.html | Native section-sidebar.html retains the title, accessible navigation container, generated subtree and all six original generator options. Three native landing pages consume it. |
| Q-0123 favicon.png | A 64 by 64 native symbol icon is derived from the existing native logo and copied by Sphinx. The crop/resize recipe and input/output hashes are retained externally. |
| Q-0126 mode-sensitive logo and Q-0127 small logo | Existing assets/logo_flagquantum.png supplies native header and scaled logo roles. The native dark-theme CSS supplies the contrasting logo treatment. |
| Q-0128 vertical and Q-0129 horizontal Google Quantum AI branding | Existing native project branding supplies these presentation identity roles through the header/footer logo. Original Google images remain frozen audit inputs. Native site assets use FlagQuantum identity and local paths. |

The native configuration connects Concepts, Register components and API/contract
landing pages to the existing document structure. The new register portal links
fourteen existing authored arithmetic, comparison, permutation and finite-field
references. Root installation and source-development instructions use native
package/repository names. The actual Bell-circuit example executes in acceptance.

The subsequent [homepage audit](DOCUMENTATION_HOMEPAGE_FILE_AUDIT.md) connects
the complete native fundamentals and published component catalog to the root.
Its acceptance is separate from this configuration/asset batch. The full API
reference generator, hosted ReadTheDocs installation and scientific
documentation requirements retain their own parent scope.

## Actual renderer comparison

Both actual source and native configurations render the same controlled
Markdown/notebook fixture with warnings treated as errors. Every custom sidebar
route renders its own title. Mathematical text, definition lists, generated
heading anchors, complete stylesheets, footer image and favicon are checked.
Stored notebook output survives, and a code cell that raises if executed remains
unexecuted. Native footer paths resolve to the local input image at every depth.

The source configuration's remote logo and favicon URLs are overridden only in
the controlled fixture with their immutable local source images. The source
configuration file stays byte-identical. Its default URLs name upstream main,
and its footer image uses a historical remote commit. Those external-resource
behaviors are recorded separately from the local native publication policy.
This comparison establishes configuration and asset behavior on the fixture.

The complete native documentation tree also builds through the ordinary Sphinx
entry. Its content warnings are retained as a separate publication boundary.
The initial full-tree build generated 1325 pages with zero errors and 1565
warnings. Most warnings concern authored pages outside a toctree, repository
links outside the Sphinx source root and unavailable code-block lexers. The
original two configured MyST warning suppressions are preserved. The final
full-tree build contains 1326 documents, zero errors and the same 1565 warnings.
The new configuration, landing page and register portal introduce no warnings.
This audit itself retains one orphan-page warning under the existing
development documentation navigation policy.
Strict whole-tree warning-free publication remains open.

## Optional environment

The isolated renderer uses Sphinx 8.1.3, myst-nb 1.3.0 and pydata-sphinx-theme
0.16.1. All source-constrained installed versions and all active base requirement
edges are verified across its 73 distributions. Bundled license files remain
present and hash-recorded. This development renderer stays outside the framework
runtime and uses the standard Sphinx configuration/CLI boundary.

The unchanged source lock has 194 pins. Pip 25.3 rejects its two extras-bearing
lines as constraints. A derived constraints file retains all versions and
markers while removing only those constraint extras. Darwin-only appnope 1.0.0
is the single explicit platform addition omitted by that source lock. The
ambient pip warning about virtualenv's packaging requirement is retained.
The isolated renderer's requirement closure passes. Full ambient environment
and cross-platform scientific-lock acceptance retain their separate scope.

Receipts use `ftqc-reference-env/g2-docs-site-` and cover immutable source files,
constraint derivation, platform additions, installed metadata/licenses, branding
derivation, strict source/native fixture rendering and full native publication.
No quantum gate body, measured branch, private work or resource formula is
defined by these presentation files. Their quantum resource categories are
inapplicable. Existing component-owner numerical and erasure evidence is reused.
