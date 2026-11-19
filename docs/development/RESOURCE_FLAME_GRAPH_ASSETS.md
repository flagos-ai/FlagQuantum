# Complete resource-flame renderer asset integration

Q-0939/Q-0940/Q-0941 cover the frozen renderer, its complete CDDL license and
its provenance metadata at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The complete asset triple and its native example passed local acceptance.

The rendering implementation belongs to Drawer presentation assets under
`flagquantum/drawer/_assets`. The existing explicit-executable SVG renderer
owns input streaming, temporary files, timeout and diagnostics. Compiler retains
all resource costs and graph expansion. The frozen Perl script is distributed
byte-for-byte with its original copyright notices, license and metadata. Its
native asset filename describes resource flame graphs. No upstream Python
package facade or public export is introduced.

Packaging and license declarations are an Integration-owned change. The complete
asset triple must appear in source distributions and wheels, with the license
also recorded in distribution license metadata. A separate native example will
pass the packaged executable explicitly to the existing renderer. Perl remains
a caller-installed executable dependency. The direct executable workflow will
be qualified on the current Unix host. Original source/test rendering and browser
evidence is reused only after exact frozen-byte verification.

Acceptance requires the three frozen blobs, packaging declarations, actual
source/wheel members and license metadata, executable file mode, native resource
rows and complete SVG equality, relevant failure cases, focused and concentrated
regression, strict types, formatting, scope and architecture checks. Rendering
emits zero quantum gates, measurements or work wires. QFT's original expected
496 T versus actual 190 T remains an independent open source-test defect.

## Frozen bytes and distribution evidence

| Native asset | Frozen Git blob |
| --- | --- |
| `_assets/resource_flame_graph.pl` | `8c917ecf323e13c1ab3c1107c1963f33dec0123e` |
| `_assets/resource_flame_graph.metadata` | `8c6696d1a9b9c36c929d0a01a00dfe46f4de1cbb` |
| `_assets/licenses/cddl-1.0.txt` | `2230598ebad9e470736e12119db740007c1a764b` |

The package license expression is now
`Apache-2.0 AND MIT AND LPPL-1.3c AND CDDL-1.0`. Both built distributions
contain all three exact blobs and record the CDDL text as a `License-File`.
The source archive is 3,476,702 bytes and the native macOS arm64 wheel is
4,717,265 bytes. Both pass the existing artifact verifier's 5,000,000-byte
limit. The wheel contains the compiled native CPU extension. The Perl script
retains executable mode 0755 in both formats. Omission checks reject each
missing asset independently.

An actual wheel installation into a temporary target outside the checkout
rendered the complete native example into a 13,632-byte SVG. Its XML contains
Algorithm, Select, four T counts, search and zoom. The native extension also
loads when the already installed Torch package is linked adjacent to the
temporary target, matching its relative runtime-library layout. An isolated
target without adjacent Torch failed to resolve `libomp.dylib`; this check does
not claim a fresh dependency installation. Rendering itself succeeded before
that extension check. Perl remains external. Direct execution was qualified on
macOS arm64 and is explicitly excluded on Windows in the new execution tests.

The untouched source CSwap(10) and the packaged native renderer produce equal
complete SVG trees, including scripts, with 70 T counts. Canonicalization only
sorts non-overlapping frame siblings whose order depends on Perl hash iteration.
Existing frozen-byte browser search/zoom evidence remains applicable. Empty
input, timeout and temporary-file ownership checks pass.

Local acceptance includes 49 focused cases and 383 concentrated cases without
skips, strict mypy for five changed Python files, Black, Ruff, team scope,
architecture, IR and protected API checks. The FTQC CI editable-install command
now reuses its provisioned CPU Torch with `--no-build-isolation`, satisfying the
existing installation contract. The complete dependency scan still reports
706 existing findings, with zero in this batch's changed paths.

Evidence is recorded under `ftqc-reference-env/g2-flame-assets-` in the build,
artifact JSON/log, focused, regression, strict and dependency outputs.

Run the native workflow from a checkout with the optional drawing dependencies
and Perl available.

```sh
python examples/algorithms/resource_flame_graph.py /tmp/resource-flame.svg
```

This produces a four-T analytical resource graph. It emits no quantum primitive
operations, measurements or work wires. Model counts remain separate from
executed primitive counts.
