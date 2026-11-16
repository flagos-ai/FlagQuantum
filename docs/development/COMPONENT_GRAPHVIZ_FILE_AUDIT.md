# Component Graphviz source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen `graphviz.py` blob
is `b22e5252ec474f629be155f8da1f27750eb5896c`.

| Source responsibilities | Native owner |
| --- | --- |
| Instance and soquet identifier allocation | The register exporter assigns instance ordinals, register port tokens, and separate input and output identifiers before writing edges. Every identifier comes from the verified score. |
| Side partitioning and table layout | The shared signature table preserves group order, opposite-side ports, through ports, and asymmetric row spans. Core signature admission checks port uniqueness before drawing. |
| Boundary nodes and connections | Input and output nodes use formal signature names in declared order, including array subscripts. Edge endpoints retain the producer and consumer ports and the source compass directions. |
| Base rendering | The component graph exporter uses unadorned width edges and visible instance headers. |
| Compact rendering | The same exporter uses small instance headers and dotted arrows. A pure split or join retains its ports and omits their text and its header. The classifier checks directional shapes, rebinding opcodes, and bit order. |
| Typed rendering | Both endpoint declarations supply the type label. The same compact table and connection geometry carry the label. |
| SVG and Notebook output | The existing bounded Graphviz subprocess renders the DOT. The notebook display helper consumes the same output as an IPython SVG. |

The source virtual methods for tables and edges correspond to writer
callbacks and specialized exporters. The compact source's unused
`get_default_text` helper has no display effect. Native graph identifiers
use ordinals and explicit port tokens. The typed view uses native register
descriptor names. The graph admits verified native one-block call programs and retains
the existing component-score restrictions on symbolic, parameterized, and
conditioned calls. Standalone signature metadata has its own exporter.

The three presentations are compared with the frozen parallel split/three
atom/join graph. Each comparison checks all five instance tables, eight
edge connections, per-edge attributes, and real SVG edge counts. Cases
cover both boundary directions, formal output names, HTML escaping and
multidimensional array subscripts. The existing graph tutorial tests supply
the original source assertions, port integrity, unequal side counts, and
additional signature and real-kernel fixtures.

Rendering reads the declared register topology and emits zero quantum
instructions. The retained append regression verifies the parallel fixture's
complex matrix and restored work. Its physical three-X stream and the
source synthetic model's 300 T are independent observations.
