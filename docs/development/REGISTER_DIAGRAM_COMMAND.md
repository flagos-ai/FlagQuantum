# Physical circuit diagram command

The command implements K-0034 and K-0035, the locked
`main_diagram.cc` and its original test. The pinned blobs are
`90ba4360a04bb19ea3f8f127e1ac377fcfa8ad0c` and
`215b88beaa5b6dcafaebac44244665686e0f18a6`.

```console
python -m flagquantum.drawer.kmx_diagram_command --in circuit.kmb --kind svg --out circuit.svg
```

The command detects KMX or KMB input and passes the validated
physical circuit directly to the existing drawer. KMB is read into
physical records without a text round trip. Input defaults to stdin,
output defaults to stdout, and `--kind` defaults to `text`.
The other formats are `html` and `svg`. The command adds a trailing
newline, including for an empty text diagram. Diagnostic failures
return status 1. The native command validates and renders before
opening the destination file.

The original incrementer C++ test passed. Native fixtures reproduce
the exact ASCII diagram and its final newline, including the default
stdin/stdout route. Both fixtures exercise all three formats from
text and binary paths. SVG comparisons cover the complete XML node
tree and geometry. HTML compares the circuit statistics and viewport
bounds against the source command output. The existing interactive
renderer retains its secure JSON geometry payload and browser-verified
zoom and pan behavior. The command emits diagrams and does not execute
gates or change physical record counts.
