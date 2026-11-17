# KMB conversion commands

Run `python -m flagquantum.compiler.formats.kmb_convert kmx2kmb --in input.kmx
--out output.kmb` or use `kmb2kmx` for the reverse direction. Omitted paths
select standard input or output. The reverse direction adds a final newline to
nonempty KMX, matching the locked writer.

The reference files are `main_kmb2kmx.cc`, `main_kmx2kmb.cc` and
`main_kmx2kmb.test.cc` (K-0037–K-0039, blobs
`f214810d99eee8d013e9f1b51b2d735f4e8982d3`,
`be3b9c3dba078f3798d0fa52444b7c9f4d8cf7bc` and
`f1a9129987f06b681a3e5c73d646310f56e4a754`). The native tests reproduce the
original incrementer and empty round trips and verify standard streams. Native
argument parsing uses argparse, including status 2 on argument errors.
Conversion errors use status 1. Validation completes before opening an output
path, so invalid input preserves an existing destination.
