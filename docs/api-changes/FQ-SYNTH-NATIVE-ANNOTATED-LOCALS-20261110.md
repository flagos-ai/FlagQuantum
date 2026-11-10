# Standard Python annotations for unsigned arithmetic locals

The authorized native example migration admits `Annotated[QNum, literal_width]`
for a fresh unsigned arithmetic local inside the existing root `within`/`apply`
profile. Ports already accept this standard Python annotation. Local declarations
normalize it to exactly the same width and existing `qreg.local`/`qreg.compute`
effects as `QNum[literal_width]`, which remains accepted unchanged. This avoids
invalid static type parameters in runnable, strictly typed Python examples.

The metadata is exactly one plain integer literal in the existing 1..256 range.
Signed/scaled tuple metadata, booleans, expressions, symbolic widths, additional
metadata and non-QNum base types are refused before lowering. Existing fresh
storage, borrowed input, uncompute, release, alias, scope and budget rules remain
authoritative. There is no new public export, function signature, physical IR
opcode, numeric format or lifetime semantic. This is an unreleased Compiler
syntax admission, not Stable Core or release promotion.

Acceptance compares both annotation spellings on a complete complex state,
checks malformed annotations and lifecycle violations, and runs the concrete
26-phase fixed-point QSVT source against an independent Circuit. Complete original
corpus replay must preserve all216 identities, stages and measured resources.
