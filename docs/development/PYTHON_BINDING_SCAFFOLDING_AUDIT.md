# Python binding scaffolding disposition

## Frozen complete files

Kickmix revision `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
Google LLC, Apache-2.0. The complete `types.pybind.h` (94 lines),
`util.pybind.cc` (one line), and `util.pybind.h` (79 lines) were read.
Sixteen source and recursively quoted dependency blobs are Git-object verified.
Parents K-0252, K-0253 and K-0254 are assessed together.

The references remain unchanged. Original compile probes pass 12 positive
constexpr checks, reject triple quotes and two overlong line cases, and compile
the complete original types header. They use installed Torch pybind11 and
interpreter Python headers. This establishes original header behavior with those
headers. Original extension ABI, compiler performance and build-version equality
are outside this observation.

## Complete ownership map

| Source definition | Native owner and disposition |
| --- | --- |
| PyArrayXZ forward declaration | Compiler OperandArray owns immutable storage and lifetime-retaining views. |
| Qubit, bit, X-bit and X-Boolean type globals | Compiler OperandIdentity and OperandKind retain separate family and payload values. |
| Register type global | The same identity owner retains REGISTER independently of physical operand tags. |
| Array type global | Existing OperandArray type admission and operand-word binding retain array identity. |
| All six unsafe pointer casts | Existing bind_operand, identify_operand, OperandArray and PhysicalBasisSession register resolution consume native values. Foreign pointer arithmetic and pybind instance-layout assumptions are retired. |
| Prefix and substring constexpr helpers | Private implementation details of the retired C++ docstring compiler. Python documentation continues through tools.development.docstrings. No runtime consumer requires a second string utility. |
| Indentation copy and zero-filled array wrapper | Native Python class docstrings and the existing documentation renderer own storage and normalization. Native strings supply lifetime and termination without a fixed C++ buffer. |
| Compile-time quote and line-length refusals | C++ binding-generation policy is retired explicitly. Existing Python documentation policy remains authoritative. |
| util.pybind.cc header inclusion | No independent behavior. Retired with the foreign binding build unit. |

This is a complete native consumer replacement and scaffolding retirement.
The documentation policies differ deliberately. The source strips spaces using
the first nonempty line, leaves tabs intact, and counts a trailing newline in its
80-character limit. It exempts lines beginning with @signature or @overload and
lines containing https URLs. It rejects triple double quotes. The native renderer
uses inspect.cleandoc and Google-style sections, normalizes tabs and subsequent
indentation, and accepts long or triple-quote text. Empty-substring details remain
source-only compiler evidence. No formatting-equivalence claim is made.

## Consumer acceptance

The locked extension oracle exercises quantum, classical, X-classical,
X-Boolean, register and array storage through original constructors, iteration,
reversed views and simulator register aliases. Four batch sizes, including 257,
retain exact arbitrary-width lane masks. X-basis identities are observed through
array iteration, while simulator reads use its admitted Z-basis operands.

Native consumers replace those foreign objects with existing native owners.
Binding round trips preserve all five scalar family/value cases. Array views
remain valid after the array variable is deleted. Register numeric identity and
name aliases read identical physical values. Invalid register-as-wire and
untyped-array construction are rejected by the existing owners. Native diagnostic
spellings and repr remain separately owned, with source X-Boolean spelling used
only in the oracle comparison.

Ten documentation examples exercise the actual existing Markdown renderer,
including every source refusal and exception category. The accepted complete
Qualtran documentation parser and its full source class corpus are rerun. Existing
mixed-operand, unary iteration, file/stream, full-complex, all-small-measurement-
branch, work-zero and circuit.append(op) execution tests are reused in the batch
regression. The existing file and simulator owners remain the replacement
boundaries.

Scaffolding adds zero primitives, measurements or quantum workspace.
Analytical model is None. Consumer execution counts remain in their existing
actual physical and analytical-model receipts. This batch makes no throughput
or native SIMD claim.

The generator requires the pinned Kickmix 0.1 Simulator and exact extension hash.
Its four observations regenerate byte-exactly. Local acceptance results are
recorded in the matching G2 scaffolding receipt and synchronized ledger.

The complete batch passes 1620 concentrated tests, 121 local integration tests
and strict typing for both Python files. Black, Ruff, Verification scope,
architecture, multilevel IR, public API and whitespace checks pass. The dependency
baseline retains the identical 720 findings. All 15 focused checks pass. These
results accept all three complete parents with the explicit retirement and
native documentation-policy dispositions above.
