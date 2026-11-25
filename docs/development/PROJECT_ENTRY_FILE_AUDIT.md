# Complete project entry, citation, contribution and license source disposition

This batch reviews five complete project files from Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen inventory hashes were
verified before adaptation. The three missing metadata files were recovered
by their exact Git blob identifiers; existing README and hosted-build inputs
already matched. Original files remain unchanged in reference evidence.

| Parent and source | Frozen Git blob | Native disposition |
|---|---|---|
| Q-0016 CITATION.cff | 619139543a3f861d0b8f08851ef302e28e35f499 | Existing FlagQuantum root citation remains authoritative. The full original preferred paper citation is included under references, with the invalid source type adapted as described below. Software identity and source-version discrepancy are documented separately. |
| Q-0018 CONTRIBUTING.md | 002e7e2634400bdb9db1f758e238ddc1541f53d0 | Existing native contribution guide owns installation, branching, review, code of conduct, bug reports and quality checks. The added register-adaptation section connects domain ownership, source attribution and scientific evidence. |
| Q-0019 LICENSE | d645695673349e3947e8e5ae42332d0ac3164cd7 | Existing native Apache-2.0 text is byte-identical apart from the source's leading newline. NOTICE carries explicit Qualtran and physical-execution source attribution into package license metadata. Existing asset licenses are retained. |
| Q-0021 README.md | 74ce34b1ad2c8a5785b9b0ea5886a5a1b1076288 | Existing native installation, usage, documentation, examples, community/review and citation routes retain FlagQuantum identity. Added register-library/resource links and upstream citation/notice links close discoverability and provenance. |
| Q-0014 .readthedocs.yaml | 0f615f6e8f8e17a0d5c0f61083e35e9c67367ea2 | Complete source profile reviewed: version 2, Ubuntu 22.04, Python 3.10, docs/conf.py and docs requirements. Local native Sphinx ownership is already accepted; hosted installation/publication is not implemented or claimed here. This parent stays open. |

## Citation and source discrepancies

The frozen source CFF declares version 0.6.0 and release date 2025-04-01.
Those values are not silently changed to match the v0.7.0 integration label.
The actual source revision and this metadata difference are documented in
[the upstream software reference](../reference/UPSTREAM_SOFTWARE.md).

The official versioned CFF 1.2.0 JSON schema also rejects the original paper's
`type: misc`. Native references use its supported `generic` category. Every
other preferred-paper field, including eight authors, affiliations, ORCIDs,
DOI, year, identifier and keywords, equals the original. The original CFF's
single schema failure is retained. Native CFF validates against that same
schema, and all preexisting native root metadata remains equal to the batch
base. No upstream author is relabeled as a FlagQuantum author.

The original contribution policy's Google CLA and restriction on AI-generated
submissions remain rules for submissions to that upstream repository. They are
not adopted as new FlagQuantum rules or presented as satisfied by this port.
FlagQuantum uses its existing protected-API, review, testing and ownership
policies. Source mailing lists, meetings, Google contact/product disclaimer and
logos are upstream identity and governance, replaced by native project routes;
they are not a missing numerical capability. No message or contribution is
submitted to an upstream project.

## Distribution and documentation consumers

The existing setuptools license-files declaration includes NOTICE. The existing
manifest also retains CITATION.cff. A real PEP 517 source distribution is built
and opened: README, LICENSE, NOTICE, citation, manifest and program inputs must
be byte-identical to their owning files. PKG-INFO must retain the native name,
release, Torch requirement and NOTICE license-file entry. Existing package-local
native source inclusion and tests/examples/benchmark quarantine remain checked.
This is source-package evidence, not compiled-wheel or release promotion.

The exact new upstream-reference page is rendered with the actual native Sphinx
configuration, theme and assets, with warnings as errors. Repository-root file
links use repository URLs so the hosted documentation does not interpret a
root CONTRIBUTING.md as an absent Sphinx document. Source/native theme fixtures,
local relative links, CFF schema and original license text are checked at their
existing boundaries. No new runtime or optional production dependency is added.

These files define no quantum gates, register transformation, measurement
history, private workspace or resource formula. Primitive/model and complex
amplitude comparisons are inapplicable to this metadata batch; prior component
execution certificates are reused without rerunning them as documentation tests.

The four citation/contribution/license/README parents can close after batch
acceptance. Hosted ReadTheDocs installation, full scientific documentation
requirements, warning-free whole-site publication and the complete homepage
retain their separate open parents. Local fixture rendering does not certify
those deployment or content scopes. No G3 validation or PR merge is performed.

## Batch acceptance

All 31 concentrated source-distribution, physical-packaging, native/source
renderer and link checks pass. The exact new reference page separately passes
the real native Sphinx renderer with warnings as errors. Both CFF inputs are
validated with the same official 1.2.0 schema: native passes and the original's
single invalid `misc` type is retained as a source counterexample. The actual
native archive contains byte-identical NOTICE and citation metadata.

Strict typing covers the changed source-distribution test. Black, Ruff,
Integration scope, local links, language, architecture, IR, public API and
documentation-authority checks pass. The dependency scan remains the identical
720-finding baseline, not globally green. Earlier 145-case local execution
acceptance is reused because no executable quantum owner changed. No new
compiled-wheel, hosted service or whole-site warning-free claim is made.
Receipts use `g2-project-entry-*-20261125` in the reference evidence directory.
