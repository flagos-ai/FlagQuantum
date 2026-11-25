# Upstream software and algorithm references

FlagQuantum is the software distributed by this repository. Its citation is
[CITATION.cff](https://github.com/flagos-ai/FlagQuantum/blob/main/CITATION.cff);
the root authors and release identify
FlagQuantum. The references below credit source material used by integrated
components. They do not introduce another import namespace or execution API.

## Qualtran

The frozen integration reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC,
Apache-2.0. Adapted modules retain their original notices and source links.
The [distribution notice](https://github.com/flagos-ai/FlagQuantum/blob/main/NOTICE)
and [license](https://github.com/flagos-ai/FlagQuantum/blob/main/LICENSE) accompany
the packaged implementation.

Matthew P. Harrigan, Tanuj Khattar, Charles Yuan, Anurudh Peduri,
Noureldin Yosri, Fionn D. Malone, Ryan Babbush and Nicholas C. Rubin.
*Expressing and Analyzing Quantum Algorithms with Qualtran* (2024).
[doi:10.48550/arXiv.2409.04643](https://doi.org/10.48550/arXiv.2409.04643).
The same paper metadata, including the original author identifiers, is retained
under `references` in FlagQuantum's CFF file.

The source paper entry uses `type: misc`, which the CFF 1.2.0 schema rejects.
The native reference uses the schema's `generic` category; every other paper
field is retained. The original file remains an unchanged comparison input.

The frozen upstream CFF reports software version **0.6.0**, released
**2025-04-01**, despite belonging to the v0.7.0 source snapshot. This source
metadata discrepancy is preserved in the audit; it does not change the frozen
integration commit or FlagQuantum's release version. Cite the exact revision
used for a comparison and the particular algorithm's scientific reference.

## Physical execution source

The locked Kickmix reference is the subtree at
`quantumlib/Qualtran` commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
copyright 2026 Google LLC, Apache-2.0. Its reference extension is used for
independent acceptance comparisons. Integrated allocation, gate, format and
execution responsibilities belong to FlagQuantum's existing domains.
Reference-extension execution and native execution are identified separately in
their evidence. No source-oracle installation is required for native operations.

## Contribution and evidence scope

Contribute through
[FlagQuantum's guide](https://github.com/flagos-ai/FlagQuantum/blob/main/CONTRIBUTING.md).
Upstream
contributor agreements and submission restrictions govern contributions to
their respective upstream repositories. Porting a component does not transfer
upstream authorship to FlagQuantum or imply upstream review or endorsement.

Source limitations and native extensions are recorded at the relevant component
owner. A resource formula alone is not an executable decomposition, and
agreement on a mathematical operator does not establish identical gate counts.
