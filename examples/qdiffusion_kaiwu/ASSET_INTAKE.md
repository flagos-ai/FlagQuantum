# Frozen protein asset intake

This runbook controls the non-source inputs for the final QDiffusion protein
acceptance. It does not authorize a download, accept provider terms, or publish
the resulting files. Model, tokenizer, FASTA, proprietary wheel, and generated
checkpoint bytes stay outside Git and inside the private evidence boundary.

## Current candidates

| Input | Candidate source | Candidate revision or release | License status | Intake status |
| --- | --- | --- | --- | --- |
| Human proteome FASTA | `https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28proteome%3AUP000005640%29` | Exact UniProt release, query semantics, canonical/isoform choice, and returned bytes still to freeze | UniProt declares CC BY 4.0 for copyrightable database content; the exact release and attribution record still require review | Not downloaded; blocked on an approved frozen query and release |
| DPLM 150M checkpoint | `https://huggingface.co/airkingbd/dplm_150m` | Candidate upstream `main` head observed on 2026-10-06: `49b7125a5d28c6418fcc2f3c4fe799352ac1488b` | The public model metadata has no license field and its seven-file inventory has no license file. The Apache-2.0 official `bytedance/dplm` repository says it contains pretrained weights and names the `airkingbd/dplm_150m` family. This is strong candidate linkage evidence, but the separately hosted bytes still require an explicit license review and approval. | Not downloaded; source identity and linkage evidence identified, still blocked on explicit approval |
| DPLM tokenizer | Same frozen DPLM repository and revision as the checkpoint | Must equal the checkpoint revision | The pinned Hugging Face commit adds the checkpoint and tokenizer files together, and the official repository names the same model family; applicability of Apache-2.0 to those separately hosted bytes still requires explicit review | Not downloaded; linkage evidence identified, still blocked on explicit approval |
| ESM2 evaluation checkpoint | `https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt` | Exact downloaded `.pt` bytes and upstream identity still to freeze | Candidate model distribution is identified as MIT by the official model repository; applicability to the selected `.pt` bytes must be recorded during review | Not downloaded; blocked on review and exact digest |
| Kaiwu SDK wheel | Qboson-owned PyPI release metadata at `https://pypi.org/pypi/kaiwu/1.3.1/json`, or the QBoson platform | Linux candidate `kaiwu-1.3.1-cp310-none-manylinux1_x86_64.whl`; published and independently reproduced SHA-256 `7334cabd4ff0ae02e042d1c38ed292211573e83e2ed8e92fdf41af52e8991455` | PyPI identifies owner `nixd` and author `Qboson Inc` but exposes neither a license expression nor license files. Static wheel inspection also found no `LICENSE`, `COPYING`, or `NOTICE` member. The public QBoson platform agreement effective 2026-07-09 covers Kaiwu SDK, KPP, and remote APIs, but its use, transfer, data, and risk clauses require organizational review | Downloaded only to an owner-only temporary review directory and statically inspected; not installed, imported, executed, or approved; blocked on explicit package and service-terms approval |

The candidate DPLM commit above was obtained from both the public repository
reference and the model metadata API. It is discovery metadata, not an approved
frozen revision. Public availability alone is insufficient for acceptance.

The linkage evidence was rechecked on 2026-10-06 against the official
[`bytedance/dplm` repository](https://github.com/bytedance/dplm), its
[Apache-2.0 license](https://github.com/bytedance/dplm/blob/main/LICENSE), and
the exact candidate
[`airkingbd/dplm_150m` tree](https://huggingface.co/airkingbd/dplm_150m/tree/49b7125a5d28c6418fcc2f3c4fe799352ac1488b).
The Hugging Face metadata enumerates `.gitattributes`, `README.md`,
`config.json`, `pytorch_model.bin`, `special_tokens_map.json`,
`tokenizer_config.json`, and `vocab.txt`; it records no license field or license
file. These facts narrow the review question; they do not authorize download or
use. Re-query and compare the upstream reference immediately before an approved
acquisition so later mutable-branch movement cannot be mistaken for the reviewed
candidate.

The Kaiwu 1.3.1 documentation points to a QBoson user agreement. The public
[QBoson Quantum Cloud Platform User Service Agreement](https://platform.qboson.com/agreement?type=QBoson-SPQC-Platform-Users-Agreement),
effective 2026-07-09, expressly includes Kaiwu SDK, Kaiwu-PyTorch-Plugin, remote
computing APIs, and SPQC-1000/SPQC-550/SPQC-X services. Its clauses state that
SDK/API use is limited to lawful use under the agreement, prohibit sale,
transfer, or sublicensing without written permission, assign installation and
use risk to the user, and describe the service as for the user's own use rather
than provision to a third party. The agreement also contains platform-data and
confidentiality provisions. This inventory records the terms as review input;
it is not legal interpretation, acceptance, or approval, and it does not prove
that the agreement alone grants the package, redistribution, organizational,
or CI/container usage rights required by this integration.

## Freeze procedure

1. Obtain explicit approval for the exact source, license or service terms, and
   storage location before downloading any large or proprietary artifact.
   Record the Kaiwu decision in a private copy of
   `sdk_approval.example.json`; placeholders or public file permissions cannot
   authorize a provider smoke run. Embed the identical object under
   `acceptance_config.json.kaiwu_sdk` for the later QDiffusion lane.
2. Record the upstream release or full commit, exact HTTPS source URL, license
   identifier, license-evidence URL, and a timezone-aware review timestamp in a
   private copy of `acceptance_config.example.json`.
3. Acquire only the selected files. Do not use implicit `from_pretrained`
   downloads on either A800 host. For ESM2, use the fair-esm `.pt` format
   consumed by the evaluator rather than an unrelated Transformers weight
   format.
4. Reject symlinks and special files, compute the repository's file or
   path-normalized tree SHA-256 identity, and place that digest in the frozen
   config. The checkpoint and tokenizer must come from the same DPLM revision.
5. Run `preflight_protein_artifacts.py` offline. Keep its mode-0600 record in a
   private mode-0700 directory; do not add artifact paths or bytes to Git.
6. Build and verify the closed-world Python environment lock from separately
   reviewed wheels. This does not install, license, or approve a wheel.
7. Only after all placeholders are removed and the config passes the fail-closed
   validator may a quota-consuming launcher resolve QBoson credentials.

## Remaining decisions

- Approve or reject the candidate linkage between the official Apache-2.0
  `https://github.com/bytedance/dplm` repository and the exact
  `airkingbd/dplm_150m` checkpoint and tokenizer bytes. Retain the review
  decision and evidence; do not convert the repository statements above into
  approval automatically. If the linkage is rejected or remains inconclusive,
  choose a separately reviewed compatible checkpoint or stop final protein
  acceptance.
- Freeze one UniProt release and explicitly choose whether isoforms are included.
- Confirm that the official ESM2 model license applies to the selected `.pt`
  checkpoint and retain the evidence URL used for that decision.
- Approve or reject the statically inspected PyPI Kaiwu 1.3.1 Linux wheel and
  retain the applicable package and cloud-service terms. Resolve whether the
  public user agreement permits the intended organizational development, isolated
  container execution, and FlagQuantum adapter distribution, and whether a
  separate package license or written QBoson permission is required. Only
  after approval, promote only the exact reviewed filename and SHA-256 into the
  isolated environment lane; do not substitute the public Kaiwu Community
  source repository for the proprietary provider wheel. The temporary review
  copy and its metadata do not themselves establish any right to execute or
  redistribute it.
