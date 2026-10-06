# Frozen protein asset intake

This runbook controls the non-source inputs for the final QDiffusion protein
acceptance. It does not authorize a download, accept provider terms, or publish
the resulting files. Model, tokenizer, FASTA, proprietary wheel, and generated
checkpoint bytes stay outside Git and inside the private evidence boundary.

## Current candidates

| Input | Candidate source | Candidate revision or release | License status | Intake status |
| --- | --- | --- | --- | --- |
| Human proteome FASTA | UniProt reference-proteome directory and `RELEASE.metalink` under `https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/` | Candidate release `2026_03`; canonical base file `UP000005640_9606.fasta.gz`, 7,728,297 compressed bytes, upstream MD5 `4e4f5aca22ba12eabda1e347765db069`; explicitly exclude the separate `_additional` file | The same official metalink declares CC BY 4.0. The exact release, attribution, acquisition, decompression, and final SHA-256 still require review | Not downloaded; candidate semantics are canonical-only and match the plugin's default `UP000005640_9606.fasta` name, but the `current_release` URL is mutable and cannot be frozen without approved acquisition and a content digest |
| DPLM 150M checkpoint | `https://huggingface.co/airkingbd/dplm_150m` | Candidate commit `49b7125a5d28c6418fcc2f3c4fe799352ac1488b`; repository metadata binds `pytorch_model.bin` to a 595,359,662-byte LFS object with SHA-256 `ea4eaa99536b60ed76f945f71a1a5e604f08447ec3def5104a93ca6001a59961` | The model card says the repository contains the 150M checkpoint and points to the official `bytedance/dplm` repository; that repository says it contains pretrained DPLM weights and loads the `airkingbd` model family under Apache-2.0. The model metadata still has no license field and its seven-file inventory has no license file, so the reciprocal linkage does not by itself prove that Apache-2.0 governs the separately hosted bytes. | Weight bytes were not downloaded; the pinned resolver metadata narrows the candidate identity, but acquisition, an independent digest, and explicit use approval remain required |
| DPLM tokenizer | Same frozen DPLM repository and revision as the checkpoint | Candidate commit `49b7125a5d28c6418fcc2f3c4fe799352ac1488b`; resolver metadata identifies `config.json`, `special_tokens_map.json`, `tokenizer_config.json`, and `vocab.txt` by Git object IDs `4910cb02f1840e9ac577026f601829604af58c74`, `ba0f9b53dbbf27934f7555e5d31e37bdea9317f1`, `dbcdd9fb2e742627ee310713615e0d7aeed0c34e`, and `6b946952cc35537226f07fd70957ee2f848880d2` | The pinned Hugging Face commit adds the checkpoint and tokenizer files together, and the official repository names the same model family; applicability of Apache-2.0 to those separately hosted bytes still requires explicit review | Files were not downloaded; Git object IDs are source metadata rather than the path-normalized SHA-256 required by final intake, and explicit approval remains required |
| ESM2 evaluation checkpoint | `https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt` | The archived official ESM README directly names and links this exact model file; a read-only upstream response reports 2,604,537,549 bytes, last modified 2022-08-18, S3 version `2G4typsUSwOKQLH35sqPMJ27ZLX1AEJG`, and multipart ETag `12a18098227c0ff911354647d25d494d-311`; downloaded bytes and SHA-256 still to freeze | The same official repository is MIT-licensed and its loader consumes the linked `.pt` file. Applicability of that license to the checkpoint bytes must still be explicitly recorded rather than inferred by the intake | Not downloaded; source-to-file linkage and upstream object metadata are identified, blocked on explicit review and an independent exact digest |
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
file. A read-only resolver-header check at the same commit additionally reports
the checkpoint LFS size and SHA-256 above and the four tokenizer/config Git
object IDs above. No weight or tokenizer body was downloaded. These upstream
identifiers are acquisition cross-checks, not independent local verification,
license approval, or substitutes for the final stable-tree SHA-256. Re-query
and compare the upstream reference immediately before an approved acquisition
so later mutable-branch movement cannot be mistaken for the reviewed candidate.

The UniProt candidate was refreshed against the official release metalink. It
identifies release `2026_03`, declares CC BY 4.0, and distinguishes the
7,728,297-byte canonical base FASTA archive from the 41,421,368-byte
`_additional` archive. The pinned plugin names its missing input
`UP000005640_9606.fasta`, which matches the canonical base filename. This makes
canonical-only the review candidate; it does not approve acquisition. The
published MD5 is discovery evidence only, while the final configuration
requires a SHA-256 of the exact uncompressed FASTA consumed by the workflow.
Because the official path is under `current_release`, retain the metalink and
release identifier and require a matching archived release or reviewed bytes
before treating the source as reproducible.

The ESM candidate was refreshed against the archived official ESM README and
license. The README directly maps `esm2_t33_650M_UR50D` to the selected
`dl.fbaipublicfiles.com` URL, and the repository is MIT-licensed. A body-free
response-header check records the candidate size, modification time, S3 version,
and multipart ETag above; the multipart ETag is not an MD5 or SHA-256 and must
not be promoted into the final artifact identity. This is strong source
identity evidence, but the organizational review must still state whether the
license applies to the linked checkpoint and retain that decision. Approved
acquisition must independently hash the complete bytes with SHA-256.

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
- Approve or reject UniProt release `2026_03` canonical-only input, explicitly
  excluding the `_additional` FASTA. Retain the official metalink, CC BY 4.0
  attribution decision, compressed-source identity, decompression procedure,
  and SHA-256 of the exact uncompressed FASTA.
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
