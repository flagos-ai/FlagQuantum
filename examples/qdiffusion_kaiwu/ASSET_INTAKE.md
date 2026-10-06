# Frozen protein asset intake

This runbook controls the non-source inputs for the final QDiffusion protein
acceptance. It does not authorize a download, accept provider terms, or publish
the resulting files. Model, tokenizer, FASTA, proprietary wheel, and generated
checkpoint bytes stay outside Git and inside the private evidence boundary.

## Current candidates

| Input | Candidate source | Candidate revision or release | License status | Intake status |
| --- | --- | --- | --- | --- |
| Human proteome FASTA | `https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28proteome%3AUP000005640%29` | Exact UniProt release, query semantics, canonical/isoform choice, and returned bytes still to freeze | UniProt declares CC BY 4.0 for copyrightable database content; the exact release and attribution record still require review | Not downloaded; blocked on an approved frozen query and release |
| DPLM 150M checkpoint | `https://huggingface.co/airkingbd/dplm_150m` | Candidate upstream commit `7362881bcf802245a1a074e2d24137575f30d79f` | The model card reports missing YAML metadata and exposes no license declaration. The Apache-2.0 official `bytedance/dplm` repository says it contains the pretrained weights and its generation command names the `airkingbd/dplm_150m` family. This is strong candidate linkage evidence, but the separately hosted bytes still require an explicit license review and approval. | Not downloaded; linkage evidence identified, still blocked on explicit approval |
| DPLM tokenizer | Same frozen DPLM repository and revision as the checkpoint | Must equal the checkpoint revision | The pinned Hugging Face commit adds the checkpoint and tokenizer files together, and the official repository names the same model family; applicability of Apache-2.0 to those separately hosted bytes still requires explicit review | Not downloaded; linkage evidence identified, still blocked on explicit approval |
| ESM2 evaluation checkpoint | `https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt` | Exact downloaded `.pt` bytes and upstream identity still to freeze | Candidate model distribution is identified as MIT by the official model repository; applicability to the selected `.pt` bytes must be recorded during review | Not downloaded; blocked on review and exact digest |
| Kaiwu SDK wheel | QBoson platform download associated with the approved account | Exact SDK 1.3.1 wheel required by the selected plugin lane | Proprietary package and cloud-service terms are not present in the public source tree | Unavailable; blocked on QBoson account, terms, and reviewed wheel |

The candidate DPLM commit above is discovery metadata, not an approved frozen
revision. Public availability alone is insufficient for acceptance.

The linkage evidence was rechecked on 2026-10-06 against the official
[`bytedance/dplm` repository](https://github.com/bytedance/dplm), its
[Apache-2.0 license](https://github.com/bytedance/dplm/blob/main/LICENSE), and
the pinned
[`airkingbd/dplm_150m` commit](https://huggingface.co/airkingbd/dplm_150m/commit/7362881bcf802245a1a074e2d24137575f30d79f).
The Hugging Face commit records the checkpoint and tokenizer files together,
but contains no license file or model-card license metadata. These facts narrow
the review question; they do not authorize download or use.

## Freeze procedure

1. Obtain explicit approval for the exact source, license or service terms, and
   storage location before downloading any large or proprietary artifact.
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
- Obtain the Kaiwu 1.3.1 wheel and service terms through the authorized QBoson
  account; do not substitute the public Kaiwu source repository for that wheel.
