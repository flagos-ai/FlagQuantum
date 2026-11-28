# Replace three offline Runtime recipients with same-host qualified peers

Owner: Integration. This declaration precedes its workflow implementation.

At PR #1360 head 82dcea8632c3a92ae5b252280e767898eb95791d, Main run
46150484792 retained three queued Runtime selections after the other scientific
jobs finished. They require offline BW11cpu02, BW10cpu04 and BW10cpu05 services.
A separate A800 acquisition incident cancelled51 jobs before usable execution;
retain those cancellations and missing reports rather than claiming a pass.
No code or runner configuration change can relabel that incident as success.

Change only the existing Runtime matrix recipient scalars:

| Shard | Offline selector | Existing qualified replacement |
| --- | --- | --- |
| 41 | fq-bw1100-11-cpu02-validation | fq-bw1100-11-cpu13-validation |
| 56 | fq-bw1100-10-cpu04-validation | fq-bw1100-10-cpu13-validation |
| 64 | fq-bw1100-10-cpu05-validation | fq-bw1100-10-cpu15-validation |

All three replacements were online and idle in the read-only inventory. Each
stays on its original physical BW host and has the original qualification for
all four locked profiles. Six original API-digest-verified report ZIPs, covering
all old/new services, show the same dedicated finite4CPU/24GiB service admission,
four workers and one numerical thread per worker. BW10cpu14 was considered but
rejected as offline; do not route to it. The original qualification record and
image lock stay immutable; present availability is not a future-health promise.

Preserve every128 index/hosted bucket, all96 distinct BW Runtime recipients,
the first32 A800 placements, all node selections and seven serial coverage
recipients. No test body, count, marker, seed, width, precision, scientific
budget, admission formula, service, label or quota changes. No source-transport
or artifact semantics change is included. Do not add wrappers or fallbacks.

Compare expanded complete workflow YAML allowing only these three scalars.
Execute existing routing/source policy tests and static gates. The resulting
head must independently verify original complete selections, all scientific
call phases, real admission reports, source/install inputs and every applicable
job elapsed time below600s. Retain old cancellations/failures and all misses.
No merge, force push, CI trial rerun, runner administration or Rust.
