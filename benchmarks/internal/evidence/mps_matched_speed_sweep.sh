#!/usr/bin/env bash
# Time every rung of the frozen matched-speed ladder at every world size named.
#
# The release contract's speed threshold is a ratio, and a ratio at one world
# size cannot say whether the shortfall belongs to the algorithm or to the
# implementation: a per-step cost that does not shard and a per-site cost that
# does both grow with the rung and both fall with the world, so measuring at a
# single world size confounds them.  Timing every rung at two or more world sizes
# separates them, because one device pays `serial + shardable` per site and `N`
# devices pay `serial + shardable / N`, which is two equations for the two terms.
#
# `benchmarks/build_mps_shardability_calibration.py` performs that fit.  This
# driver only produces the records it reads.
#
# A world is split evenly across two hosts: `world / 2` devices per host, the low
# ranks on the launch host and the rest on the peer, because the frozen contract
# names a two-host topology and a release world of sixteen.  World one stays on
# the launch host, where it belongs: one device is one rank and has no topology to
# choose.
#
# Nothing here is a result.  The probe writes one record per leg, and every leg's
# record is checked for one entry per rank before the leg is called complete,
# because a rank that died early leaves the others waiting rather than shortening
# the list and a record three quarters full is not a measurement of that world.
#
# Environment:
#   FQ_SWEEP_OUTPUT           directory the records are written to (required)
#   FQ_SWEEP_PYTHON           interpreter that can import flagquantum
#   FQ_SWEEP_ROOT             checkout the ranks import from
#   FQ_SWEEP_PEER             hostname of the second host
#   FQ_SWEEP_PEER_ADDRESS     address the peer's ranks rendezvous on
#   FQ_SWEEP_MASTER_ADDRESS   address rank zero listens on
#   FQ_SWEEP_INTERFACE        network interface the collectives bind to
#   FQ_SWEEP_HCA              comma-separated InfiniBand devices
#   FQ_SWEEP_VISIBLE          devices each host exposes
#   FQ_SWEEP_WORLDS           world sizes to time, in the order to time them
#   FQ_SWEEP_RUNGS            ladder rungs to time, in the order to time them
#   FQ_SWEEP_BASE_PORT        first rendezvous port; one port per leg
#   FQ_SWEEP_IDLE_LIMIT_MIB   per-device occupancy above which the sweep waits
#   FQ_SWEEP_TIMEOUT_SECONDS  budget for one leg
set -u

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)

python=${FQ_SWEEP_PYTHON:-python3}
root=${FQ_SWEEP_ROOT:-$repo}
probe=${FQ_SWEEP_PROBE:-$here/mps_matched_speed_sweep.py}
manifest=${FQ_SWEEP_MANIFEST:-$root/benchmarks/manifests/mps_release_v1.json}
peer=${FQ_SWEEP_PEER:-jp-a800-171}
peer_address=${FQ_SWEEP_PEER_ADDRESS:-10.1.15.171}
master_address=${FQ_SWEEP_MASTER_ADDRESS:-10.1.15.172}
interface=${FQ_SWEEP_INTERFACE:-ens22f0}
hca=${FQ_SWEEP_HCA:-mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107,mlx5_108}
visible=${FQ_SWEEP_VISIBLE:-0,1,2,3,4,5,6,7}
worlds=${FQ_SWEEP_WORLDS:-1 8 16}
rungs=${FQ_SWEEP_RUNGS:-"matched_speed_512sites_chi64 matched_speed_1024sites_chi64 \
matched_speed_2048sites_chi64 matched_speed_4096sites_chi64 \
matched_speed_8192sites_chi64"}
base_port=${FQ_SWEEP_BASE_PORT:-31971}
limit_mib=${FQ_SWEEP_IDLE_LIMIT_MIB:-1024}
leg_timeout=${FQ_SWEEP_TIMEOUT_SECONDS:-1800}

: "${FQ_SWEEP_OUTPUT:?set FQ_SWEEP_OUTPUT to the directory the records land in}"
outdir=$FQ_SWEEP_OUTPUT
probe_name=$(basename "$probe")
probe_pattern="[${probe_name:0:1}]${probe_name:1}"

export PYTHONPATH=$root
export FQ_REPOSITORY_ROOT=$root
export GLOO_SOCKET_IFNAME=$interface
export NCCL_SOCKET_IFNAME=$interface
export NCCL_IB_HCA=$hca
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

mkdir -p "$outdir"
lock="$outdir/leg.lock"
if ! mkdir "$lock" 2>/dev/null; then
  echo "another sweep holds $lock"
  exit 3
fi
trap 'rmdir "$lock"' EXIT

# Report every device of both hosts that is above the limit, or say nothing.
idle_check() {
  local host line index used mib bad=0
  for host in local peer; do
    if [ "$host" = local ]; then
      line=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
    else
      line=$(ssh -o ConnectTimeout=40 "$peer" \
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader") || return 1
    fi
    while read -r index used; do
      [ -n "${index:-}" ] || continue
      mib=$(printf '%s' "$used" | tr -dc '0-9')
      if [ "${mib:-0}" -gt "$limit_mib" ]; then
        echo "BUSY $host device$index ${mib}MiB"
        bad=1
      fi
    done <<<"$line"
  done
  return "$bad"
}

# A leg that has just exited keeps its device for a few seconds while the CUDA
# context tears down, so asking once would either refuse a device that is free by
# the time its ranks start or, worse, race a neighbour's job and record a timing
# taken beside it.  A measurement taken beside a co-tenant is not evidence, so the
# sweep waits for the devices to settle and stops rather than skipping a leg it
# could not measure: a missing rung is indistinguishable from one that was never
# asked for, and the fit needs every rung it was given.
settle() {
  local attempt
  for attempt in $(seq 1 12); do
    if idle_check; then
      return 0
    fi
    echo "waiting for the devices to settle (attempt $attempt)"
    sleep 10
  done
  return 1
}

# Every rank of the leg must appear in the published record.
complete() {
  "$python" - "$1" "$2" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
ranks = record.get("ranks") or []
raise SystemExit(0 if len(ranks) == int(sys.argv[2]) else 1)
PY
}

overall=0
port=$base_port
for world in $worlds; do
  per=$((world / 2))
  [ "$per" -lt 1 ] && per=1
  for rung in $rungs; do
    label="${rung}_w${world}"
    record="$outdir/$label.json"
    port=$((port + 1))
    echo "=== leg $label world=$world per=$per port=$port $(date -Is)"
    pkill -f "$probe_pattern" 2>/dev/null
    ssh -o ConnectTimeout=40 "$peer" "pkill -f '$probe_pattern'" 2>/dev/null
    sleep 5
    if ! settle; then
      echo "refusing: the devices never went idle for $label"
      overall=1
      break 2
    fi
    start=$(date +%s)
    if [ "$per" -lt "$world" ]; then
      remote=""
      for ((r = per; r < world; r++)); do
        remote+="RANK=$r WORLD_SIZE=$world LOCAL_RANK=$((r - per)) LOCAL_WORLD_SIZE=$per \
CUDA_VISIBLE_DEVICES=$visible MASTER_ADDR=$master_address MASTER_PORT=$port \
PYTHONPATH=$root GLOO_SOCKET_IFNAME=$interface NCCL_SOCKET_IFNAME=$interface \
NCCL_IB_HCA=$hca OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 FQ_REPOSITORY_ROOT=$root \
$python -u $probe --release-manifest $manifest --rung $rung --output $record \
> $outdir/$label.rank$r.log 2>&1 & "
      done
      timeout "$leg_timeout" ssh -o ConnectTimeout=40 "$peer" \
        "cd $root && $remote wait" >"$outdir/$label.remote.log" 2>&1 &
      ssh_pid=$!
    else
      ssh_pid=""
    fi
    pids=()
    for ((r = 0; r < per; r++)); do
      (
        cd "$root" || exit 1
        RANK=$r WORLD_SIZE=$world LOCAL_RANK=$r LOCAL_WORLD_SIZE=$per \
        CUDA_VISIBLE_DEVICES=$visible \
        MASTER_ADDR=$master_address MASTER_PORT=$port \
        "$python" -u "$probe" --release-manifest "$manifest" --rung "$rung" \
          --output "$record" >"$outdir/$label.rank$r.log" 2>&1
      ) &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    if [ -n "$ssh_pid" ]; then wait "$ssh_pid" || true; fi
    end=$(date +%s)
    # The probe publishes from rank zero alone and prints on no other rank, so
    # the record itself is what a leg is judged by.
    rc=0
    if [ ! -f "$record" ] || ! complete "$record" "$world"; then
      echo "the leg published no record carrying every rank of world $world"
      rc=1
    fi
    echo "leg $label rc=$rc seconds=$((end - start)) $(date -Is)"
    if [ "$rc" -ne 0 ]; then
      overall=1
      echo "--- rank0 tail"
      tail -5 "$outdir/$label.rank0.log"
    fi
  done
done
echo "=== all legs done overall=$overall $(date -Is)"
if settle; then
  echo "devices idle after"
else
  echo "devices busy after"
  overall=1
fi
exit "$overall"
