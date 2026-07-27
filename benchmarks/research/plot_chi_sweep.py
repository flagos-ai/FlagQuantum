"""Plot measured chi sweep artifacts for a fixed Heisenberg workload."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--directory",type=Path,required=True); ap.add_argument("--prefix",default="chi_sweep_n32_p16"); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    rows=[]
    for chi in (128,256,512,768):
        p=a.directory/f"{a.prefix}_chi{chi}.json"
        if chi == 128 and not p.exists():
            p=a.directory/f"chi_probe_n32_p16_chi{chi}.json"
        if not p.exists(): continue
        d=json.loads(p.read_text()); r=d["rank_records"][0]; m=r["training"]["step_metrics"][0]
        rows.append((chi,r["elapsed_seconds"],m["forward_seconds"],m["reverse_seconds"],r["peak_memory_bytes"]/2**30,m["largest_realized_bond"],m["discarded_weight"]))
    if not rows: raise SystemExit("no sweep artifacts")
    x=[r[0] for r in rows]; fig,ax=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    ax[0].plot(x,[r[1] for r in rows],"o-",label="total"); ax[0].plot(x,[r[2] for r in rows],"o-",label="forward"); ax[0].plot(x,[r[3] for r in rows],"o-",label="reverse"); ax[0].set_xlabel("requested χ"); ax[0].set_ylabel("seconds"); ax[0].legend(frameon=False); ax[0].grid(True)
    ax[1].plot(x,[r[4] for r in rows],"o-",color="#DC2626"); ax[1].set_xlabel("requested χ"); ax[1].set_ylabel("peak GiB"); ax[1].grid(True)
    ax[2].plot(x,[r[5] for r in rows],"o-",label="realized bond"); ax[2].plot(x,[r[6] for r in rows],"o-",label="discarded weight"); ax[2].set_xlabel("requested χ"); ax[2].set_yscale("log"); ax[2].legend(frameon=False); ax[2].grid(True)
    fig.suptitle("FlagQuantum distributed MPS χ sweep",weight="bold")
    a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output.with_suffix(".png"),dpi=220); fig.savefig(a.output.with_suffix(".svg")); fig.savefig(a.output.with_suffix(".pdf")); print(json.dumps({"points":len(rows),"output":str(a.output)}))
if __name__=="__main__": main()
