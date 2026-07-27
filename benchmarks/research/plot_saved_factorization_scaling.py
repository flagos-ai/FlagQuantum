"""Plot saved-factorization reverse and end-to-end scaling."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt

def metrics(path):
 d=json.loads(path.read_text());rows=[r["training"]["step_metrics"][0] for r in d["rank_records"]]
 return {"n":d["n_sites"],"forward":max(x["forward_seconds"] for x in rows),"reverse":max(x["reverse_seconds"] for x in rows),"e2e":max(x["end_to_end_seconds"] for x in rows),"peak_gib":max(r["peak_memory_bytes"] for r in d["rank_records"])/2**30,"energy":d["best_variational_energy"]}
def main():
 p=argparse.ArgumentParser();p.add_argument("--recompute",nargs="+",type=Path,required=True);p.add_argument("--saved",nargs="+",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 aa={x["n"]:x for x in map(metrics,a.recompute)};bb={x["n"]:x for x in map(metrics,a.saved)};ns=sorted(set(aa)&set(bb));rev=[aa[n]["reverse"]/bb[n]["reverse"] for n in ns];e2e=[aa[n]["e2e"]/bb[n]["e2e"] for n in ns];mem=[bb[n]["peak_gib"]-aa[n]["peak_gib"] for n in ns]
 fig,axes=plt.subplots(1,2,figsize=(11,4.8));w=6.0
 axes[0].bar([n-w/2 for n in ns],[aa[n]["e2e"] for n in ns],w,label="Recompute factorization",color="#d24b40");axes[0].bar([n+w/2 for n in ns],[bb[n]["e2e"] for n in ns],w,label="Save factorization",color="#238b45");axes[0].set_xlabel("MPS sites / qubits N");axes[0].set_ylabel("One complete training step (s)");axes[0].set_title("Measured end-to-end training time");axes[0].grid(alpha=.25,axis="y");axes[0].legend(fontsize=8)
 axes[1].axhline(1,color="black",linestyle="--",linewidth=1);axes[1].plot(ns,rev,marker="s",linewidth=2,label="Reverse speedup",color="#6a3d9a");axes[1].plot(ns,e2e,marker="o",linewidth=2,label="End-to-end speedup",color="#2468b4")
 for n,v in zip(ns,e2e):axes[1].annotate(f"{v:.2f}x",(n,v),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
 axes[1].set_xlabel("MPS sites / qubits N");axes[1].set_ylabel("Speedup (recompute / saved)");axes[1].set_title("Benefit after paying the cache cost");axes[1].grid(alpha=.25);axes[1].legend(fontsize=8,loc="upper left");other=axes[1].twinx();other.plot(ns,mem,marker="^",linestyle=":",color="#d98c10",label="Extra peak memory");other.set_ylabel("Additional peak memory / rank (GiB)",color="#a96b00");other.tick_params(axis="y",labelcolor="#a96b00")
 fig.suptitle("Saved-factorization scaling · p=2 · χ=512 · 8×A800 · one sharded MPS training step");fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output.with_suffix('.png'),dpi=180);fig.savefig(a.output.with_suffix('.svg'));plt.close(fig)
 out={str(n):{"recompute":aa[n],"saved":bb[n],"reverse_speedup":rev[i],"end_to_end_speedup":e2e[i],"extra_peak_memory_gib_per_rank":mem[i],"energy_abs_difference":abs(aa[n]["energy"]-bb[n]["energy"])} for i,n in enumerate(ns)};a.output.with_suffix('.json').write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
