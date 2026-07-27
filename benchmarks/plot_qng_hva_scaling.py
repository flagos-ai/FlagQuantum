"""Plot Triton block-QNG scaling from matched raw artifacts."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
def main():
 p=argparse.ArgumentParser();p.add_argument("inputs",nargs="+",type=Path);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();rows=sorted((json.loads(x.read_text()) for x in a.inputs),key=lambda x:x["n_wires"]);a.output_dir.mkdir(parents=True,exist_ok=True);n=[r["n_wires"] for r in rows]
 fig,axis=plt.subplots(figsize=(8.4,5));
 for key,label in (("tangent","State + tangents"),("geometry","Block geometry"),("qng_direction","Full QNG direction")):axis.plot(n,[r["speedups"][key] for r in rows],marker="o",linewidth=2,label=label)
 axis.set_yscale("log");axis.set_xlabel("Qubits N (depth=2)");axis.set_ylabel("Vectorized reverse / Triton speedup");axis.set_title("A800 Triton block-QNG speedup scaling");axis.grid(alpha=.25,which="both");axis.legend();fig.tight_layout();fig.savefig(a.output_dir/"qng_speedup_scaling.svg");fig.savefig(a.output_dir/"qng_speedup_scaling.png",dpi=180);plt.close(fig)
 fig,axis=plt.subplots(figsize=(8.4,5));values=[r["speedups"]["qng_direction"] for r in rows];axis.plot(n,values,marker="o",linewidth=2.5,color="#238b45")
 for x,y,row in zip(n,values,rows):axis.annotate(f"{y:.1f}x"+("*" if row["repeats"]==1 else ""),(x,y),textcoords="offset points",xytext=(0,8),ha="center")
 axis.set_yscale("log");axis.set_xlabel("Qubits N (depth=2)");axis.set_ylabel("Full QNG direction speedup");axis.set_title("A800 Triton block-QNG acceleration grows with state size");axis.grid(alpha=.25,which="both");axis.text(.02,.03,"* N=12 uses one bounded sample; N=2–10 use median of 3",transform=axis.transAxes,fontsize=9);fig.tight_layout();fig.savefig(a.output_dir/"qng_full_direction_speedup_scaling.svg");fig.savefig(a.output_dir/"qng_full_direction_speedup_scaling.png",dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for key,label in (("reference_qng_direction","Reference QNG"),("triton_qng_direction","Triton QNG")):axes[0].plot(n,[r["results"][key]["median_seconds"] for r in rows],marker="o",linewidth=2,label=label);axes[1].plot(n,[r["results"][key]["peak_memory_bytes"]/2**20 for r in rows],marker="o",linewidth=2,label=label)
 axes[0].set_yscale("log");axes[0].set_ylabel("Full QNG direction time (s)");axes[1].set_yscale("log");axes[1].set_ylabel("Peak allocated memory (MiB)")
 for axis in axes:axis.set_xlabel("Qubits N (depth=2)");axis.grid(alpha=.25,which="both");axis.legend()
 fig.suptitle("A800 block-QNG time and memory scaling");fig.tight_layout();fig.savefig(a.output_dir/"qng_time_memory_scaling.svg");fig.savefig(a.output_dir/"qng_time_memory_scaling.png",dpi=180)
if __name__=="__main__":main()
