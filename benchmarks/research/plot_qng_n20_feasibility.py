"""Plot measured Triton QNG scaling and reference feasibility through 20 qubits."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument("inputs",nargs="+",type=Path);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    payloads=sorted((json.loads(x.read_text()) for x in a.inputs),key=lambda x:x["n_wires"])
    n=[x["n_wires"] for x in payloads];e=[x["experiments"][0] for x in payloads]
    wall=[x["trace"][-1]["wall_time_seconds"] for x in e];memory=[x["peak_memory_bytes"]/2**30 for x in e]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    time_line=axes[0].plot(n,wall,color="#238b45",marker="o",linewidth=2,label="100-step wall time")[0]
    axes[0].set_xlabel("Qubits N (depth=2)");axes[0].set_ylabel("Triton QNG wall time for 100 steps (s)");axes[0].grid(alpha=.25)
    other=axes[0].twinx();memory_line=other.plot(n,memory,color="#6a3d9a",marker="s",linewidth=2,label="Peak memory")[0];other.set_yscale("log");other.set_ylabel("Peak allocated memory (GiB)");axes[0].legend([time_line,memory_line],["100-step wall time","Peak memory"],loc="upper left")
    for nx,t in zip(n,wall):axes[0].annotate(f"{t:.1f}s",(nx,t),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
    status=["3 steps\n51.6 s","3 steps\n232.5 s","OOM\n(256 GiB)","OOM","OOM","OOM"]
    colors=["#d9a441","#d9a441"]+["#d24b40"]*4
    axes[1].bar(n,[1]*len(n),width=1.25,color=colors)
    for nx,label in zip(n,status):axes[1].text(nx,0.5,label,ha="center",va="center",fontsize=8,color="white",fontweight="bold")
    axes[1].set_ylim(0,1);axes[1].set_yticks([]);axes[1].set_xlabel("Qubits N (depth=2)");axes[1].set_title("Reference reverse-Jacobian boundary")
    axes[1].text(11,.96,"Runnable but slow",ha="center",va="top",fontsize=8,color="white");axes[1].text(17,.96,"Cannot complete one step",ha="center",va="top",fontsize=8,color="white")
    fig.suptitle("QNG feasibility through 20 qubits (single A800)");fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output.with_suffix('.png'),dpi=180);fig.savefig(a.output.with_suffix('.svg'));plt.close(fig)
    summary={str(nx):{"triton_100step_seconds":t,"triton_peak_gib":m,"reference_probe":s} for nx,t,m,s in zip(n,wall,memory,status)}
    a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__":main()
