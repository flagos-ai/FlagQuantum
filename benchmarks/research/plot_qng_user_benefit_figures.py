"""Generate the four end-user Triton-QNG benefit figures."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
LABELS={"adam":"Adam","reference_qng":"Reference block-QNG","triton_qng":"Triton block-QNG"};COLORS={"adam":"#2468b4","reference_qng":"#d24b40","triton_qng":"#238b45"}
def experiment(p,b):return next(e for e in p["experiments"] if e["backend"]==b)
def save(fig,path):fig.tight_layout();fig.savefig(path.with_suffix(".svg"));fig.savefig(path.with_suffix(".png"),dpi=180);plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument("--time-summary",type=Path,required=True);p.add_argument("--fixed-step",nargs="+",type=Path,required=True);p.add_argument("--fixed-budget",nargs="+",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);s=json.loads(a.time_summary.read_text());fixed=sorted((json.loads(x.read_text()) for x in a.fixed_step),key=lambda x:x["n_wires"]);budget=sorted((json.loads(x.read_text()) for x in a.fixed_budget),key=lambda x:x["n_wires"])
 # Figure 1: wall-time convergence plus time-to-target.
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for b,m in s["methods"].items():
  c=m["step_curve"];x=[r["median_wall_time_seconds"] for r in c];y=[r["median_relative_error"] for r in c];lo=[r["min_relative_error"] for r in c];hi=[r["max_relative_error"] for r in c];axes[0].plot(x,y,linewidth=2,label=LABELS[b],color=COLORS[b]);axes[0].fill_between(x,lo,hi,alpha=.15,color=COLORS[b])
 axes[0].axhline(s["convergence_tolerance"],color="black",linestyle="--",label="1e-5 target");axes[0].set_yscale("log");axes[0].set_xlabel("Cumulative wall time (s)");axes[0].set_ylabel("Relative energy error");axes[0].grid(alpha=.25,which="both");axes[0].legend(fontsize=8)
 bs=["reference_qng","triton_qng"];vals=[s["methods"][b]["accuracy_at_target"]["median"] for b in bs];lo=[vals[i]-s["methods"][b]["accuracy_at_target"]["min"] for i,b in enumerate(bs)];hi=[s["methods"][b]["accuracy_at_target"]["max"]-vals[i] for i,b in enumerate(bs)];axes[1].bar([LABELS[b] for b in bs],vals,yerr=[lo,hi],capsize=4,color=[COLORS[b] for b in bs]);axes[1].axhline(s["convergence_tolerance"],color="black",linestyle="--",label="1e-5 target");axes[1].set_ylim(0,1.15*s["convergence_tolerance"]);axes[1].set_ylabel("Relative error at first target hit");axes[1].set_title("Accuracy-aligned comparison");axes[1].grid(alpha=.2,axis="y");axes[1].legend(fontsize=8);fig.suptitle("Figure 1 · Time to useful accuracy (A800, 3 seeds)");save(fig,a.output_dir/"fig1_time_to_accuracy")
 # Figure 2: fixed ten-step end-to-end time and speedup.
 n=[x["n_wires"] for x in fixed];ref=[experiment(x,"reference_qng")["trace"][-1]["wall_time_seconds"] for x in fixed];tri=[experiment(x,"triton_qng")["trace"][-1]["wall_time_seconds"] for x in fixed];speed=[r/t for r,t in zip(ref,tri)];fig,axis=plt.subplots(figsize=(8.8,5));width=.34;axis.bar([x-width/2 for x in n],ref,width,label="Reference block-QNG",color=COLORS["reference_qng"]);axis.bar([x+width/2 for x in n],tri,width,label="Triton block-QNG",color=COLORS["triton_qng"]);axis.set_yscale("log");axis.set_xlabel("Qubits N (depth=2)");axis.set_ylabel("Wall time for 10 complete training steps (s)");axis.grid(alpha=.25,axis="y",which="both");axis.legend();other=axis.twinx();other.plot(n,speed,color="black",marker="o",linewidth=2,label="End-to-end speedup");other.set_ylabel("End-to-end speedup");
 for x,y in zip(n,speed):other.annotate(f"{y:.2f}x",(x,y),xytext=(0,7),textcoords="offset points",ha="center");axis.set_title("Figure 2 · Time for the same user-visible training work");save(fig,a.output_dir/"fig2_fixed_10step_time")
 # Figure 3: complete-loop peak memory.
 refm=[experiment(x,"reference_qng")["peak_memory_bytes"]/2**20 for x in fixed];trim=[experiment(x,"triton_qng")["peak_memory_bytes"]/2**20 for x in fixed];fig,axis=plt.subplots(figsize=(8.6,5));axis.plot(n,refm,marker="o",linewidth=2,label="Reference block-QNG",color=COLORS["reference_qng"]);axis.plot(n,trim,marker="o",linewidth=2,label="Triton block-QNG",color=COLORS["triton_qng"]);axis.set_yscale("log");axis.set_xlabel("Qubits N (depth=2)");axis.set_ylabel("Peak allocated memory for training loop (MiB)");axis.set_title("Figure 3 · End-to-end training memory");axis.grid(alpha=.25,which="both");axis.legend();save(fig,a.output_dir/"fig3_training_peak_memory")
 # Figure 4: completed updates and best error inside ten seconds.
 fig,axes=plt.subplots(1,2,figsize=(11,4.8));width=.24;backends=["adam","reference_qng","triton_qng"]
 for j,b in enumerate(backends):
  done=[];errors=[];exact=[]
  for x in budget:
   e=experiment(x,b);eligible=[r for r in e["trace"] if r["wall_time_seconds"]<=x["wall_budget_seconds"]];done.append(len(eligible));errors.append(min((r["relative_error"] for r in eligible),default=float("nan")))
  exact.append(bool(errors[-1] == 0.0))
  errors=[1e-12 if value == 0.0 else value for value in errors]
  positions=[x["n_wires"]+(j-1)*width for x in budget];axes[0].bar(positions,done,width,label=LABELS[b],color=COLORS[b]);axes[1].plot(n,errors,marker="o",linewidth=2,label=LABELS[b],color=COLORS[b])
  for nx,is_exact in zip(n,exact):
   if is_exact: axes[1].annotate("exact",(nx,1e-12),xytext=(0,5),textcoords="offset points",ha="center",fontsize=7,color=COLORS[b])
 axes[0].set_xlabel("Qubits N (depth=2)");axes[0].set_ylabel("Complete optimizer steps within 10 s");axes[0].legend(fontsize=8);axes[0].grid(alpha=.2,axis="y");axes[1].set_yscale("log");axes[1].set_xlabel("Qubits N (depth=2)");axes[1].set_ylabel("Best relative error within 10 s");axes[1].grid(alpha=.25,which="both");axes[1].legend(fontsize=8);fig.suptitle("Figure 4 · Optimization progress under the same 10-second GPU budget");save(fig,a.output_dir/"fig4_fixed_10s_budget")
if __name__=="__main__":main()
