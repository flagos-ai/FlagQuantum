"""Plot three-seed HVA convergence and time-to-accuracy summary."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
LABELS={"adam":"Adam","reference_qng":"Reference block-QNG","triton_qng":"Triton block-QNG"}
def main():
 p=argparse.ArgumentParser();p.add_argument("summary",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args();d=json.loads(a.summary.read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for backend,m in d["methods"].items():
  c=m["step_curve"];x=[r["optimizer_step"] for r in c];y=[r["median_relative_error"] for r in c];lo=[r["min_relative_error"] for r in c];hi=[r["max_relative_error"] for r in c];axes[0].plot(x,y,linewidth=2,label=LABELS[backend]);axes[0].fill_between(x,lo,hi,alpha=.16)
 axes[0].axhline(d["convergence_tolerance"],color="black",linestyle="--",label="1e-5 target");axes[0].set_yscale("log");axes[0].set_xlabel("Optimizer step");axes[0].set_ylabel("Relative energy error");axes[0].grid(alpha=.25,which="both");axes[0].legend(fontsize=8)
 q=["reference_qng","triton_qng"];vals=[d["methods"][x]["time_to_1e-5_seconds"]["median"] for x in q];bars=axes[1].bar([LABELS[x] for x in q],vals);axes[1].set_ylabel("Median time to 1e-5 (s)");axes[1].grid(alpha=.2,axis="y");axes[1].text(1,vals[1],f'{d["median_time_to_1e-5_speedup"]:.2f}x',ha="center",va="bottom")
 fig.suptitle(f'N={d["n_wires"]}, depth={d["depth"]} HVA on A800 (median and 3-seed range)');fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output.with_suffix(".svg"));fig.savefig(a.output.with_suffix(".png"),dpi=180)
if __name__=="__main__":main()
