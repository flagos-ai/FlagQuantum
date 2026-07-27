"""Plot matched HVA optimizer step and wall-time convergence."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
LABELS={"adam":"Adam","reference_qng":"Reference block-QNG","triton_qng":"Triton block-QNG"}
def main():
 p=argparse.ArgumentParser();p.add_argument("result",type=Path);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();data=json.loads(a.result.read_text());a.output_dir.mkdir(parents=True,exist_ok=True)
 for x,key in (("optimizer_step","optimizer_step"),("wall_time_seconds","wall_time_seconds")):
  fig,axis=plt.subplots(figsize=(8.4,5.0))
  for e in data["experiments"]:axis.plot([r[key] for r in e["trace"]],[r["relative_error"] for r in e["trace"]],linewidth=2,label=LABELS[e["backend"]])
  axis.axhline(data["convergence_tolerance"],color="black",linestyle="--",label="1e-5 target");axis.set_yscale("log");axis.set_xlabel("Optimizer step" if key=="optimizer_step" else "Wall time (s)");axis.set_ylabel("Relative energy error");axis.set_title(f'N={data["n_wires"]}, depth={data["depth"]} HVA time-to-accuracy');axis.grid(alpha=.25,which="both");axis.legend();fig.tight_layout();fig.savefig(a.output_dir/f"{x}_relative_error.svg");fig.savefig(a.output_dir/f"{x}_relative_error.png",dpi=180);plt.close(fig)
if __name__=="__main__":main()
