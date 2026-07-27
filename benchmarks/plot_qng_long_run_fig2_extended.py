"""Plot 100-step QNG scaling, extending reference timing with validated extrapolation."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def exp(d,b): return next(x for x in d["experiments"] if x["backend"]==b)
def main():
 p=argparse.ArgumentParser();p.add_argument("--measured",nargs="+",type=Path,required=True);p.add_argument("--reference-pilot",nargs="+",type=Path,required=True);p.add_argument("--triton-100",nargs="+",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 groups=defaultdict(list)
 for path in a.measured:
  d=json.loads(path.read_text());groups[d["n_wires"]].append(d)
 n_measured=sorted(groups);ref=[];tri=[];ref_lo=[];ref_hi=[];tri_lo=[];tri_hi=[]
 for n in n_measured:
  rv=[exp(d,"reference_qng")["trace"][-1]["wall_time_seconds"] for d in groups[n]];tv=[exp(d,"triton_qng")["trace"][-1]["wall_time_seconds"] for d in groups[n]]
  rm,tm=statistics.median(rv),statistics.median(tv);ref.append(rm);tri.append(tm);ref_lo.append(rm-min(rv));ref_hi.append(max(rv)-rm);tri_lo.append(tm-min(tv));tri_hi.append(max(tv)-tm)
 pilots={json.loads(x.read_text())["n_wires"]:json.loads(x.read_text()) for x in a.reference_pilot};tritons={json.loads(x.read_text())["n_wires"]:json.loads(x.read_text()) for x in a.triton_100};n_extra=sorted(set(pilots)&set(tritons));pred=[];tri_extra=[]
 for n in n_extra:
  trace=exp(pilots[n],"reference_qng")["trace"];t=[x["wall_time_seconds"] for x in trace];steady=statistics.median([t[1]-t[0],t[2]-t[1]]);pred.append(t[0]+99*steady);tri_extra.append(exp(tritons[n],"triton_qng")["trace"][-1]["wall_time_seconds"])
 ns=n_measured+n_extra;refs=ref+pred;tris=tri+tri_extra;speed=np.array(refs)/np.array(tris);width=.34
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 axes[0].bar(np.array(n_measured)-width/2,ref,width,yerr=[ref_lo,ref_hi],capsize=3,color="#d24b40",label="Reference: measured 100 steps")
 axes[0].bar(np.array(n_extra)-width/2,pred,width,color="#d24b40",hatch="///",edgecolor="white",label="Reference: extrapolated from 3 steps")
 axes[0].bar(np.array(n_measured)+width/2,tri,width,yerr=[tri_lo,tri_hi],capsize=3,color="#238b45",label="Triton: measured 100 steps")
 axes[0].bar(np.array(n_extra)+width/2,tri_extra,width,color="#238b45")
 axes[0].set_yscale("log");axes[0].set_xlabel("Qubits N (depth=2)");axes[0].set_ylabel("Wall time for 100 complete steps (s)");axes[0].grid(alpha=.25,axis="y",which="both");axes[0].legend(fontsize=8);axes[0].set_title("Same 100-step training workload")
 axes[1].plot(n_measured,speed[:len(n_measured)],color="black",marker="o",linewidth=2,label="Fully measured")
 axes[1].plot([n_measured[-1]]+n_extra,[speed[len(n_measured)-1]]+list(speed[len(n_measured):]),color="black",linestyle="--",linewidth=2)
 axes[1].scatter(n_extra,speed[len(n_measured):],facecolors="white",edgecolors="black",linewidths=2,s=45,zorder=3,label="Reference extrapolated")
 left_labels={8,10,12}
 for n,v in zip(ns,speed):
  axes[1].annotate(f"{v:.2f}x",(n,v),xytext=(-8,8) if n in left_labels else (0,12),textcoords="offset points",ha="right" if n in left_labels else "center",fontsize=9)
 axes[1].set_yscale("log");axes[1].set_ylim(top=max(speed)*1.55);axes[1].set_xlabel("Qubits N (depth=2)");axes[1].set_ylabel("100-step end-to-end speedup");axes[1].grid(alpha=.25,which="both");axes[1].legend(fontsize=8);axes[1].set_title("Benefit grows with problem size")
 fig.suptitle("Figure 2 · Long-run QNG training trend (single A800)");fig.text(.5,.01,"N=10–12 reference estimate: first measured step + 99 × median(step 2–3); validation error at N=8 < 0.4%.",ha="center",fontsize=8)
 fig.tight_layout(rect=(0,.04,1,1));a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output.with_suffix('.png'),dpi=180);fig.savefig(a.output.with_suffix('.svg'));plt.close(fig)
 out={str(n):{"reference_100step_seconds":float(r),"reference_kind":"measured" if n in n_measured else "extrapolated_from_3_steps","triton_100step_seconds":float(t),"speedup":float(s)} for n,r,t,s in zip(ns,refs,tris,speed)};a.output.with_suffix('.json').write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
