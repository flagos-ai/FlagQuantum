"""Aggregate repeated HVA time-to-accuracy runs."""
import argparse,json,statistics
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("inputs",nargs="+",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args();runs=[json.loads(x.read_text()) for x in a.inputs];backends=[e["backend"] for e in runs[0]["experiments"]];methods={}
 for backend in backends:
  es=[next(e for e in run["experiments"] if e["backend"]==backend) for run in runs];times=[e["time_to_1e-5_seconds"] for e in es if e["time_to_1e-5_seconds"] is not None];best=[e["best_relative_error"] for e in es];totals=[e["trace"][-1]["wall_time_seconds"] for e in es]
  hit_errors=[next(p["relative_error"] for p in e["trace"] if p["relative_error"]<=runs[0]["convergence_tolerance"]) for e in es if e["time_to_1e-5_seconds"] is not None]
  methods[backend]={"converged_seeds":len(times),"total_seeds":len(es),"time_to_1e-5_seconds":{"values":times,"median":statistics.median(times) if times else None},"accuracy_at_target":{"values":hit_errors,"median":statistics.median(hit_errors) if hit_errors else None,"min":min(hit_errors) if hit_errors else None,"max":max(hit_errors) if hit_errors else None},"best_relative_error":{"values":best,"median":statistics.median(best)},"total_time_seconds":{"values":totals,"median":statistics.median(totals)},"step_curve":[{"optimizer_step":i+1,"median_wall_time_seconds":statistics.median(e["trace"][i]["wall_time_seconds"] for e in es),"median_relative_error":statistics.median(e["trace"][i]["relative_error"] for e in es),"min_relative_error":min(e["trace"][i]["relative_error"] for e in es),"max_relative_error":max(e["trace"][i]["relative_error"] for e in es)} for i in range(len(es[0]["trace"]))]}
 out={"schema":"flagquantum.qng_hva_time_to_accuracy_summary.v1","device":runs[0]["device"],"dtype":runs[0]["dtype"],"n_wires":runs[0]["n_wires"],"depth":runs[0]["depth"],"seeds":[r["seed"] for r in runs],"convergence_tolerance":runs[0]["convergence_tolerance"],"methods":methods};ref=methods["reference_qng"]["time_to_1e-5_seconds"]["median"];tri=methods["triton_qng"]["time_to_1e-5_seconds"]["median"];out["median_time_to_1e-5_speedup"]=ref/tri
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
