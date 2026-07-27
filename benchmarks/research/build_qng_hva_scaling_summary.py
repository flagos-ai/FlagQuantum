"""Build aligned JSON/CSV from block-QNG scaling artifacts."""
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("inputs",nargs="+",type=Path);p.add_argument("--json-output",type=Path,required=True);p.add_argument("--csv-output",type=Path,required=True);a=p.parse_args();data=sorted((json.loads(x.read_text()) for x in a.inputs),key=lambda x:x["n_wires"]);rows=[]
 for x in data:
  r=x["results"];rows.append({"n_wires":x["n_wires"],"state_dimension":x["state_dimension"],"depth":x["depth"],"parameter_count":x["parameter_count"],"repeats":x["repeats"],"reference_qng_seconds":r["reference_qng_direction"]["median_seconds"],"triton_qng_seconds":r["triton_qng_direction"]["median_seconds"],"qng_direction_speedup":x["speedups"]["qng_direction"],"reference_peak_mib":r["reference_qng_direction"]["peak_memory_bytes"]/2**20,"triton_peak_mib":r["triton_qng_direction"]["peak_memory_bytes"]/2**20,"geometry_speedup":x["speedups"]["geometry"],"tangent_speedup":x["speedups"]["tangent"]})
 payload={"schema":"flagquantum.qng_hva_scaling_summary.v1","execution_semantics":"single_device_fast_path","scalability_claim_allowed":False,"device":data[0]["device"],"dtype":data[0]["dtype"],"fixed_depth":data[0]["depth"],"note":"N=12 is one bounded sample; N=2-10 are medians of three steady samples.","rows":rows};a.json_output.parent.mkdir(parents=True,exist_ok=True);a.json_output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");a.csv_output.parent.mkdir(parents=True,exist_ok=True)
 with a.csv_output.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=tuple(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
