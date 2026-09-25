import argparse,json
from . import d6_a1_utility_mass_pilot as p
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--device",required=True);a.add_argument("--full-gt-path",required=True);x=a.parse_args(); rows=[p.run_seed(s,x.device,x.full_gt_path) for s in p.SEEDS]; r=p.summarize(rows); p.ROOT.mkdir(parents=True,exist_ok=True); (p.ROOT/"d6_a1_summary.json").write_text(json.dumps(r,indent=2)+"\n"); (p.ROOT/"d6_a1_summary.txt").write_text(json.dumps(r,indent=2)+"\n"); (p.ROOT/"source_sha256.json").write_text(json.dumps({"module":p._sha(p.__file__)},indent=2)+"\n"); [print("[D6-A1] GATE %s = %s"%(k,"PASS" if v else "FAIL"),flush=True) for k,v in r["gates"].items()]; print("[D6-A1] FINAL DECISION = "+r["final_decision"],flush=True)
