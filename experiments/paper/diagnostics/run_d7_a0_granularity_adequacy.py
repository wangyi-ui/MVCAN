import hashlib,json
from pathlib import Path
from . import d7_a0_granularity_adequacy as d
def main():
 rows=[d.run_one(dataset,seed) for dataset in d.DATASETS for seed in d.SEEDS]
 report={"rows":rows,"gates":d.gates(rows),"no_gt":True,"no_training":True,"frozen_v1_only":True}
 d.ROOT.mkdir(parents=True,exist_ok=True)
 (d.ROOT/"d7_a0_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 (d.ROOT/"d7_a0_summary.txt").write_text("D7-A0 "+report["gates"]["final"]+"\n")
 source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(d.__file__),Path(__file__)]}
 (d.ROOT/"source_sha256.json").write_text(json.dumps(source,indent=2,sort_keys=True)+"\n")
 print("[D7-A0] completed 9 read-only diagnostic cells",flush=True)
if __name__=="__main__": main()
