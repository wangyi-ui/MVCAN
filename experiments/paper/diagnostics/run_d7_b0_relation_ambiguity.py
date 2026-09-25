import json,hashlib
from pathlib import Path
from . import d7_b0_relation_ambiguity as d
def main():
 rows=[d.run_one(x,s) for x in d.DATASETS for s in d.SEEDS];report={"rows":rows,"gates":d.gates(rows),"no_gt":True,"no_training":True};d.ROOT.mkdir(parents=True,exist_ok=True);(d.ROOT/"d7_b0_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");(d.ROOT/"d7_b0_summary.txt").write_text("D7-B0 "+report['gates']['final']+"\n");(d.ROOT/"source_sha256.json").write_text(json.dumps({str(Path(d.__file__)):hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest()},indent=2)+"\n");print('[D7-B0] completed 9 read-only cells',flush=True)
if __name__=='__main__':main()
