import json
from . import d7_c1_orientation_preflight as d
def main():
 rows=[d.run_one(x,s)for x in d.DATASETS for s in d.SEEDS];r={'rows':rows,'gates':d.gates(rows)};d.ROOT.mkdir(parents=True,exist_ok=True);(d.ROOT/'d7_c1_stage0_summary.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');(d.ROOT/'d7_c1_stage0_summary.txt').write_text('D7-C1 Stage0 '+r['gates']['stage0']+'\n');print('[D7-C1] Stage 0 completed',flush=True)
if __name__=='__main__':main()
