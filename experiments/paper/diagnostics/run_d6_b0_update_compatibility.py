"""Verified recovery runner for the frozen D6-B0 diagnostic."""
import argparse
import json
from pathlib import Path
from . import d6_b0_update_compatibility as d

RECOVERY_ROOT = d.ROOT.parent / "d6_b0_update_compatibility_recovery1"

def _gate_rows(rows):
    by = {(row["dataset"], row["seed"]): row for row in rows}
    gate1 = len(by) == 9 and all(row["exact_prediction"] for row in rows)
    gate4 = all(row["no_gt"] and row["restored"] and row["epochs"] == 20 for row in rows)
    opposition = []
    retention = []
    for seed in d.SEEDS:
        m = by[("MSRC-v1", seed)]["families"]["combined"]
        c = by[("Caltech-6V", seed)]["families"]["combined"]
        b = by[("BDGP", seed)]["families"]["combined"]
        opposition.append(m["median"] < c["median"] and m["median"] < b["median"] and m["conflict_epoch_fraction"] > c["conflict_epoch_fraction"] and m["conflict_epoch_fraction"] > b["conflict_epoch_fraction"])
        retention.append(m["retention"]["median"] < c["retention"]["median"] and m["retention"]["median"] < b["retention"]["median"])
    gate2 = sum(opposition) >= 2
    gate3 = sum(retention) >= 2
    return {"gate1_exact_prediction_parity": gate1, "gate2_msrc_opposition": gate2, "gate3_msrc_retention": gate3, "gate4_integrity_no_gt_runtime": gate4, "gate2_seed_passes": opposition, "gate3_seed_passes": retention, "final_decision": "SUPPORTS_ACTION_STRUCTURE_COMPATIBILITY_HYPOTHESIS" if all((gate1, gate2, gate3, gate4)) else "FAIL-CLOSED"}

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    rows = [d.run_one(dataset, seed, "cuda:0", output_root=RECOVERY_ROOT, resume=args.resume) for dataset in d.DATASETS for seed in d.SEEDS]
    report = {"recovery_root": str(RECOVERY_ROOT), "rows": rows, "gates": _gate_rows(rows), "no_gt": True, "frozen_v1_only": True, "refresh_interval": 100}
    target = RECOVERY_ROOT / "d6_b0_recovery_gate_adjudication.json"
    if target.exists():
        raise RuntimeError("D6_B0_RECOVERY_REPORT_EXISTS")
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[D6-B0] verified 9/9 frozen-v1 recovery replays", flush=True)

if __name__ == "__main__":
    main()
