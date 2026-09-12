"""Summarize the preregistered C3-B0 three-seed/four-arm pilot."""

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    c3_b0_relation_action_protocol as c3b0,
)
from experiments.cyclic_utility import (
    train_c3_b0_relation_action_pilot as train,
)


DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs/cyclic_utility/c3_b0_relation_action_pilot_multiseed"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_formal_records(input_root=None):
    """Read only the twelve C3-B0 outputs; never load labels or predictions."""
    root = REPOSITORY_ROOT if input_root is None else Path(input_root)
    records = {}
    provenance = {}
    for seed in c3b0.SEEDS:
        records[seed] = {}
        provenance[str(seed)] = {}
        for arm in c3b0.ARMS:
            arm_dir = (
                train.default_output_dir(seed, arm)
                if input_root is None
                else root / (
                    "c3_b0_relation_action_pilot_seed" + str(seed) + "_"
                    + c3b0.ARM_DIRECTORY_NAMES[arm]
                )
            )
            paths = {
                name: arm_dir / name
                for name in (
                    "config.json", "audit.json", "metrics.json",
                    "training_summary.json",
                )
            }
            _require(all(path.is_file() for path in paths.values()),
                     "missing formal C3-B0 arm output: " + str(arm_dir))
            config = _read_json(paths["config.json"])
            audit = _read_json(paths["audit.json"])
            metric_record = _read_json(paths["metrics.json"])
            training = _read_json(paths["training_summary.json"])
            _require(
                config.get("stage") == audit.get("stage")
                == metric_record.get("stage") == training.get("stage")
                == c3b0.STAGE
                and config.get("seed") == audit.get("seed")
                == metric_record.get("seed") == training.get("seed") == seed
                and config.get("arm") == audit.get("arm")
                == metric_record.get("arm") == training.get("arm") == arm
                and audit.get("GT_loaded_during_training") is False
                and training.get("test_GT_used_for_epoch_selection") is False
                and audit.get("additive_combined_loss_used") is False,
                "C3-B0 formal record boundary mismatch",
            )
            metrics = metric_record.get("metrics", {})
            _require(set(metrics) == set(c3b0.METRICS),
                     "C3-B0 metric schema mismatch")
            records[seed][arm] = metrics
            provenance[str(seed)][arm] = {
                name: {
                    "path": str(path),
                    "file_sha256": train.file_sha256(path),
                }
                for name, path in paths.items()
            }
    return records, provenance


def summarize(input_root=None, output_dir=DEFAULT_OUTPUT_DIR):
    target = Path(output_dir)
    _require(not target.exists(), "refusing to overwrite C3-B0 summary")
    records, provenance = load_formal_records(input_root)
    decision = c3b0.summarize_multiseed_metrics(records)
    target.mkdir(parents=True, exist_ok=False)
    summary = {
        "stage": c3b0.STAGE,
        "seeds": list(c3b0.SEEDS),
        "arms": list(c3b0.ARMS),
        "metrics_by_seed": {
            str(seed): records[seed] for seed in c3b0.SEEDS
        },
        "comparisons": decision["comparisons"],
    }
    audit = {
        "stage": c3b0.STAGE,
        "input_provenance": provenance,
        "formal_seed_set_exact": True,
        "formal_arm_set_exact": True,
        "gate_preregistered_before_formal_execution": True,
        "GT_loaded_by_summarizer": False,
        "memory_automatically_enabled": False,
    }
    c1.write_json(target / "c3_b0_multiseed_summary.json", summary)
    c1.write_json(target / "c3_b0_multiseed_decision.json", decision)
    c1.write_json(target / "c3_b0_multiseed_audit.json", audit)
    return {"summary": summary, "decision": decision, "audit": audit}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = summarize(args.input_root, args.output_dir)
    print("DECISION=" + result["decision"]["decision"])
    return 0 if result["decision"]["C3_B0_RELATION_ACTION_PILOT_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
