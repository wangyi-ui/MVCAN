"""Emit the P1-A0 protocol recovery report without running an experiment."""

import json
from pathlib import Path

from . import p1_a0_formal_protocol as protocol


OUTPUT = Path("outputs/paper/protocol/p1_a0_formal_protocol_recovery.json")


def build_report():
    resolved = protocol.validate_formal_protocol()
    return {
        "schema": "paper-p1-a0-formal-protocol-recovery-v1",
        "decision": "FORMAL_PROTOCOL_RESOLVED" if resolved else "FORMAL_PROTOCOL_UNRESOLVED",
        "datasets": [item.__dict__ for item in protocol.FORMAL_DATASETS],
        "dataset_matrix": [dict(item.__dict__, status="RESOLVED") for item in protocol.FORMAL_DATASETS],
        "seed_matrix": {"training_seeds": protocol.FORMAL_TRAINING_SEEDS, "weak_quality_seed": protocol.WEAK_QUALITY_PROTOCOL["realization_seed"], "label_seed": protocol.SPARSE_LABEL_PROTOCOL["label_seed"], "native_config_seeds": {item.name: item.native_config_seed for item in protocol.FORMAL_DATASETS}},
        "training_seeds": protocol.FORMAL_TRAINING_SEEDS,
        "weak_quality": protocol.WEAK_QUALITY_PROTOCOL,
        "sparse_labels": protocol.SPARSE_LABEL_PROTOCOL,
        "native_preparation": protocol.NATIVE_PREPARATION_PROTOCOL,
        "action_generation": protocol.ACTION_GENERATION_PROTOCOL,
        "alternating": protocol.ALTERNATING_PROTOCOL.__dict__,
        "refresh_schedule": {"interval": protocol.ALTERNATING_PROTOCOL.refresh_interval, "zero_based_epoch_0_refresh": True, "subsequent_epochs": [], "final_prediction_refresh_always_once": True},
        "final_prediction": {"source": protocol.ALTERNATING_PROTOCOL.final_prediction_source, "q_argmax_allowed": False, "sample_id_alignment": protocol.EVALUATION_PROTOCOL["sample_id_alignment"]},
        "evaluation": protocol.EVALUATION_PROTOCOL,
        "arms": protocol.ARM_PROTOCOL,
        "forbidden_science": protocol.FORBIDDEN_SCIENCE,
        "source_provenance": protocol.SOURCE_PROVENANCE,
        "resolved_fields": protocol.FORMAL_FIELD_STATUS,
        "unresolved_fields": list(protocol.UNRESOLVED_FIELDS),
        "formal_training_executed": False,
    }


def write_report(path=OUTPUT):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(build_report(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return target


if __name__ == "__main__":
    write_report()
