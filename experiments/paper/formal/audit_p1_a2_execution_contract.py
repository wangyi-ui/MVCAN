"""Emit static P1-A2 capability preflight; it never trains or creates run paths."""

import json
from pathlib import Path

from . import p1_a0_formal_protocol as protocol
from . import p1_a2_execution_contract as execution

OUTPUT = Path("outputs/paper/protocol/p1_a2_formal_execution_preflight.json")


def build_report():
    execution.validate_execution_contract()
    rows = []
    for dataset in (item.name for item in protocol.FORMAL_DATASETS):
        for seed in protocol.FORMAL_TRAINING_SEEDS:
            for arm in ("BASE", "OURS_TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U"):
                capability, reason = execution.arm_capability(dataset, arm)
                rows.append({"dataset": dataset, "training_seed": seed, "arm": arm,
                             "capability": capability, "reason": reason})
    main = [row for row in rows if row["arm"] in ("BASE", "OURS_TRUE_U")]
    main_count = sum(row["capability"] == "AUTHORIZED" for row in main)
    return {"schema": "paper-p1-a2-formal-execution-preflight-v1", "cells": rows,
            "cell_count": len(rows), "main_authorized_count": main_count,
            "main_decision": "FORMAL_MAIN_RUNNER_READY" if main_count == 18 else "FORMAL_MAIN_RUNNER_NOT_READY",
            "ablation_decision": "FORMAL_ABLATION_PARTIALLY_BLOCKED",
            "generator_contract": execution.NATIVE_DATALOADER_GENERATOR_CONTRACT,
            "action_carrier_contract": execution.ACTION_CARRIER_CONTRACT,
            "true_uniform_contract": execution.TRUE_UNIFORM_CONTRACT,
            "source_provenance": execution.SOURCE_PROVENANCE,
            "real_formal_runs": "NONE"}


def write_report(path=OUTPUT):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(build_report(), stream, sort_keys=True, indent=2)
        stream.write("\n")
    return target


if __name__ == "__main__":
    write_report()
