"""E2 controlled continuation with frozen B7 sparse semantic anchors.

No full ground-truth array is loaded before final predictions are saved and
hashed.  The only training targets are the 14 class memberships already
encoded in the frozen B7 split provenance.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.b7_sparse_supervision.b7_sparse_anchor_protocol import (
    load_fixed_label_split,
)
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    robust_inter_affinity,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    whole_row_reliability_shuffle,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    E2_ARMS,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    build_sparse_class_prototypes,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    semantic_pairwise_cooperation_loss,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    sparse_semantic_probabilities,
)
from irv.b3_audit import hash_backbone
from weak_quality import ndarray_sha256


STAGE = "E2"
DATASET = e1_train.DATASET
SEED = e1_train.SEED
SAMPLE_NUM = e1_train.SAMPLE_NUM
VIEW_NUM = e1_train.VIEW_NUM
CLUSTER_NUM = e1_train.CLUSTER_NUM
LAMBDA1 = e1_train.LAMBDA1
TARGET_REFRESH_INTERVAL = e1_train.TARGET_REFRESH_INTERVAL
TARGET_WEIGHT_UPDATES = e1_train.TARGET_WEIGHT_UPDATES
KMEANS_N_INIT = e1_train.KMEANS_N_INIT

LABELED_NUM = 14
UNLABELED_NUM = 1386
EXPECTED_LABELED_IDS = np.asarray(
    [67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385],
    dtype=np.int64,
)
EXPECTED_LABELED_IDS_SHA256 = (
    "8bb6d96a7593bb7916915a66fd25a5671df01d923c811ebdaaa5cc5be1f381b3"
)
EXPECTED_LABEL_SPLIT_SHA256 = (
    "0463cf7155bc2b90a6133f0a79fd30d8c5a9e78079c6692dd1fa4106cb174487"
)

DEFAULT_SPLIT_DIR = (
    REPOSITORY_ROOT / "outputs/b7_sparse_supervision/b7a0_seed20"
)
DEFAULT_E1_LWC_SMOKE_DIR = (
    REPOSITORY_ROOT / "outputs/e1_pairwise_utility/lwc_2ep_seed20"
)
DEFAULT_E1_LWC_FORMAL_DIR = (
    REPOSITORY_ROOT / "outputs/e1_pairwise_utility/lwc_100ep_seed20"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def load_sparse_training_labels(split_dir=DEFAULT_SPLIT_DIR):
    """Load only the frozen B7 IDs and their 14 class memberships.

    ``label_split.json`` already commits each labeled ID to a class through
    ``per_class_labeled_ids``.  Consequently no Caltech full-label file is
    opened at this training boundary.
    """
    split = load_fixed_label_split(_resolve(split_dir))
    labeled_ids = np.asarray(split["labeled_sample_ids"], dtype=np.int64)
    unlabeled_ids = np.asarray(split["unlabeled_sample_ids"], dtype=np.int64)
    record = split["record"]
    _require(
        split["label_split_sha256"] == EXPECTED_LABEL_SPLIT_SHA256,
        "B7 frozen label split logical SHA mismatch",
    )
    _require(
        split["labeled_ids_sha256"] == EXPECTED_LABELED_IDS_SHA256
        and ndarray_sha256(labeled_ids) == EXPECTED_LABELED_IDS_SHA256,
        "B7 labeled ID logical SHA mismatch",
    )
    _require(
        labeled_ids.shape == (LABELED_NUM,)
        and np.array_equal(labeled_ids, EXPECTED_LABELED_IDS),
        "B7 labeled IDs differ from the frozen 14-ID list",
    )
    _require(
        unlabeled_ids.shape == (UNLABELED_NUM,)
        and np.intersect1d(labeled_ids, unlabeled_ids).size == 0
        and np.array_equal(
            np.sort(np.concatenate((labeled_ids, unlabeled_ids))),
            np.arange(SAMPLE_NUM, dtype=np.int64),
        ),
        "B7 labeled/unlabeled partition mismatch",
    )

    target_by_id = {}
    per_class_ids = record.get("per_class_labeled_ids", {})
    _require(set(per_class_ids) == set(map(str, range(CLUSTER_NUM))),
             "B7 split is missing one or more classes")
    for class_id in range(CLUSTER_NUM):
        class_ids = np.asarray(per_class_ids[str(class_id)], dtype=np.int64)
        _require(class_ids.shape == (2,), "B7 class does not have two anchors")
        for sample_id in class_ids:
            _require(int(sample_id) not in target_by_id,
                     "duplicate ID in B7 per-class mapping")
            target_by_id[int(sample_id)] = class_id
    _require(
        set(target_by_id) == set(map(int, labeled_ids)),
        "B7 per-class mapping does not exactly cover labeled IDs",
    )
    labeled_targets = np.ascontiguousarray(
        [target_by_id[int(sample_id)] for sample_id in labeled_ids],
        dtype=np.int64,
    )
    class_counts = np.bincount(labeled_targets, minlength=CLUSTER_NUM)
    _require(
        np.array_equal(class_counts, np.full(CLUSTER_NUM, 2, dtype=np.int64)),
        "sparse training target counts must be [2,2,2,2,2,2,2]",
    )
    labeled_ids.setflags(write=False)
    unlabeled_ids.setflags(write=False)
    labeled_targets.setflags(write=False)
    audit = {
        "training_label_source": (
            "frozen B7 label_split.json per_class_labeled_ids only"
        ),
        "full_label_file_opened_before_training": False,
        "split_regenerated": False,
        "label_split_path": _display(split["label_split_path"]),
        "labeled_ids_path": _display(split["labeled_ids_path"]),
        "unlabeled_ids_path": _display(split["unlabeled_ids_path"]),
        "label_split_sha256": split["label_split_sha256"],
        "labeled_ids_sha256": split["labeled_ids_sha256"],
        "labeled_ids": labeled_ids.tolist(),
        "labeled_targets": labeled_targets.tolist(),
        "class_counts": class_counts.tolist(),
        "labeled_count": LABELED_NUM,
        "unlabeled_count": UNLABELED_NUM,
        "two_labels_per_class_pass": True,
        "exact_partition_pass": True,
    }
    return labeled_ids, labeled_targets, unlabeled_ids, audit


@torch.no_grad()
def refresh_native_target_and_semantics(model, full_views, view_weights, device):
    """E1-native refresh plus its already-computed aligned-q snapshot.

    The target, view-weight, KMeans, and Hungarian operations are copied
    verbatim from the frozen E1 refresh.  Capturing q_local_full avoids an
    extra forward and therefore avoids extra BatchNorm state mutation.
    """
    kmeans = KMeans(
        n_clusters=CLUSTER_NUM,
        n_init=KMEANS_N_INIT,
        random_state=SEED,
    )
    y_views = None
    q_views_full = None
    latent_fusion = None
    for _ in range(TARGET_WEIGHT_UPDATES):
        latent_views = []
        y_views = []
        q_views_full = []
        for view_id in range(VIEW_NUM):
            x_view = full_views[view_id].to(device)
            latent = model.autoencoders[view_id].encoder(x_view)
            q_local = model.autoencoders[view_id].clustering(latent)
            scaled = MinMaxScaler().fit_transform(
                latent.detach().cpu().numpy()
            )
            latent_views.append(scaled * view_weights[view_id])
            y_views.append(q_local.detach().cpu().numpy().argmax(1))
            q_views_full.append(q_local.detach())
        latent_fusion = np.hstack(latent_views)
        y_pred = kmeans.fit_predict(latent_fusion)
        for view_id in range(VIEW_NUM):
            nmi = round(
                normalized_mutual_info_score(y_pred, y_views[view_id]), 5
            )
            view_weights[view_id] = float(np.exp(nmi))

    match_arrays = []
    for view_id in range(VIEW_NUM):
        _, _, _, match = model.Match(y_views[view_id], y_pred)
        match_arrays.append(match)
    # M: [6,7,7], constant/no grad.
    matches = torch.from_numpy(np.stack(match_arrays)).float().to(device)
    matches.requires_grad_(False)
    # P_all retains the exact frozen MVCAN definition and is never rewritten.
    p_all_numpy = model.target_distribution(
        model.new_P(latent_fusion, kmeans.cluster_centers_)
    )
    p_all = torch.from_numpy(p_all_numpy).float()
    # q_local_full/q_aligned_full/h_sem_full: [N,6,7], all detached.
    q_local_full = torch.stack(q_views_full, dim=1).detach()
    q_aligned_full, h_sem_full = align_semantic_probabilities(
        q_local_full, matches
    )
    _require(
        tuple(q_local_full.shape) == (SAMPLE_NUM, VIEW_NUM, CLUSTER_NUM)
        and tuple(q_aligned_full.shape) == (SAMPLE_NUM, VIEW_NUM, CLUSTER_NUM)
        and tuple(h_sem_full.shape) == (SAMPLE_NUM, VIEW_NUM, CLUSTER_NUM),
        "full aligned-q semantic shape audit failed",
    )
    _require(
        not q_local_full.requires_grad
        and not q_aligned_full.requires_grad
        and not h_sem_full.requires_grad,
        "prototype semantic representation must be stop-gradient",
    )
    return (
        p_all,
        matches,
        np.asarray(y_pred, dtype=np.int64),
        view_weights,
        h_sem_full.detach(),
    )


def _merge_probability_audit(aggregate, current):
    if aggregate is None:
        return {
            "batch_count": 1,
            "shape_pass": current["shape_pass"],
            "finite_pass": current["finite_pass"],
            "nonnegative_pass": current["nonnegative_pass"],
            "row_sum_one_pass": current["row_sum_one_pass"],
            "stop_gradient_pass": current["stop_gradient_pass"],
            "minimum": current["minimum"],
            "maximum": current["maximum"],
            "max_row_sum_error": current["max_row_sum_error"],
            "temperature_used": False,
        }
    aggregate["batch_count"] += 1
    for key in (
        "shape_pass",
        "finite_pass",
        "nonnegative_pass",
        "row_sum_one_pass",
        "stop_gradient_pass",
    ):
        aggregate[key] = bool(aggregate[key] and current[key])
    aggregate["minimum"] = min(aggregate["minimum"], current["minimum"])
    aggregate["maximum"] = max(aggregate["maximum"], current["maximum"])
    aggregate["max_row_sum_error"] = max(
        aggregate["max_row_sum_error"], current["max_row_sum_error"]
    )
    return aggregate


def train_semantic_continuation(
    arm,
    epochs,
    batch_size,
    model,
    optimizers,
    views,
    sample_ids,
    reliability,
    labeled_ids,
    labeled_targets,
    device,
):
    """Train S/RS/shuffled-RS; LWC replay is intentionally handled elsewhere."""
    _require(
        arm in ("S_LWC", "RS_LWC", "SHUFFLED_RS_LWC"),
        "semantic continuation received a non-semantic arm",
    )
    _require(int(epochs) in (2, 100), "E2 permits only 2 or 100 epochs")
    _require(int(batch_size) == 256, "E2 batch size must remain 256")
    generator = torch.Generator()
    generator.manual_seed(SEED)
    view_weights = [1.0] * VIEW_NUM
    full_views = [torch.from_numpy(view) for view in views]
    id_tensor = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64))

    prototype_reliability = reliability
    shuffle_audit = None
    if arm == "SHUFFLED_RS_LWC":
        prototype_reliability, _, shuffle_audit = whole_row_reliability_shuffle(
            reliability, seed=SEED
        )
    # Writable independent storage, frozen/no-grad. R is used only at refresh.
    reliability_tensor = e1_train.frozen_reliability_tensor(
        prototype_reliability
    ).to(device=device)
    reliability_tensor.requires_grad_(False)
    labeled_id_tensor = torch.from_numpy(
        np.array(labeled_ids, copy=True, dtype=np.int64)
    ).to(device=device)
    labeled_target_tensor = torch.from_numpy(
        np.array(labeled_targets, copy=True, dtype=np.int64)
    ).to(device=device)

    p_all = None
    matches = None
    prototypes = None
    prototype_audits = []
    probability_audit = None
    pair_gradient_audit = None
    graph_stopgrad_pass = True
    semantic_compatibility_range_pass = True
    semantic_compatibility_min = 1.0
    semantic_compatibility_max = 0.0
    interaction_utility_stopgrad_pass = True
    loss_finite_pass = True
    epoch_records = []
    native_target_refresh_count = 0

    for epoch in range(int(epochs)):
        if epoch % TARGET_REFRESH_INTERVAL == 0:
            (
                p_all,
                matches,
                _,
                view_weights,
                h_sem_full,
            ) = refresh_native_target_and_semantics(
                model, full_views, view_weights, device
            )
            prototype_r = (
                None if arm == "S_LWC" else reliability_tensor
            )
            prototypes, prototype_audit = build_sparse_class_prototypes(
                h_sem_full=h_sem_full.detach(),
                labeled_sample_ids=labeled_id_tensor,
                labeled_targets=labeled_target_tensor,
                class_num=CLUSTER_NUM,
                reliability=prototype_r,
            )
            prototype_audit["epoch"] = epoch + 1
            prototype_audits.append(prototype_audit)
            native_target_refresh_count += 1
        _require(
            p_all is not None and matches is not None and prototypes is not None,
            "native target or sparse prototype is missing",
        )
        dataset = torch.utils.data.TensorDataset(*full_views, p_all, id_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
            generator=generator,
            num_workers=0,
        )
        visited = []
        native_sum = 0.0
        pair_sum = 0.0
        total_sum = 0.0
        batch_count = 0
        for packed in loader:
            x_views = [packed[view_id].to(device) for view_id in range(VIEW_NUM)]
            p_batch = packed[VIEW_NUM].to(device)
            batch_ids = packed[VIEW_NUM + 1]
            visited.append(batch_ids.numpy().astype(np.int64, copy=False))
            for optimizer in optimizers:
                optimizer.zero_grad()
            reconstructions = []
            q_views = []
            for view_id in range(VIEW_NUM):
                x_hat, z, q = model.autoencoders[view_id](x_views[view_id])
                # z: [B,10], per-view q: [B,7].
                reconstructions.append(x_hat)
                q_views.append(q)
            # q_local/q_aligned/h_sem: [B,6,7], differentiable.
            q_local = torch.stack(q_views, dim=1)
            native_losses, _ = e1_train.native_mvcan_losses(
                x_views, reconstructions, q_views, p_batch, matches
            )
            native_total = torch.stack(native_losses).sum()
            q_aligned, h_sem = align_semantic_probabilities(q_local, matches)
            _require(q_aligned.requires_grad, "minibatch aligned q lost gradient")
            # G_inter: [6,6,B,B], frozen from h_sem.detach().
            graph_inter = robust_inter_affinity(h_sem.detach())
            graph_stopgrad_pass = bool(
                graph_stopgrad_pass
                and not graph_inter.requires_grad
                and graph_inter.grad_fn is None
            )
            # semantic_prob: [B,6,7], frozen prototype evidence only.
            semantic_prob, current_probability_audit = (
                sparse_semantic_probabilities(h_sem.detach(), prototypes)
            )
            probability_audit = _merge_probability_audit(
                probability_audit, current_probability_audit
            )
            pair_loss, pair_diagnostics = semantic_pairwise_cooperation_loss(
                h_sem, graph_inter, semantic_prob
            )
            if pair_gradient_audit is None:
                pair_gradient_audit = e1_train._pair_gradient_audit(
                    pair_loss, q_local, model
                )
            for pair_record in pair_diagnostics:
                semantic_compatibility_range_pass = bool(
                    semantic_compatibility_range_pass
                    and pair_record["S_range_pass"]
                )
                semantic_compatibility_min = min(
                    semantic_compatibility_min, pair_record["S_min"]
                )
                semantic_compatibility_max = max(
                    semantic_compatibility_max, pair_record["S_max"]
                )
                interaction_utility_stopgrad_pass = bool(
                    interaction_utility_stopgrad_pass
                    and pair_record["U_sem_stop_gradient_pass"]
                )
            # No new coefficient: reuse MVCAN lambda1=0.01 exactly.
            total_loss = native_total + LAMBDA1 * pair_loss
            finite = bool(
                torch.isfinite(native_total).item()
                and torch.isfinite(pair_loss).item()
                and torch.isfinite(total_loss).item()
            )
            loss_finite_pass = bool(loss_finite_pass and finite)
            _require(finite, "non-finite E2 loss")
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            native_sum += float(native_total.detach().item())
            pair_sum += float(pair_loss.detach().item())
            total_sum += float(total_loss.detach().item())
            batch_count += 1

        coverage = e1_train._coverage_audit(visited)
        _require(
            coverage["all_1400_sample_ids_exactly_once_pass"],
            "E2 epoch did not cover all sample IDs exactly once",
        )
        epoch_records.append({
            "epoch": epoch + 1,
            "batch_count": batch_count,
            "native_loss_mean": native_sum / batch_count,
            "pair_loss_mean": pair_sum / batch_count,
            "total_loss_mean": total_sum / batch_count,
            **coverage,
        })

    # Fixed final epoch only; no labels or model-selection criterion enter.
    _, final_matches, final_predictions, _ = e1_train.refresh_native_target(
        model, full_views, view_weights, device
    )
    return final_predictions, {
        "epoch_records": epoch_records,
        "native_target_refresh_count": native_target_refresh_count,
        "P_all_definition": "MVCAN target_distribution(new_P(latent_fusion, centers))",
        "P_all_rewritten": False,
        "prototype_audits": prototype_audits,
        "prototype_validity_pass": bool(
            all(
                all(
                    audit[key]
                    for key in (
                        "shape_pass",
                        "anchor_count_two_every_view_class_pass",
                        "denominator_positive_every_view_class_pass",
                        "prototype_finite_pass",
                        "prototype_unit_norm_pass",
                        "no_missing_class_pass",
                        "prototype_stop_gradient_pass",
                    )
                )
                for audit in prototype_audits
            )
        ),
        "semantic_probability_audit": probability_audit,
        "semantic_probability_validity_pass": bool(
            probability_audit is not None
            and all(
                probability_audit[key]
                for key in (
                    "shape_pass",
                    "finite_pass",
                    "nonnegative_pass",
                    "row_sum_one_pass",
                    "stop_gradient_pass",
                )
            )
        ),
        "S_range_pass": bool(semantic_compatibility_range_pass),
        "S_min": semantic_compatibility_min,
        "S_max": semantic_compatibility_max,
        "U_sem_stop_gradient_pass": bool(interaction_utility_stopgrad_pass),
        "graph_stopgrad_pass": bool(graph_stopgrad_pass),
        "reliability_frozen_no_grad_pass": bool(
            not reliability_tensor.requires_grad
            and reliability_tensor.grad_fn is None
        ),
        "pair_gradient_audit": pair_gradient_audit,
        "loss_finite_pass": bool(loss_finite_pass),
        "all_epochs_exact_sample_coverage_pass": bool(
            all(
                record["all_1400_sample_ids_exactly_once_pass"]
                for record in epoch_records
            )
        ),
        "sample_order_sha256_per_epoch": [
            record["sample_order_sha256"] for record in epoch_records
        ],
        "final_M_shape": list(final_matches.shape),
        "M_no_grad_pass": bool(
            not final_matches.requires_grad and final_matches.grad_fn is None
        ),
        "shuffle_audit": shuffle_audit,
        "R_role": "prototype construction only",
        "R_directly_multiplied_into_pair_weight": False,
        "epoch_selection_used": False,
    }


def audit_lwc_replay_identity(
    initial_model_hash,
    final_model_hash,
    prediction_audit,
    all_metrics,
    reference_dir=DEFAULT_E1_LWC_SMOKE_DIR,
):
    reference_dir = _resolve(reference_dir)
    reference_path = reference_dir / "e1_audit.json"
    _require(reference_path.is_file(), "frozen E1 LWC smoke audit is missing")
    reference = _read_json(reference_path)
    checks = {
        "reference_E1_engineering_pass": reference.get("E1_ENGINEERING_PASS") is True,
        "initial_model_hash_exact_match": initial_model_hash
        == reference.get("initial_model_hash"),
        "final_model_aggregate_hash_exact_match": final_model_hash.get("aggregate")
        == reference.get("final_model_hash", {}).get("aggregate"),
        "final_prediction_logical_hash_exact_match": prediction_audit.get(
            "logical_sha256"
        )
        == reference.get("prediction_audit", {}).get("logical_sha256"),
        "ACC_exact_match": all_metrics.get("ACC")
        == reference.get("metrics", {}).get("ACC"),
        "NMI_exact_match": all_metrics.get("NMI")
        == reference.get("metrics", {}).get("NMI"),
        "ARI_exact_match": all_metrics.get("ARI")
        == reference.get("metrics", {}).get("ARI"),
    }
    return {
        "reference_dir": _display(reference_dir),
        "reference_audit": _display(reference_path),
        "checks": checks,
        "LWC_REPLAY_EXACT_MATCH_PASS": bool(all(checks.values())),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="E2 sparse semantic interaction utility; seed20 only"
    )
    parser.add_argument("--arm", required=True, choices=E2_ARMS)
    parser.add_argument("--epochs", required=True, type=int, choices=(2, 100))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=str(e1_train.DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--reliability-path", default=str(e1_train.DEFAULT_RELIABILITY_PATH))
    parser.add_argument("--d2-audit-path", default=str(e1_train.DEFAULT_D2_AUDIT_PATH))
    parser.add_argument("--feature-path", default=str(e1_train.DEFAULT_FEATURE_PATH))
    parser.add_argument(
        "--feature-audit-path", default=str(e1_train.DEFAULT_FEATURE_AUDIT_PATH)
    )
    parser.add_argument("--glgc-repository", default=str(e1_train.DEFAULT_GLGC_REPOSITORY))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--label-path", default=str(e1_train.DEFAULT_LABEL_PATH))
    parser.add_argument(
        "--e1-lwc-smoke-reference-dir",
        default=str(DEFAULT_E1_LWC_SMOKE_DIR),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(
        not (args.arm == "LWC_REPLAY" and args.epochs != 2),
        "LWC_REPLAY is permitted only for the 2-epoch engineering smoke",
    )
    output_dir = (
        REPOSITORY_ROOT
        / "outputs/e2_sparse_semantic_utility"
        / (args.arm.lower() + "_" + str(args.epochs) + "ep_seed20")
        if args.output_dir is None
        else _resolve(args.output_dir)
    )
    _require(not output_dir.exists(), "refusing to overwrite E2 output")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA device unavailable")
        torch.cuda.set_device(device)

    e1_train.set_global_seed(SEED)
    glgc_provenance = e1_train.verify_glgc_released_logic(
        args.glgc_repository
    )
    views, sample_ids, feature_provenance = (
        e1_train.load_frozen_feature_artifact(
            args.feature_path, args.feature_audit_path
        )
    )
    reliability, reliability_provenance = e1_train.load_frozen_reliability(
        sample_ids,
        args.reliability_path,
        args.d2_audit_path,
        feature_provenance,
    )
    # This boundary reads only the frozen 14-ID class mapping, never full Y.
    labeled_ids, labeled_targets, unlabeled_ids, label_split_audit = (
        load_sparse_training_labels(args.split_dir)
    )
    model, config, checkpoint_provenance = e1_train.build_model_from_frozen_d1(
        args.checkpoint_dir, device
    )
    d2_audit = _read_json(_resolve(args.d2_audit_path))
    _require(
        checkpoint_provenance["initial_backbone_hash"]
        == d2_audit["source_provenance"]["backbone_hash_before"],
        "initial model differs from frozen D1 provenance",
    )
    optimizers = e1_train.build_fresh_optimizers(model)
    optimizer_audit = {
        "policy": "fresh independent Adam per MVCAN view from model-only D1 checkpoint",
        "optimizer_count": len(optimizers),
        "initial_state_empty_pass": bool(
            all(len(optimizer.state) == 0 for optimizer in optimizers)
        ),
        "learning_rate": e1_train.LEARNING_RATE,
        "same_policy_all_arms": True,
    }

    output_dir.mkdir(parents=True)
    if args.arm == "LWC_REPLAY":
        # Direct call into frozen E1 is the identity mechanism; E2 math is never
        # entered on this arm.
        predictions, runtime_audit = e1_train.train_continuation(
            arm="LWC",
            epochs=args.epochs,
            batch_size=int(config["training"]["batch_size"]),
            model=model,
            optimizers=optimizers,
            views=views,
            sample_ids=sample_ids,
            reliability=reliability,
            device=device,
        )
        runtime_audit.update({
            "E2_semantic_branch_entered": False,
            "replay_source": (
                "E1 train_continuation with exact arm='LWC'"
            ),
            "epoch_selection_used": False,
        })
    else:
        predictions, runtime_audit = train_semantic_continuation(
            arm=args.arm,
            epochs=args.epochs,
            batch_size=int(config["training"]["batch_size"]),
            model=model,
            optimizers=optimizers,
            views=views,
            sample_ids=sample_ids,
            reliability=reliability,
            labeled_ids=labeled_ids,
            labeled_targets=labeled_targets,
            device=device,
        )
        runtime_audit["E2_semantic_branch_entered"] = True

    # Prediction persistence/hash is the hard boundary before full labels.
    prediction_path, prediction_audit = e1_train.save_and_hash_predictions(
        predictions, output_dir
    )
    # The final checkpoint is also fixed before any unlabeled GT is opened.
    final_model_hash = hash_backbone(model.autoencoders)
    model_outputs = e1_train.save_final_models(model, output_dir)
    full_labels = e1_train.load_labels_after_predictions(
        args.label_path, prediction_path
    )
    _require(
        np.array_equal(full_labels[labeled_ids], labeled_targets),
        "post-training sparse/full label consistency check failed",
    )
    primary_unlabeled_metrics = e1_train.evaluate_predictions(
        full_labels[unlabeled_ids], predictions[unlabeled_ids]
    )
    secondary_all_metrics = e1_train.evaluate_predictions(
        full_labels, predictions
    )
    replay_audit = None
    if args.arm == "LWC_REPLAY":
        replay_audit = audit_lwc_replay_identity(
            checkpoint_provenance["initial_backbone_hash"],
            final_model_hash,
            prediction_audit,
            secondary_all_metrics,
            args.e1_lwc_smoke_reference_dir,
        )

    semantic_arm = args.arm != "LWC_REPLAY"
    engineering_checks = {
        "initial_model_matches_frozen_d1_pass": True,
        "optimizer_initial_state_policy_pass": optimizer_audit[
            "initial_state_empty_pass"
        ],
        "all_1400_sample_ids_covered_each_epoch_pass": runtime_audit[
            "all_epochs_exact_sample_coverage_pass"
        ],
        "loss_finite_pass": runtime_audit["loss_finite_pass"],
        "fixed_B7_split_pass": label_split_audit["label_split_sha256"]
        == EXPECTED_LABEL_SPLIT_SHA256,
        "exact_14_sparse_labels_pass": label_split_audit[
            "two_labels_per_class_pass"
        ],
        "unlabeled_1386_GT_training_isolation_pass": bool(
            not label_split_audit["full_label_file_opened_before_training"]
        ),
        "full_labels_after_prediction_hash_pass": prediction_audit[
            "saved_and_reloaded_before_labels_pass"
        ],
        "no_corruption_mask_loaded_pass": bool(
            not feature_provenance["corruption_mask_present"]
            and not reliability_provenance["corruption_mask_loaded"]
        ),
        "no_target_rewriting_pass": not runtime_audit["P_all_rewritten"],
        "M_no_grad_pass": runtime_audit["M_no_grad_pass"],
        "graph_branch_no_grad_pass": runtime_audit["graph_stopgrad_pass"],
        "pair_branch_gradient_exists_pass": runtime_audit[
            "pair_gradient_audit"
        ]["pair_branch_gradient_exists_pass"],
        "prototype_validity_pass": bool(
            not semantic_arm or runtime_audit["prototype_validity_pass"]
        ),
        "semantic_probability_validity_pass": bool(
            not semantic_arm
            or runtime_audit["semantic_probability_validity_pass"]
        ),
        "S_range_pass": bool(
            not semantic_arm or runtime_audit["S_range_pass"]
        ),
        "U_sem_stop_gradient_pass": bool(
            not semantic_arm or runtime_audit["U_sem_stop_gradient_pass"]
        ),
        "R_frozen_no_grad_pass": runtime_audit[
            "reliability_frozen_no_grad_pass"
        ],
        "LWC_replay_exact_match_pass": bool(
            semantic_arm or replay_audit["LWC_REPLAY_EXACT_MATCH_PASS"]
        ),
        "fixed_last_epoch_no_model_selection_pass": bool(
            not runtime_audit["epoch_selection_used"]
        ),
    }
    engineering_pass = bool(all(engineering_checks.values()))
    result = {
        "stage": STAGE,
        "arm": args.arm,
        "epochs": int(args.epochs),
        "seed": SEED,
        "dataset": DATASET,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLUSTER_NUM,
        "lambda1_native_and_pair": LAMBDA1,
        "pair_loss_coefficient": LAMBDA1,
        "initial_model_hash": checkpoint_provenance["initial_backbone_hash"],
        "final_model_hash": final_model_hash,
        "checkpoint_provenance": checkpoint_provenance,
        "optimizer_audit": optimizer_audit,
        "feature_provenance": feature_provenance,
        "D2_R_provenance": reliability_provenance,
        "B7_label_split_provenance": label_split_audit,
        "GLGC_released_logic_provenance": glgc_provenance,
        "runtime": runtime_audit,
        "prediction_audit": prediction_audit,
        "final_model_outputs": model_outputs,
        "label_protocol": {
            "training_label_count": LABELED_NUM,
            "training_labels_source": "frozen B7 split provenance",
            "unlabeled_training_GT_count": 0,
            "full_labels_loaded_only_after_final_predictions_saved_and_hashed": True,
            "full_labels_used_only_for_final_metrics": True,
            "post_training_sparse_label_consistency_pass": True,
        },
        "primary_unlabeled_1386_metrics": primary_unlabeled_metrics,
        "secondary_all_1400_metrics": secondary_all_metrics,
        "LWC_replay_audit": replay_audit,
        "frozen_formal_LWC_reference": _display(DEFAULT_E1_LWC_FORMAL_DIR),
        "engineering_checks": engineering_checks,
        "E2_ENGINEERING_PASS": engineering_pass,
    }
    _write_json(output_dir / "e2_audit.json", result)
    print("E2_ENGINEERING_PASS=" + str(engineering_pass))
    print("ARM=" + args.arm)
    print("PRIMARY_UNLABELED_ACC={:.10f}".format(primary_unlabeled_metrics["ACC"]))
    print("PRIMARY_UNLABELED_NMI={:.10f}".format(primary_unlabeled_metrics["NMI"]))
    print("PRIMARY_UNLABELED_ARI={:.10f}".format(primary_unlabeled_metrics["ARI"]))
    print("SECONDARY_ALL_ACC={:.10f}".format(secondary_all_metrics["ACC"]))
    print("SECONDARY_ALL_NMI={:.10f}".format(secondary_all_metrics["NMI"]))
    print("SECONDARY_ALL_ARI={:.10f}".format(secondary_all_metrics["ARI"]))
    print("Saved: " + _display(output_dir / "e2_audit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
