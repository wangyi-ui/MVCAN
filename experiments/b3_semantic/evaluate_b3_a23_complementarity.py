"""B3-A2.3 offline Q/T complementarity diagnostic.

This script performs no neural-network training, no backward(),
and no parameter update.

Q: OOF target-only centroid cosine evidence.
T: OOF cross-view Ridge consensus predictability evidence.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.linear_model import Ridge

from configure import get_default_config
from datasets import load_data
from model import MvCAN
from weak_quality import apply_weak_quality_protocol

from irv.b3_predictability_diagnostics import (
    make_fold_assignment,
    fold_assignment_sha256,
    oof_ridge_predictability,
    bootstrap_reliability,
)
from irv.b3_reliability_replication import (
    oof_target_only_centroid_score,
    corruption_label_permutation_test,
    per_view_reliability,
)


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "experiments/b3_semantic/b3_a22_condition_manifest.json"
)

OUT = (
    ROOT
    / "outputs/b3_semantic/a23_reproduction"
)

OUT.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "MSRC-v1"

N = 210
V = 5
K = 7
LATENT_DIM = 10

SEEDS = [20, 30, 50]
SNR_DB = 2.5

RIDGE_ALPHA = 1.0

BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260815

PERMUTATION_REPEATS = 2000
PERMUTATION_SEED = 20260815


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def crossfit_residual(target, predictor, folds):
    """
    Per-view label-free OOF residualization.

    target:
        [N, V]

    predictor:
        [N, V]

    folds:
        [N]
    """
    target = np.asarray(target, dtype=np.float64)
    predictor = np.asarray(predictor, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int64)

    if target.shape != (N, V):
        raise RuntimeError("target shape mismatch")

    if predictor.shape != (N, V):
        raise RuntimeError("predictor shape mismatch")

    if folds.shape != (N,):
        raise RuntimeError("fold shape mismatch")

    residual = np.empty_like(target)
    prediction = np.empty_like(target)

    coverage = np.zeros(
        (N, V),
        dtype=np.int64,
    )

    for view_id in range(V):

        for fold_id in np.unique(folds):

            train_ids = np.flatnonzero(
                folds != fold_id
            )

            test_ids = np.flatnonzero(
                folds == fold_id
            )

            # [N_train, 1]
            x_train = predictor[
                train_ids,
                view_id:view_id + 1
            ]

            # [N_train]
            y_train = target[
                train_ids,
                view_id
            ]

            # [N_test, 1]
            x_test = predictor[
                test_ids,
                view_id:view_id + 1
            ]

            model = Ridge(
                alpha=RIDGE_ALPHA,
                fit_intercept=True,
            )

            model.fit(
                x_train,
                y_train,
            )

            pred = model.predict(
                x_test
            )

            prediction[
                test_ids,
                view_id
            ] = pred

            residual[
                test_ids,
                view_id
            ] = (
                target[
                    test_ids,
                    view_id
                ]
                - pred
            )

            coverage[
                test_ids,
                view_id
            ] += 1

    if not np.all(coverage == 1):
        raise RuntimeError(
            "OOF residual coverage failed"
        )

    if not np.isfinite(residual).all():
        raise RuntimeError(
            "non-finite residual"
        )

    return residual, prediction


def rank_percentile_per_view(scores):
    """
    Per-view empirical-rank normalization.

    scores:
        [N, V]
    """
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    output = np.empty_like(scores)

    for view_id in range(V):

        ranks = rankdata(
            scores[:, view_id],
            method="average",
        )

        output[:, view_id] = (
            ranks - 1.0
        ) / float(N - 1)

    return output


def compact_metrics(values):
    return {
        "clean_mean": float(
            values[
                "clean_predictability_mean"
            ]
        ),
        "corrupted_mean": float(
            values[
                "corrupted_predictability_mean"
            ]
        ),
        "gap": float(
            values[
                "predictability_gap"
            ]
        ),
        "gap_ci_low": float(
            values["gap_ci_low"]
        ),
        "gap_ci_high": float(
            values["gap_ci_high"]
        ),
        "auc": float(
            values[
                "clean_corrupted_auc"
            ]
        ),
        "auc_ci_low": float(
            values["auc_ci_low"]
        ),
        "auc_ci_high": float(
            values["auc_ci_high"]
        ),
        "spearman": float(
            values[
                "spearman_clean_indicator_vs_predictability"
            ]
        ),
        "spearman_ci_low": float(
            values["spearman_ci_low"]
        ),
        "spearman_ci_high": float(
            values["spearman_ci_high"]
        ),
    }


manifest = json.loads(
    MANIFEST.read_text()
)

entries = [
    entry
    for entry in manifest
    if int(entry["model_seed"]) in SEEDS
    and abs(
        float(entry["snr_db"])
        - SNR_DB
    ) < 1e-12
]

entries = sorted(
    entries,
    key=lambda x: int(x["model_seed"]),
)

found_seeds = [
    int(entry["model_seed"])
    for entry in entries
]

if found_seeds != SEEDS:
    raise RuntimeError(
        "missing SNR2.5 seeds: "
        + str(found_seeds)
    )


config = get_default_config(
    DATASET_NAME
)

config["dataset"] = DATASET_NAME

clean_views, _ = load_data(config)

if len(clean_views) != V:
    raise RuntimeError(
        "view count mismatch"
    )


results = {}

conditional_passes = []
specificity_passes = []


for entry in entries:

    seed = int(
        entry["model_seed"]
    )

    corruption_seed = int(
        entry["corruption_seed"]
    )

    print()
    print("=" * 80)
    print(
        "B3-A2.3 SNR2.5 seed",
        seed,
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Reconstruct frozen B2 corruption.
    # --------------------------------------------------------

    evaluation_views, corruption_audit = (
        apply_weak_quality_protocol(
            clean_views,
            mode="heterogeneous_gaussian",
            k=2,
            snr_db=2.5,
            corruption_seed=corruption_seed,
        )
    )

    mask = np.load(
        resolve(
            entry["corruption_mask"]
        ),
        allow_pickle=False,
    ).astype(bool)

    if mask.shape != (N, V):
        raise RuntimeError(
            "mask shape mismatch"
        )

    if int(mask.sum()) != 420:
        raise RuntimeError(
            "mask count mismatch"
        )

    if not np.all(
        mask.sum(axis=1) == 2
    ):
        raise RuntimeError(
            "row corruption count mismatch"
        )

    if not np.array_equal(
        mask.sum(axis=0),
        np.full(V, 84),
    ):
        raise RuntimeError(
            "view corruption count mismatch"
        )

    if (
        corruption_audit[
            "mask_sha256"
        ]
        != entry[
            "expected_mask_sha256"
        ]
    ):
        raise RuntimeError(
            "mask hash mismatch"
        )

    # --------------------------------------------------------
    # Load frozen B2 Native MVCAN.
    # --------------------------------------------------------

    view_sizes = [
        int(view.shape[1])
        for view in evaluation_views
    ]

    models = MvCAN(
        config,
        view_num=V,
        view_size=view_sizes,
        n_clusters=K,
        seed=seed,
        data_size=N,
        semantic_config=None,
    )

    backbone_dir = resolve(
        entry["backbone_dir"]
    )

    for view_id, autoencoder in enumerate(
        models.autoencoders
    ):

        checkpoint = (
            backbone_dir
            / (
                DATASET_NAME
                + str(view_id + 1)
                + "V.pth"
            )
        )

        autoencoder.load_state_dict(
            torch.load(
                checkpoint,
                map_location="cpu",
            ),
            strict=True,
        )

        autoencoder.eval()

    # --------------------------------------------------------
    # Frozen Native z extraction.
    #
    # X_v: [N, input_dim_v]
    # Z_v: [N, latent_dim=10]
    # --------------------------------------------------------

    native_views = []

    with torch.no_grad():

        for view_id, autoencoder in enumerate(
            models.autoencoders
        ):

            X_v = torch.from_numpy(
                evaluation_views[
                    view_id
                ]
            ).float()

            Z_v = autoencoder.encoder(
                X_v
            )

            Z_v = F.normalize(
                Z_v,
                p=2,
                dim=1,
                eps=1e-12,
            )

            native_views.append(
                Z_v.cpu().numpy()
            )

    for Z_v in native_views:
        if Z_v.shape != (
            N,
            LATENT_DIM,
        ):
            raise RuntimeError(
                "Native z shape mismatch"
            )

    # --------------------------------------------------------
    # Deterministic sample-level OOF folds.
    # --------------------------------------------------------

    folds = make_fold_assignment(
        N,
        n_splits=5,
        random_state=seed,
    )

    fold_hash = (
        fold_assignment_sha256(
            folds
        )
    )

    # --------------------------------------------------------
    # T:
    # Cross-view predictability evidence.
    #
    # T: [N, V]
    # --------------------------------------------------------

    prediction_result = (
        oof_ridge_predictability(
            native_views,
            folds,
            alpha=RIDGE_ALPHA,
        )
    )

    T = np.asarray(
        prediction_result[
            "oof_consensus_cosine"
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Q:
    # OOF target-only intrinsic evidence.
    #
    # Q: [N, V]
    # --------------------------------------------------------

    Q = np.stack(
        [
            oof_target_only_centroid_score(
                native_views[view_id],
                folds,
            )
            for view_id in range(V)
        ],
        axis=1,
    )

    if Q.shape != (N, V):
        raise RuntimeError(
            "Q shape mismatch"
        )

    if T.shape != (N, V):
        raise RuntimeError(
            "T shape mismatch"
        )

    # --------------------------------------------------------
    # Raw Q/T reliability.
    # --------------------------------------------------------

    raw = bootstrap_reliability(
        {
            "Q": Q,
            "T": T,
        },
        mask,
        repeats=BOOTSTRAP_REPEATS,
        seed=BOOTSTRAP_SEED,
    )

    Q_raw = raw["arms"]["Q"]
    T_raw = raw["arms"]["T"]

    pearson = float(
        pearsonr(
            Q.reshape(-1),
            T.reshape(-1),
        )[0]
    )

    spearman = float(
        spearmanr(
            Q.reshape(-1),
            T.reshape(-1),
        ).correlation
    )

    # --------------------------------------------------------
    # Conditional evidence:
    #
    # T | Q
    # Q | T
    # --------------------------------------------------------

    T_residual, T_hat_from_Q = (
        crossfit_residual(
            target=T,
            predictor=Q,
            folds=folds,
        )
    )

    Q_residual, Q_hat_from_T = (
        crossfit_residual(
            target=Q,
            predictor=T,
            folds=folds,
        )
    )

    residual = bootstrap_reliability(
        {
            "T_given_Q": T_residual,
            "Q_given_T": Q_residual,
        },
        mask,
        repeats=BOOTSTRAP_REPEATS,
        seed=BOOTSTRAP_SEED,
    )

    T_given_Q = (
        residual["arms"]["T_given_Q"]
    )

    Q_given_T = (
        residual["arms"]["Q_given_T"]
    )

    # --------------------------------------------------------
    # Conditional permutation specificity.
    # --------------------------------------------------------

    residual_permutation = (
        corruption_label_permutation_test(
            T_residual,
            mask,
            repeats=PERMUTATION_REPEATS,
            seed=PERMUTATION_SEED,
        )
    )

    # --------------------------------------------------------
    # Unsupervised equal-rank diagnostic fusion.
    #
    # This is NOT Information Utility U.
    # --------------------------------------------------------

    Q_rank = (
        rank_percentile_per_view(Q)
    )

    T_rank = (
        rank_percentile_per_view(T)
    )

    equal_rank = (
        0.5 * Q_rank
        + 0.5 * T_rank
    )

    equal_rank_result = (
        bootstrap_reliability(
            {
                "equal_rank":
                    equal_rank
            },
            mask,
            repeats=BOOTSTRAP_REPEATS,
            seed=BOOTSTRAP_SEED,
        )["arms"]["equal_rank"]
    )

    # --------------------------------------------------------
    # Seed20 per-view T|Q diagnostic.
    # --------------------------------------------------------

    per_view = None

    if seed == 20:

        per_view = per_view_reliability(
            T_residual,
            mask,
            repeats=BOOTSTRAP_REPEATS,
            seed=BOOTSTRAP_SEED,
        )

    conditional_pass = bool(
        T_given_Q[
            "auc_ci_low"
        ] > 0.5
        and T_given_Q[
            "gap_ci_low"
        ] > 0.0
    )

    specificity_pass = bool(
        residual_permutation[
            "real_auc"
        ]
        >
        residual_permutation[
            "permutation_auc_p95"
        ]
        and residual_permutation[
            "p_auc"
        ] < 0.01
        and residual_permutation[
            "real_gap"
        ]
        >
        residual_permutation[
            "permutation_gap_p95"
        ]
        and residual_permutation[
            "p_gap"
        ] < 0.01
    )

    conditional_passes.append(
        conditional_pass
    )

    specificity_passes.append(
        specificity_pass
    )

    np.savez_compressed(
        OUT
        / (
            "a23_scores_snr2p5_seed"
            + str(seed)
            + ".npz"
        ),
        Q=Q,
        T=T,
        T_residual_given_Q=T_residual,
        Q_residual_given_T=Q_residual,
        equal_rank_fusion=equal_rank,
        corruption_mask=mask,
        fold_assignment=folds,
    )

    results[str(seed)] = {
        "fold_sha256": fold_hash,

        "qt_correlation": {
            "pearson": pearson,
            "spearman": spearman,
        },

        "Q_raw":
            compact_metrics(Q_raw),

        "T_raw":
            compact_metrics(T_raw),

        "T_residual_given_Q":
            compact_metrics(T_given_Q),

        "Q_residual_given_T":
            compact_metrics(Q_given_T),

        "equal_rank_QT_fusion":
            compact_metrics(
                equal_rank_result
            ),

        "T_given_Q_conditional_pass":
            conditional_pass,

        "T_given_Q_permutation":
            residual_permutation,

        "conditional_specificity_pass":
            specificity_pass,

        "per_view_T_given_Q_seed20":
            per_view,
    }

    print(
        "Q AUC =",
        Q_raw[
            "clean_corrupted_auc"
        ],
    )

    print(
        "T AUC =",
        T_raw[
            "clean_corrupted_auc"
        ],
    )

    print(
        "T|Q AUC =",
        T_given_Q[
            "clean_corrupted_auc"
        ],
        "CI =",
        [
            T_given_Q[
                "auc_ci_low"
            ],
            T_given_Q[
                "auc_ci_high"
            ],
        ],
    )

    print(
        "T|Q gap =",
        T_given_Q[
            "predictability_gap"
        ],
        "CI =",
        [
            T_given_Q[
                "gap_ci_low"
            ],
            T_given_Q[
                "gap_ci_high"
            ],
        ],
    )


multiseed_pass = bool(
    all(conditional_passes)
)

specificity_pass = bool(
    all(specificity_passes)
)

strong_candidate = bool(
    multiseed_pass
    and specificity_pass
)


summary = {
    "stage":
        "B3-A2.3-reproduction",

    "definition": {
        "Q":
            "OOF target-view training-centroid cosine",

        "T":
            "OOF cross-view Ridge consensus cosine",

        "residualization":
            "per-view label-free 5-fold Ridge alpha=1.0",
    },

    "seeds":
        results,

    "gates": {
        "B3_A23_ENGINEERING_PASS":
            True,

        "B3_A23_T_CONDITIONAL_MULTISEED_PASS":
            multiseed_pass,

        "B3_A23_CONDITIONAL_SPECIFICITY_PASS":
            specificity_pass,

        "B3_RELIABILITY_DECOMPOSITION_STRONG_CANDIDATE":
            strong_candidate,

        "B4_READY_DECLARED":
            False,
    },
}


output_path = (
    OUT
    / "b3_a23_reproduction_summary.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as output_file:

    json.dump(
        summary,
        output_file,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    output_file.write("\n")


print()
print("=" * 80)
print("FINAL GATES")
print("=" * 80)

for key, value in summary[
    "gates"
].items():

    print(
        key,
        "=",
        value,
    )

print()
print(
    "Saved:",
    output_path,
)
