"""Frozen P0-A3 paths, hashes, and non-tunable diagnostic constants."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_ROOT = Path("/root/autodl-tmp/CVPR24-MVCAN")

MSRC_INPUT_DIR = REPOSITORY_ROOT / (
    "outputs/paper/transfer_audit/MSRC-v1/snr2p5_half_seed20/inputs"
)
MSRC_A2_ROOT = REPOSITORY_ROOT / (
    "outputs/paper/transfer_diagnostics/MSRC-v1/"
    "p0_a2_same_condition_init_seed20"
)
MSRC_INIT_DIR = MSRC_A2_ROOT / "init"
MSRC_TRUE_U_DIR = MSRC_A2_ROOT / "true_u"

HISTORICAL_MSRC_ROOT = HISTORICAL_ROOT / (
    "outputs/generic_contract/g0_b0_msrc_structural_pilot_seed20_20260916"
)
HISTORICAL_MSRC_ARTIFACT = HISTORICAL_MSRC_ROOT / (
    "msrc_structural_pre_gt_artifact.npz"
)
HISTORICAL_MSRC_AUDIT = HISTORICAL_MSRC_ROOT / (
    "msrc_structural_pre_gt_audit.json"
)
HISTORICAL_MSRC_SEAL = HISTORICAL_MSRC_ROOT / (
    "msrc_structural_pre_gt_seal.json"
)
HISTORICAL_MSRC_FILES = {
    HISTORICAL_MSRC_ARTIFACT: "b04f107f6279839d534e55500fbb0f9aba94d7686944ab20f255337a511af234",
    HISTORICAL_MSRC_AUDIT: "7674a6d46f5092e20a9be8f166cb6d28c29f10422176f79aad259b20ce291b1e",
    HISTORICAL_MSRC_SEAL: "c2fd8b35ffe0a065dbc8df69b7488fd75d88d39c2d3a100e665cc014729fa50e",
}
HISTORICAL_MSRC_LOGICAL = {
    "corruption_mask": "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d",
    "split": "4f16e8b1b8213591a7ea1d36115335ccee43bf568d32bbcb007f929bc9b15231",
    "U_cycle": "3552d88de9f894276d0d1ccae8f4c55147e98aac9a8ae146eaa1cf54dcc9dd10",
    "PredRelation_true": "cbd2d2de1dfd51abba9b8e80fcc1ee82f4e902df6532ca2af388e1266ff0d2b8",
    "relation_balance_weights_true": "2d69642382947465851d69fe9f151df440979ce626481f9ad4c060909f92ff9a",
    "final_predictions": "d5679cff6f7e6faa6a269dd4deeb020e64e331511ed9cb6632be18459d48251d",
}

CALTECH_FEATURE = HISTORICAL_ROOT / (
    "outputs/e0_glgc_adapter/caltech6v_snr2p5_k3_seed20.npz"
)
CALTECH_FEATURE_AUDIT = HISTORICAL_ROOT / "outputs/e0_glgc_adapter/export_audit.json"
CALTECH_ACTION = HISTORICAL_ROOT / (
    "outputs/cyclic_utility/c3_a0_utility_conditioned_action_granularity_seed20/"
    "c3_a0_action_pre_gt.npz"
)
CALTECH_ACTION_SEAL = CALTECH_ACTION.parent / "c3_a0_action_seal.json"
CALTECH_CHECKPOINT_ROOT = HISTORICAL_ROOT / (
    "outputs/e1_pairwise_utility/lwc_100ep_seed20/models"
)
CALTECH_CHECKPOINTS = tuple(
    CALTECH_CHECKPOINT_ROOT / ("Caltech-6V%dV.pth" % index)
    for index in range(1, 7)
)
CALTECH_CHECKPOINT_AUDIT = HISTORICAL_ROOT / (
    "outputs/e1_pairwise_utility/lwc_100ep_seed20/e1_audit.json"
)
CALTECH_REFERENCE_ROOT = HISTORICAL_ROOT / (
    "outputs/final_core/f0_a0_engineering_smoke_seed20"
)
CALTECH_REFERENCE_ARTIFACT = CALTECH_REFERENCE_ROOT / "final_core_pre_gt_artifact.npz"
CALTECH_REFERENCE_AUDIT = CALTECH_REFERENCE_ROOT / "final_core_pre_gt_audit.json"
CALTECH_REFERENCE_SEAL = CALTECH_REFERENCE_ROOT / "final_core_pre_gt_seal.json"
CALTECH_FULL_GT = HISTORICAL_ROOT / "data/Caltech.mat"

CALTECH_INPUT_FILES = {
    CALTECH_FEATURE: "44133379b756fa83744d7745dd477c409d77db3f078ccab6064643c85428d8e4",
    CALTECH_FEATURE_AUDIT: "c2b720b6acd84670e9645cedc5209970300ea09478d91005dca1a7a9e6c3399b",
    CALTECH_ACTION: "b4f9ff0241b28ec9f6a8ec6c55c0f9da9c4631ffde2b032c0d3550340fdece71",
    CALTECH_ACTION_SEAL: "0e23e9cea448435d04351fc6661ab8020a69b12a660f6e6604ff855ea96bb9d0",
    CALTECH_CHECKPOINT_AUDIT: "730c30d995a7dd292eec674a264dec82857e46427a4d1fbd5cf46ea367b22dd8",
    CALTECH_CHECKPOINTS[0]: "b914af6e3740d1808d44ed2592b012dd67bb90356b745c2c117d63743256d77c",
    CALTECH_CHECKPOINTS[1]: "7c5ce3be3fc2a62b0731b58651e29af2da6e2dfd2eb51f53d2e574e019e3c576",
    CALTECH_CHECKPOINTS[2]: "69d65bfeb51617a3a560aa040e7be8c2ad9e2ef90368ca052914c49fb04a28ac",
    CALTECH_CHECKPOINTS[3]: "cd629a4b604e814595d1f9c9e720a94cdcf805d871863b457b95b4f97866a33c",
    CALTECH_CHECKPOINTS[4]: "6268ceeda5c9cf1dedc0de1550888da0d61e25ca1f786cd6c181639257df692d",
    CALTECH_CHECKPOINTS[5]: "8a9117c23a45b24e4e6641746e4f67c864c0a0638edbe410f4cf7866c79a1701",
}
CALTECH_REFERENCE_FILES = {
    CALTECH_REFERENCE_ARTIFACT: "ec78a6102546254c7d0868f4948e1a79a236734ce37ab9d67b350063fefcc010",
    CALTECH_REFERENCE_AUDIT: "255234648a40e393059d097bfecba0e0ee822b38f3b6edae1c5f9a40c87302d4",
    CALTECH_REFERENCE_SEAL: "a85828fa1c914194b97df5704c99bfbabef720e33fdf60359d1c4916a870e54f",
}
CALTECH_EXPECTED_INITIAL = "64faa7cb90c1319478575ce579d134b475ab9f78484cd1bb1cb7e86e1e06f991"
CALTECH_EXPECTED_PREDICTION = "b3a40a90c5069a2fe6926b48efe406dc51c670b204ccc1bd53c4d322e830d504"
CALTECH_EXPECTED_METRICS = {
    "acc": 0.8614285714285714,
    "nmi": 0.7734181958700173,
    "ari": 0.753482395557614,
}

OUTPUT_ROOT = REPOSITORY_ROOT / (
    "outputs/paper/diagnostics/p0_a3_cross_dataset_seed20"
)
PERMUTATION_SEED = 20
PERMUTATION_COUNT = 1000

