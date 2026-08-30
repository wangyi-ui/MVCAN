"""Protocol tests for C1-P0.  This suite never runs the formal evaluator."""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.cyclic_utility import c1_frozen_pseudo_supervision as c1
from experiments.cyclic_utility import c1_p0_action_compression_audit as p0
from experiments.cyclic_utility import evaluate_c1_p0_action_compression_audit as evaluate
from experiments.cyclic_utility import train_c1_frozen_pseudo_supervision as c1_train
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
C0_ARTIFACT = evaluate.DEFAULT_C0_ARTIFACT_PATH
C0_SEAL = evaluate.DEFAULT_C0_SEAL_PATH
C1A1_DIR = evaluate.DEFAULT_C1A1_OUTPUT_DIR

def _ast_callable_name(node):
    """Return a dotted callable name using only Python 3.8 AST APIs."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_callable_name(node.value)
        if prefix:
            return prefix + "." + node.attr
        return node.attr
    if isinstance(node, ast.Call):
        return _ast_callable_name(node.func)
    return ""



@pytest.fixture(scope="session")
def synthetic_c0_arrays():
    rows = torch.arange(c1.SAMPLE_NUM)[:, None]
    directions = torch.arange(c1.DIRECTION_COUNT)[None, :]
    y_gen = ((rows + 2 * directions) % c1.CLASS_NUM).long()
    confidence = 0.2 + 0.7 * (
        ((3 * rows + directions) % 17).float() / 16.0
    )
    cycle = 0.1 + 0.8 * (((rows + directions) % 13).float() / 12.0)
    shuffled = torch.roll(cycle, shifts=37, dims=0)
    return {
        "y_gen": y_gen.detach(),
        "C_conf": confidence.detach(),
        "U_cycle": cycle.detach(),
        "U_cycle_shuffle": shuffled.detach(),
    }


@pytest.fixture(scope="session")
def permutation_M0():
    return torch.stack(
        [
            torch.roll(torch.eye(c1.CLASS_NUM), shifts=view_id, dims=1)
            for view_id in range(c1.VIEW_NUM)
        ],
        dim=0,
    ).detach()


@pytest.fixture(scope="session")
def synthetic_targets(synthetic_c0_arrays, permutation_M0):
    return {
        arm: c1.build_frozen_pseudo_target(
            arm, synthetic_c0_arrays, permutation_M0
        )[0]
        for arm in p0.PSEUDO_ARMS
    }


@pytest.fixture(scope="session")
def loaded_c0():
    return c1.load_frozen_c0_artifact(C0_ARTIFACT, C0_SEAL)


class TinyAutoencoder(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.encoder_layer = torch.nn.Linear(3, 4)
        self.decoder_layer = torch.nn.Linear(4, 3)
        self.cluster_layer = torch.nn.Linear(4, c1.CLASS_NUM)
        torch.random.set_rng_state(generator_state)

    def forward(self, value):
        latent = torch.tanh(self.encoder_layer(value))
        reconstruction = self.decoder_layer(latent)
        q_local = torch.softmax(self.cluster_layer(latent), dim=-1)
        return reconstruction, latent, q_local


class TinyModel:
    def __init__(self):
        self.autoencoders = [TinyAutoencoder(100 + view) for view in range(6)]
        for autoencoder in self.autoencoders:
            autoencoder.train()


def tiny_inputs(sample_count=8):
    generator = np.random.RandomState(13)
    views = [
        generator.normal(size=(sample_count, 3)).astype(np.float32)
        for _ in range(c1.VIEW_NUM)
    ]
    target = torch.softmax(
        torch.arange(
            sample_count * c1.VIEW_NUM * c1.CLASS_NUM,
            dtype=torch.float32,
        ).reshape(sample_count, c1.VIEW_NUM, c1.CLASS_NUM)
        % 11,
        dim=-1,
    )
    pseudo_target = {
        "target_local": target.detach(),
        "sample_strength": torch.linspace(0.5, 1.5, sample_count).detach(),
    }
    batches = [
        np.arange(0, sample_count // 2, dtype=np.int64),
        np.arange(sample_count // 2, sample_count, dtype=np.int64),
    ]
    return views, pseudo_target, batches


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def test_01_stage_name_is_exact():
    assert p0.STAGE == "C1-P0"


def test_02_only_four_pseudo_arms_are_diagnosed():
    assert p0.PSEUDO_ARMS == (
        "UNIFORM",
        "CONF",
        "CYCLE",
        "SHUFFLED_CYCLE",
    )
    assert "BASE" not in p0.PSEUDO_ARMS


def test_03_comparator_order_is_frozen():
    assert p0.COMPARATORS == ("UNIFORM", "CONF", "SHUFFLED_CYCLE")


def test_04_y_gen_file_provenance(loaded_c0):
    arrays, _ = loaded_c0
    assert tensor_sha256(arrays["y_gen"].numpy()) == (
        c1.EXPECTED_C0_ARRAY_SHA256["y_gen"]
    )


@pytest.mark.parametrize(
    "name", ("C_conf", "U_cycle", "U_cycle_shuffle")
)
def test_05_weight_file_provenance(loaded_c0, name):
    arrays, _ = loaded_c0
    assert tensor_sha256(arrays[name].numpy()) == (
        c1.EXPECTED_C0_ARRAY_SHA256[name]
    )


def test_06_directional_vote_count_shape(synthetic_c0_arrays):
    record = p0.directional_hypothesis_structure(
        synthetic_c0_arrays["y_gen"]
    )
    assert record["vote_count_shape"] == [1400, 7]


def test_07_vote_purity_formula():
    y_gen = torch.zeros(c1.SAMPLE_NUM, c1.DIRECTION_COUNT, dtype=torch.long)
    y_gen[0, -5:] = 1
    record = p0.directional_hypothesis_structure(y_gen)
    assert record["vote_purity"]["max"] == 1.0
    unique = record["directional_unique_class_count_per_sample"]
    assert unique[0] == 2
    assert unique[1] == 1


def test_08_vote_purity_threshold_fractions():
    y_gen = torch.zeros(c1.SAMPLE_NUM, c1.DIRECTION_COUNT, dtype=torch.long)
    y_gen[0, -1] = 1
    record = p0.directional_hypothesis_structure(y_gen)
    purity = record["vote_purity"]
    assert purity["fraction_equal_1_0"] == pytest.approx(1399 / 1400)
    assert purity["fraction_ge_0_95"] == 1.0


def test_09_directional_histogram_sums_to_N(synthetic_c0_arrays):
    record = p0.directional_hypothesis_structure(
        synthetic_c0_arrays["y_gen"]
    )
    assert sum(record["directional_unique_class_count_histogram"].values()) == 1400


def test_10_uniform_weight_is_exact_one(synthetic_c0_arrays):
    weights, _ = p0.build_weight_diagnostics(synthetic_c0_arrays)
    assert torch.equal(weights["UNIFORM"], torch.ones(1400, 20))


def test_11_conf_weight_is_exact_C_conf(synthetic_c0_arrays):
    weights, _ = p0.build_weight_diagnostics(synthetic_c0_arrays)
    assert torch.equal(weights["CONF"], synthetic_c0_arrays["C_conf"])


def test_12_cycle_weight_is_exact_U_cycle(synthetic_c0_arrays):
    weights, _ = p0.build_weight_diagnostics(synthetic_c0_arrays)
    assert torch.equal(weights["CYCLE"], synthetic_c0_arrays["U_cycle"])


def test_13_shuffled_weight_is_exact_frozen_shuffle(synthetic_c0_arrays):
    weights, _ = p0.build_weight_diagnostics(synthetic_c0_arrays)
    assert torch.equal(
        weights["SHUFFLED_CYCLE"], synthetic_c0_arrays["U_cycle_shuffle"]
    )


def test_14_weight_pair_exact_formulas():
    cycle = np.array([[1.0, 2.0], [3.0, 4.0]])
    other = np.array([[1.0, 1.0], [5.0, 4.0]])
    record = p0.pairwise_weight_metrics(cycle, other)
    difference = cycle - other
    assert record["elementwise_exact_equality_count"] == 2
    assert record["mean_absolute_difference"] == np.abs(difference).mean()
    assert record["max_absolute_difference"] == 2.0
    assert record["RMSE"] == np.sqrt(np.square(difference).mean())


def test_15_weight_normalized_frobenius_formula():
    cycle = np.array([[1.0, 2.0]])
    other = np.array([[2.0, 0.0]])
    record = p0.pairwise_weight_metrics(cycle, other, eps=0.0)
    assert record["normalized_frobenius_distance"] == pytest.approx(
        np.linalg.norm(cycle - other) / np.linalg.norm(cycle)
    )


def test_16_constant_weight_correlation_is_explicitly_undefined():
    record = p0.pairwise_weight_metrics(np.arange(4.0), np.ones(4))
    assert record["pearson"]["value"] is None
    assert not record["pearson"]["defined"]
    assert record["spearman"]["reason"] == "constant_input"


def test_17_c1_e_reconstruction_exact(synthetic_c0_arrays):
    evidence = c1.build_sample_semantic_evidence(
        synthetic_c0_arrays["y_gen"], synthetic_c0_arrays["U_cycle"]
    )
    expected = torch.mean(
        synthetic_c0_arrays["U_cycle"].unsqueeze(-1)
        * F.one_hot(
            synthetic_c0_arrays["y_gen"], num_classes=c1.CLASS_NUM
        ).float(),
        dim=1,
    )
    assert torch.equal(evidence["evidence"], expected)


def test_18_c1_m_reconstruction_exact(synthetic_c0_arrays):
    evidence = c1.build_sample_semantic_evidence(
        synthetic_c0_arrays["y_gen"], synthetic_c0_arrays["U_cycle"]
    )
    assert torch.equal(evidence["mass"], evidence["evidence"].sum(dim=1))


def test_19_c1_T_reconstruction_exact(synthetic_c0_arrays):
    evidence = c1.build_sample_semantic_evidence(
        synthetic_c0_arrays["y_gen"], synthetic_c0_arrays["U_cycle"]
    )
    positive = evidence["mass"] > 0
    expected = torch.zeros_like(evidence["evidence"])
    expected[positive] = evidence["evidence"][positive] / evidence["mass"][
        positive, None
    ]
    assert torch.equal(evidence["target_global"], expected)


def test_20_c1_a_reconstruction_exact(synthetic_c0_arrays):
    evidence = c1.build_sample_semantic_evidence(
        synthetic_c0_arrays["y_gen"], synthetic_c0_arrays["U_cycle"]
    )
    expected = evidence["mass"] / (
        evidence["mass"].mean() + c1.MASS_EPSILON
    )
    assert torch.equal(evidence["sample_strength"], expected)


def test_21_frozen_M0_exact_from_C1_A1():
    audit = read_json(C1A1_DIR / "CYCLE" / "pseudo_target_audit.json")
    assert audit["M0_audit"]["logical_sha256"] == (
        c1.EXPECTED_M0_LOGICAL_SHA256
    )
    assert audit["M0_audit"]["matches_C0_seal_pass"]


def test_22_T_local_exact(synthetic_targets, permutation_M0):
    target = synthetic_targets["CYCLE"]
    expected = torch.stack(
        [
            target["target_global"] @ permutation_M0[view]
            for view in range(c1.VIEW_NUM)
        ],
        dim=1,
    )
    assert torch.equal(target["target_local"], expected)


def test_23_target_pair_exact_equality():
    target = torch.eye(7)[:4]
    record = p0.pairwise_target_metrics(target, target.clone())
    assert record["exact_equality"]
    assert record["logical_sha256_equality"]
    assert record["changed_argmax_count"] == 0


def test_24_target_pair_L1_L2_formulas():
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    right = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    record = p0.pairwise_target_metrics(left, right)
    assert record["mean_samplewise_L1_distance"] == 1.0
    assert record["max_samplewise_L2_distance"] == pytest.approx(np.sqrt(2.0))


def test_25_argmax_agreement_formula():
    left = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    right = torch.tensor([[0.6, 0.4], [0.9, 0.1]])
    record = p0.pairwise_target_metrics(left, right)
    assert record["argmax_agreement_fraction"] == 0.5
    assert record["changed_argmax_count"] == 1


def test_26_sample_strength_comparison_exact():
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([1.0, 1.0, 5.0])
    record = p0.pairwise_strength_metrics(left, right)
    assert record["mean_absolute_difference"] == 1.0
    assert record["max_absolute_difference"] == 2.0
    assert record["cycle_std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))


def test_27_strength_normalized_L2_formula():
    left = torch.tensor([3.0, 4.0])
    right = torch.tensor([0.0, 0.0])
    record = p0.pairwise_strength_metrics(left, right, eps=0.0)
    assert record["normalized_L2_distance"] == 1.0


def test_28_effective_action_formula(synthetic_targets):
    target = synthetic_targets["CYCLE"]
    action = p0.effective_action_tensor(
        target["sample_strength"], target["target_local"]
    )
    expected = target["sample_strength"][:, None, None] * target["target_local"]
    assert torch.equal(action, expected)


def test_29_effective_action_shape(synthetic_targets):
    target = synthetic_targets["CYCLE"]
    action = p0.effective_action_tensor(
        target["sample_strength"], target["target_local"]
    )
    assert tuple(action.shape) == (1400, 6, 7)


def test_30_effective_action_is_detached(synthetic_targets):
    target = synthetic_targets["CYCLE"]
    action = p0.effective_action_tensor(
        target["sample_strength"], target["target_local"]
    )
    assert not action.requires_grad
    assert action.grad_fn is None


def test_31_no_A_training_use():
    pseudo_source = inspect.getsource(p0.pseudo_loss_and_gradient)
    native_source = inspect.getsource(p0.native_loss_and_gradient)
    assert "effective_action_tensor" not in pseudo_source
    assert "effective_action_tensor" not in native_source


def test_32_effective_action_pair_metrics():
    left = torch.zeros(2, 6, 7)
    right = left.clone()
    left[:, :, 1] = 1.0
    right[:, :, 2] = 1.0
    record = p0.pairwise_effective_action_metrics(left, right)
    assert record["effective_action_argmax_agreement"] == 0.0
    assert record["per_sample_L1_mean"] == 12.0


def test_33_pseudo_CE_reuses_exact_C1_helper():
    source = inspect.getsource(p0.pseudo_loss_and_gradient)
    assert "c1.soft_pseudo_cross_entropy" in source
    assert "c1.pseudo_batch_by_sample_ids" in source


def test_34_pseudo_CE_formula_exact():
    q_local = torch.softmax(torch.arange(84.0).reshape(2, 6, 7), dim=-1)
    target = torch.full_like(q_local, 1.0 / 7.0)
    strength = torch.tensor([0.5, 1.5])
    actual = c1.soft_pseudo_cross_entropy(q_local, target, strength)
    expected = torch.mean(
        strength[:, None]
        * -torch.sum(target * torch.log(torch.clamp(q_local, min=1e-8)), dim=-1)
    )
    assert torch.equal(actual, expected)


def test_35_sample_order_is_canonical_and_complete():
    batches, audit = p0.c1_epoch1_batches(np.arange(1400, dtype=np.int64))
    order = np.concatenate(batches)
    assert audit["sample_ids_canonical"]
    assert np.array_equal(np.sort(order), np.arange(1400))
    assert len(np.unique(order)) == 1400


def test_36_sample_order_matches_frozen_C1_epoch1_hash():
    _, audit = p0.c1_epoch1_batches(np.arange(1400, dtype=np.int64))
    assert audit["sample_order_sha256"] == (
        "b07a17cad249a4f57b95b8248d3b904b524648d3e3c95e03f4da75a76cc12bd2"
    )


def test_37_sample_order_batch_boundaries():
    _, audit = p0.c1_epoch1_batches(np.arange(1400, dtype=np.int64))
    assert audit["batch_boundaries"] == [256, 256, 256, 256, 256, 120]


def test_38_deterministic_replay():
    first, first_audit = p0.c1_epoch1_batches(np.arange(1400, dtype=np.int64))
    second, second_audit = p0.c1_epoch1_batches(np.arange(1400, dtype=np.int64))
    assert first_audit["sample_order_sha256"] == second_audit["sample_order_sha256"]
    assert all(np.array_equal(x, y) for x, y in zip(first, second))


def test_39_noncanonical_sample_ids_hard_fail():
    with pytest.raises(RuntimeError, match="canonical"):
        p0.c1_epoch1_batches(np.arange(9, -1, -1, dtype=np.int64))


def test_40_training_sensitive_dropout_detected():
    model = TinyModel()
    model.autoencoders[0].dropout = torch.nn.Dropout(0.1)
    report = p0.training_sensitive_module_report(model)
    assert report["hard_fail_required"]
    assert report["training_sensitive_module_count"] == 1


def test_41_training_sensitive_batchnorm_detected():
    model = TinyModel()
    model.autoencoders[0].batchnorm = torch.nn.BatchNorm1d(3)
    report = p0.training_sensitive_module_report(model)
    assert report["hard_fail_required"]
    with pytest.raises(RuntimeError, match="Dropout/BatchNorm"):
        p0.require_c1_model_mode_safe(model)


def test_42_frozen_model_has_no_training_sensitive_layer():
    config = c1_train.e1_train.get_default_config("Caltech-6V")
    assert config["Autoencoder"]["batchnorm"] is False
    source = inspect.getsource(c1_train.e1_train.build_model_from_frozen_d1)
    assert "autoencoder.train()" in source


def test_43_autograd_grad_only_static_boundary():
    tree = ast.parse(inspect.getsource(p0))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "grad" in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes


def test_44_no_optimizer_is_imported_or_created():
    source = inspect.getsource(p0)
    assert "torch.optim" not in source
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert all("optim" not in name for name in imported)


def test_45_batch_gradient_weighting_formula():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    records = [(0, "p", parameter)]
    accumulator = p0.new_cpu_float64_accumulators(records)
    loss = (parameter * 3.0).sum()
    p0.accumulate_batch_gradient(loss, records, accumulator, 2, 8)
    assert accumulator[0].item() == pytest.approx(3.0 * 2.0 / 8.0)


def test_46_batch_gradient_accumulator_is_CPU_float64():
    parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    accumulator = p0.new_cpu_float64_accumulators([(0, "p", parameter)])
    assert accumulator[0].device.type == "cpu"
    assert accumulator[0].dtype == torch.float64


def test_47_autograd_grad_does_not_fill_parameter_grad():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    records = [(0, "p", parameter)]
    accumulator = p0.new_cpu_float64_accumulators(records)
    p0.accumulate_batch_gradient(parameter.square().sum(), records, accumulator, 1, 1)
    assert parameter.grad is None


def test_48_unused_parameter_accumulates_exact_zero():
    used = torch.nn.Parameter(torch.tensor([2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0]))
    records = [(0, "used", used), (0, "unused", unused)]
    accumulators = p0.new_cpu_float64_accumulators(records)
    missing = p0.accumulate_batch_gradient(
        used.square().sum(), records, accumulators, 1, 1
    )
    assert missing == [1]
    assert accumulators[1].item() == 0.0


def test_49_pseudo_gradient_no_parameter_mutation(monkeypatch):
    model = TinyModel()
    before = hash_backbone(model.autoencoders)
    monkeypatch.setattr(
        c1, "EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256", before["aggregate"]
    )
    views, target, batches = tiny_inputs()
    _, audit = p0.pseudo_loss_and_gradient(
        model, views, batches, target, torch.device("cpu")
    )
    assert audit["model_hash_before_after_equal"]
    assert hash_backbone(model.autoencoders) == before
    assert audit["parameter_grad_fields_untouched"]


def test_50_pseudo_full_dataset_loss_batch_weighted(monkeypatch):
    model = TinyModel()
    before = hash_backbone(model.autoencoders)
    monkeypatch.setattr(
        c1, "EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256", before["aggregate"]
    )
    views, target, batches = tiny_inputs()
    _, audit = p0.pseudo_loss_and_gradient(
        model, views, batches, target, torch.device("cpu")
    )
    assert np.isfinite(audit["full_dataset_mean_pseudo_loss"])
    assert audit["batch_weighting"] == "batch_sample_count / 1400"


def test_51_pseudo_gradient_covers_all_views(monkeypatch):
    model = TinyModel()
    before = hash_backbone(model.autoencoders)
    monkeypatch.setattr(
        c1, "EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256", before["aggregate"]
    )
    views, target, batches = tiny_inputs()
    _, audit = p0.pseudo_loss_and_gradient(
        model, views, batches, target, torch.device("cpu")
    )
    assert set(audit["gradient"]["per_view_norm"]) == set(map(str, range(6)))
    assert all(value > 0.0 for value in audit["gradient"]["per_view_norm"].values())


def test_52_pseudo_gradient_cosine_formula():
    cycle = [torch.tensor([1.0], dtype=torch.float64) for _ in range(6)]
    other = [torch.tensor([1.0], dtype=torch.float64) for _ in range(6)]
    record = p0.gradient_pair_metrics(cycle, other, list(range(6)))
    assert record["cosine"] == pytest.approx(1.0)
    assert record["relative_gradient_difference"] == 0.0


def test_53_pseudo_relative_gradient_distance():
    cycle = [torch.tensor([1.0], dtype=torch.float64) for _ in range(6)]
    other = [torch.tensor([0.0], dtype=torch.float64) for _ in range(6)]
    record = p0.gradient_pair_metrics(cycle, other, list(range(6)))
    assert record["relative_gradient_difference"] == pytest.approx(1.0)
    assert record["norm_ratio_to_cycle"] == 0.0


def test_54_per_view_gradient_cosine():
    cycle = [torch.tensor([1.0], dtype=torch.float64) for _ in range(6)]
    other = [torch.tensor([-1.0], dtype=torch.float64) for _ in range(6)]
    record = p0.gradient_pair_metrics(cycle, other, list(range(6)))
    assert all(value == pytest.approx(-1.0) for value in record["per_view_cosine"].values())


def test_55_native_objective_calls_exact_frozen_helpers():
    source = inspect.getsource(p0.native_loss_and_gradient)
    assert "refresh_native_target" in source
    assert "native_mvcan_losses" in source
    assert "build_lwc_loss" in source
    assert "rec_clu_total + c1_train.e1_train.LAMBDA1 * lwc_loss" in source


def test_56_native_gradient_no_parameter_mutation(monkeypatch):
    model = TinyModel()
    before = hash_backbone(model.autoencoders)
    monkeypatch.setattr(
        c1, "EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256", before["aggregate"]
    )
    views, _, batches = tiny_inputs()

    def fake_refresh(model_value, full_views, view_weights, device):
        sample_count = full_views[0].shape[0]
        p_all = torch.full((sample_count, 7), 1.0 / 7.0)
        matches = torch.stack([torch.eye(7) for _ in range(6)]).to(device)
        return p_all, matches, np.zeros(sample_count, dtype=np.int64), view_weights

    def fake_lwc(q_local, native_matches):
        return q_local.square().mean(), []

    monkeypatch.setattr(c1_train.e1_train, "refresh_native_target", fake_refresh)
    monkeypatch.setattr(c1_train, "build_lwc_loss", fake_lwc)
    _, audit = p0.native_loss_and_gradient(
        model, views, batches, torch.device("cpu")
    )
    assert audit["native_objective"] == "REC + 0.01 CLU + 0.01 LWC"
    assert audit["model_hash_before_after_equal"]
    assert hash_backbone(model.autoencoders) == before
    assert audit["gradient"]["unused_parameter_count"] == 0


def test_57_g_total_analytic_sum():
    native = [torch.tensor([2.0], dtype=torch.float64)]
    cycle = [torch.tensor([3.0], dtype=torch.float64)]
    other = [torch.tensor([5.0], dtype=torch.float64)]
    record = p0.analytic_total_gradient_metrics(native, cycle, other)
    expected_cycle = 2.0 + 0.01 * 3.0
    expected_other = 2.0 + 0.01 * 5.0
    assert record["cycle_total_norm"] == pytest.approx(expected_cycle)
    assert record["comparator_total_norm"] == pytest.approx(expected_other)


def test_58_pseudo_native_norm_ratio():
    native = [torch.tensor([4.0], dtype=torch.float64) for _ in range(6)]
    cycle = [torch.tensor([2.0], dtype=torch.float64) for _ in range(6)]
    other = [torch.tensor([1.0], dtype=torch.float64) for _ in range(6)]
    record = p0.gradient_scale_metrics(native, cycle, other, list(range(6)))
    assert record["cycle_pseudo_to_native_gradient_ratio"] == pytest.approx(0.005)


def test_59_pseudo_difference_native_ratio():
    native = [torch.tensor([4.0], dtype=torch.float64) for _ in range(6)]
    cycle = [torch.tensor([2.0], dtype=torch.float64) for _ in range(6)]
    other = [torch.tensor([1.0], dtype=torch.float64) for _ in range(6)]
    record = p0.gradient_scale_metrics(native, cycle, other, list(range(6)))
    assert record["pseudo_difference_to_native"] == pytest.approx(0.0025)


def test_60_total_gradient_cosine_and_distance():
    native = [torch.tensor([1.0], dtype=torch.float64)]
    cycle = [torch.tensor([1.0], dtype=torch.float64)]
    other = [torch.tensor([-1.0], dtype=torch.float64)]
    record = p0.analytic_total_gradient_metrics(native, cycle, other)
    assert record["cosine"] == pytest.approx(1.0)
    assert record["relative_total_gradient_difference"] == pytest.approx(0.02 / 1.01)


def test_61_no_GT_loader_in_evaluator():
    source = inspect.getsource(evaluate.run_audit)
    assert "load_labels" not in source
    assert "FULL_GT" not in source
    assert "full_gt" not in source.lower()


def test_62_no_R_loader_in_evaluator():
    source = inspect.getsource(evaluate.run_audit)
    assert "load_frozen_reliability" not in source
    assert "reliability_path" not in source


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("foo()", "foo"),
        ("module.foo()", "module.foo"),
        ("package.module.foo()", "package.module.foo"),
        ("self.foo()", "self.foo"),
    ),
)
def test_ast_callable_name_python38_compatibility(expression, expected):
    tree = ast.parse(expression)
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert _ast_callable_name(call.func) == expected


def test_ast_callable_name_detects_forbidden_nested_callable():
    tree = ast.parse("some_module.load_sparse_label_protocol()")
    call_text = [
        _ast_callable_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    assert any(
        "load_sparse_label_protocol" in value for value in call_text
    )


@pytest.mark.parametrize(
    "forbidden",
    (
        "load_sparse_label_protocol",
        "corruption_mask",
        "Memory",
        "P_corr",
        "P_util",
        "dynamic_cycle",
    ),
)
def test_63_forbidden_scientific_inputs_not_called(forbidden):
    tree = ast.parse(inspect.getsource(evaluate.run_audit))
    call_text = [
        _ast_callable_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    assert all(forbidden not in value for value in call_text)


def test_64_no_threshold_or_topk_call():
    tree = ast.parse(inspect.getsource(p0))
    call_names = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
        ).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert call_names.isdisjoint({"threshold", "topk", "quantile"})


def test_65_no_lambda_sweep():
    args = evaluate.parse_args([])
    assert vars(args) == {"device": "cuda:0"}
    source = inspect.getsource(evaluate.parse_args)
    assert "lambda" not in source.lower()


def test_66_C0_frozen_file_hashes():
    assert c1.file_sha256(C0_ARTIFACT) == c1.EXPECTED_C0_ARTIFACT_SHA256
    assert c1.file_sha256(C0_SEAL) == c1.EXPECTED_C0_SEAL_SHA256


def test_67_C1_frozen_audit_pass():
    audit = read_json(C1A1_DIR / "c1_audit.json")
    assert audit["C1_AUDIT_PASS"] is True
    assert audit["full_GT_loaded_during_training"] is False


@pytest.mark.parametrize("arm", p0.PSEUDO_ARMS)
def test_68_C1_frozen_target_hashes_present(arm):
    audit = read_json(C1A1_DIR / arm / "pseudo_target_audit.json")
    assert set(audit["logical_sha256"]) == {
        "M0",
        "evidence",
        "mass",
        "sample_strength",
        "target_global",
        "target_local",
        "weights",
        "y_gen",
    }
    assert audit["logical_sha256"]["y_gen"] == (
        c1.EXPECTED_C0_ARRAY_SHA256["y_gen"]
    )


@pytest.mark.parametrize(
    "view_id,expected",
    tuple(enumerate(c1.EXPECTED_E1_CHECKPOINT_SHA256, start=1)),
)
def test_69_E1_checkpoint_hashes(view_id, expected):
    path = evaluate.DEFAULT_E1_MODEL_DIR / (
        "Caltech-6V" + str(view_id) + "V.pth"
    )
    assert c1.file_sha256(path) == expected


def test_70_initial_E1_aggregate_hash_is_frozen():
    assert c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256 == (
        "a9b1e523c90fc450ba59fa206b78d1c02c5dd410c452766d3942cc37fc1e9d62"
    )


def test_71_output_schema_is_exact():
    assert evaluate.OUTPUT_FILES == (
        "c1p0_directional_structure.json",
        "c1p0_weight_comparison.json",
        "c1p0_target_comparison.json",
        "c1p0_strength_comparison.json",
        "c1p0_effective_action_comparison.json",
        "c1p0_pseudo_loss.json",
        "c1p0_gradient_comparison.json",
        "c1p0_native_gradient_scale.json",
        "c1p0_audit.json",
        "diagnostic_results.json",
    )


def test_72_formal_output_path_is_exact():
    assert evaluate.DEFAULT_OUTPUT_DIR == (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility/c1p0_action_compression_seed20"
    )


def test_73_evaluator_refuses_overwrite_before_work(tmp_path):
    with pytest.raises(RuntimeError, match="formal C1-P0 output path"):
        evaluate.run_audit(device="cpu", output_dir=tmp_path)


def test_74_audit_boundary_schema():
    audit = p0.audit_boundary_record()
    required_false = (
        "training_used",
        "optimizer_created",
        "optimizer_step_used",
        "parameter_update_used",
        "GT_loaded",
        "R_loaded",
        "sparse_labels_loaded",
        "corruption_mask_loaded",
        "oracle_used",
        "C0_scores_recomputed",
        "C1_formula_modified",
        "native_P_all_rewritten",
        "P_corr_used",
        "P_util_used",
        "Memory_used",
        "dynamic_cycle_used",
        "threshold_used",
        "top_k_used",
        "lambda_sweep_used",
    )
    assert all(audit[key] is False for key in required_false)
    assert audit["autograd_grad_used_for_diagnostic_only"] is True


def test_75_allowed_descriptive_labels_are_exact():
    assert p0.ALLOWED_DESCRIPTIVE_LABELS == (
        "TARGET_AGGREGATION_COMPRESSION_EVIDENCE",
        "GRADIENT_CARRIER_COMPRESSION_EVIDENCE",
        "NATIVE_GRADIENT_DOMINANCE_EVIDENCE",
        "NO_CLEAR_COMPRESSION_LOCATION",
    )


def test_76_no_automatic_descriptive_gate():
    audit = p0.audit_boundary_record()
    assert audit["scientific_gate_used"] is False
    assert audit["automatic_descriptive_label_selected"] is False


def test_77_target_reconstruction_calls_only_frozen_C1_builder():
    source = inspect.getsource(p0.reconstruct_frozen_targets)
    assert "c1.build_frozen_pseudo_target" in source
    assert "pseudo_target_audit.json" in source
    assert "all(checked.values())" in source


def test_78_effective_action_is_declared_diagnostic_only(synthetic_targets):
    provenance = {
        arm: {"all_available_hashes_equal": True} for arm in p0.PSEUDO_ARMS
    }
    _, _, _, record = p0.build_target_action_diagnostics(
        synthetic_targets, provenance, {"matches_C0_seal_pass": True}
    )
    assert record["diagnostic_only"] is True
    assert record["used_for_training"] is False


def test_79_core_module_never_writes_outputs():
    source = inspect.getsource(p0)
    assert "write_json" not in source
    assert "mkdir" not in source


def test_80_evaluator_does_not_import_summarizer():
    source = inspect.getsource(evaluate)
    assert "summarize_c1" not in source
    assert "summarize(" not in source
