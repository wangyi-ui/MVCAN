import numpy as np

from experiments.paper.diagnostics.relation_utility_quality import permutation_control


def test_permutation_control_is_deterministic_and_preserves_action_marginals():
    utility = np.arange(1, 13, dtype=np.float64).reshape(6, 2)
    correct = np.zeros((6, 2, 2), dtype=np.bool_)
    correct[:3] = True
    balance = np.ones_like(correct, dtype=np.float64)
    first = permutation_control(utility, correct, balance, count=25, seed=20)
    second = permutation_control(utility, correct, balance, count=25, seed=20)
    assert first == second
    assert first["permutations"] == 25
    assert 1.0 / 26.0 <= first["empirical_one_sided_p"] <= 1.0

