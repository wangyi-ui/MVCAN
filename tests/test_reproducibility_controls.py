import os
import random
import sys
from unittest import mock

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from configure import get_default_config
from model import MvCAN


with mock.patch.object(sys, 'argv', ['run.py']):
    import run


def test_set_global_seed_replays_python_numpy_and_torch_rngs():
    run.set_global_seed(20)
    first_python = [random.random() for _ in range(4)]
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)

    run.set_global_seed(20)
    second_python = [random.random() for _ in range(4)]
    second_numpy = np.random.random(4)
    second_torch = torch.rand(4)

    assert first_python == second_python
    assert np.array_equal(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)
    assert os.environ['PYTHONHASHSEED'] == '20'
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_kmeans_random_state_reproduces_cluster_assignments():
    toy_dataset = np.array([
        [-3.0, -3.0], [-3.2, -2.8], [-2.8, -3.1],
        [0.0, 4.0], [0.2, 4.2], [-0.2, 3.8],
        [4.0, 0.0], [4.2, 0.1], [3.8, -0.2],
    ])
    first = KMeans(n_clusters=3, n_init=100, random_state=20)
    second = KMeans(n_clusters=3, n_init=100, random_state=20)

    first_labels = first.fit_predict(toy_dataset)
    second_labels = second.fit_predict(toy_dataset)

    assert adjusted_rand_score(first_labels, second_labels) == 1.0


def test_mvcan_records_constructor_seed():
    config = get_default_config('MSRC-v1')
    model = MvCAN(
        config,
        view_num=1,
        view_size=[4],
        n_clusters=2,
        seed=20,
        data_size=9,
    )

    assert model.seed == 20


if __name__ == '__main__':
    tests = (
        test_set_global_seed_replays_python_numpy_and_torch_rngs,
        test_kmeans_random_state_reproduces_cluster_assignments,
        test_mvcan_records_constructor_seed,
    )
    passed = 0
    for test_case in tests:
        test_case()
        passed += 1
        print(test_case.__name__ + ': PASS')
    print(str(passed) + ' tests passed')
