import os
import unittest

import numpy as np
import scipy.io as sio

from configure import get_default_config
from datasets import MSRC_V1_PATHS, load_data


def _msrcv1_data_path():
    for path in MSRC_V1_PATHS:
        if os.path.isfile(path):
            return path
    raise unittest.SkipTest(
        "missing MSRC-v1 data file: expected " + " or ".join(MSRC_V1_PATHS)
    )


def test_msrcv1_default_config():
    config = get_default_config("MSRC-v1")

    assert config['Autoencoder'] == {
        'arch': [10],
        'channal': [1],
        'activations': 'relu',
        'batchnorm': False,
        'FCN': True,
    }
    assert config['training'] == {
        'seed': 20,
        'batch_size': 256,
        'init_epoch': 200,
        'T_1': 2,
        'T_2': 100,
        'epoch': 1000,
        'lr': 0.0001,
        'lambda1': 0.01,
    }


def test_msrcv1_loader_uses_aligned_precomputed_features():
    data_path = _msrcv1_data_path()
    config = get_default_config("MSRC-v1")
    config['dataset'] = "MSRC-v1"

    X_list, Y_list = load_data(config)
    y = Y_list[0]

    assert len(X_list) == 5
    assert len(Y_list) == 1
    assert y.shape == (210,)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.array_equal(np.unique(y), np.arange(7))

    for X in X_list:
        assert X.shape[0] == 210
        assert X.dtype == np.float32
        assert np.isfinite(X).all()

    mat = sio.loadmat(data_path)
    raw_y = np.squeeze(np.asarray(mat['gt']))
    _, expected_y = np.unique(raw_y, return_inverse=True)
    assert np.array_equal(y, expected_y)

    # Exact equality to each source cell after only orientation and float32
    # conversion proves that views were not separately shuffled and that
    # labels were not used to preprocess feature values.
    for X, raw_view in zip(X_list, mat['fea'].ravel()):
        raw_view = np.asarray(raw_view)
        if raw_view.shape[0] == y.shape[0]:
            expected_view = raw_view
        elif raw_view.shape[1] == y.shape[0] and raw_view.shape[0] != y.shape[0]:
            expected_view = raw_view.T
        else:
            raise AssertionError("source view has no valid MSRC-v1 sample axis")
        assert np.array_equal(X, expected_view.astype(np.float32))


if __name__ == '__main__':
    tests = (
        test_msrcv1_default_config,
        test_msrcv1_loader_uses_aligned_precomputed_features,
    )
    passed = 0
    for test_case in tests:
        test_case()
        passed += 1
        print(test_case.__name__ + ": PASS")
    print(str(passed) + " tests passed")
