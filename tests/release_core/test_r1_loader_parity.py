from pathlib import Path

import numpy as np
import scipy.io as sio

import datasets
from release_core.data.loaders import load_bdgp, load_caltech, load_msrc_v1


def _assert_outputs_equal(old, new):
    old_views, old_labels = old
    new_views, new_labels = new
    assert len(old_views) == len(new_views)
    for old_view, new_view in zip(old_views, new_views):
        assert old_view.dtype == new_view.dtype
        np.testing.assert_array_equal(old_view, new_view)
    np.testing.assert_array_equal(old_labels[0], new_labels[0])


def test_caltech_temporary_mat_parity(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.RandomState(4)
    fixture = {"Y": np.array([[0], [95], [2], [95], [4], [6]])}
    for index in range(1, 7):
        fixture["X%d" % index] = rng.normal(size=(6, index + 1)).astype(np.float64)
    path = data_dir / "Caltech.mat"
    sio.savemat(path, fixture)
    monkeypatch.chdir(tmp_path)
    old = datasets.load_data({"dataset": "Caltech-6V"})
    new = load_caltech(path)
    _assert_outputs_equal(old, new)
    assert new[1][0].tolist() == [0, 5, 2, 5, 4, 6]


def test_msrc_temporary_cell_fixture_parity(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.RandomState(8)
    labels = np.repeat(np.array([10, 20, 30, 40, 50, 60, 70]), 30)
    cells = np.empty((1, 5), dtype=object)
    cells[0, 0] = rng.normal(size=(210, 3))
    cells[0, 1] = rng.normal(size=(4, 210))
    cells[0, 2] = rng.normal(size=(210, 5))
    cells[0, 3] = rng.normal(size=(6, 210))
    cells[0, 4] = rng.normal(size=(210, 7))
    path = data_dir / "MSRC_v1.mat"
    sio.savemat(path, {"fea": cells, "gt": labels.reshape(-1, 1)})
    monkeypatch.chdir(tmp_path)
    old = datasets.load_data({"dataset": "MSRC-v1"})
    new = load_msrc_v1(path)
    _assert_outputs_equal(old, new)
    assert all(view.flags.c_contiguous for view in new[0])
    assert new[1][0].dtype == np.int64


def test_bdgp_ignores_x3_and_preserves_layout(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.RandomState(12)
    fixture = {
        "X1": np.asfortranarray(rng.normal(size=(9, 4))),
        "X2": np.asfortranarray(rng.normal(size=(9, 3))),
        "X3": np.full((9, 99), 12345.0),
        "Y": np.arange(9).reshape(-1, 1),
    }
    path = data_dir / "BDGP2V_N.mat"
    sio.savemat(path, fixture)
    monkeypatch.chdir(tmp_path)
    old = datasets.load_data({"dataset": "BDGP"})
    new = load_bdgp(path)
    _assert_outputs_equal(old, new)
    assert len(new[0]) == 2
    assert all(view.flags.c_contiguous for view in new[0])

