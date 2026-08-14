import os
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from ClusteringTest import test


MSRC_V1_PATHS = (
    "./data/MSRC-v1.mat",
    "./data/MSRC_v1.mat",
)


def _load_msrcv1():
    data_path = next((path for path in MSRC_V1_PATHS if os.path.isfile(path)), None)
    if data_path is None:
        expected = " or ".join(MSRC_V1_PATHS)
        raise FileNotFoundError(
            "missing MSRC-v1 data file; expected " + expected
        )

    mat = sio.loadmat(data_path)
    missing_keys = {'fea', 'gt'} - set(mat)
    if missing_keys:
        raise KeyError(
            "MSRC-v1 data file is missing MATLAB keys: "
            + ", ".join(sorted(missing_keys))
        )

    raw_views = mat['fea']
    if not isinstance(raw_views, np.ndarray) or raw_views.dtype != object:
        raise ValueError("MSRC-v1 key 'fea' must be a MATLAB cell array")
    if raw_views.size != 5:
        raise ValueError(
            "MSRC-v1 must contain 5 views; found " + str(raw_views.size)
        )

    raw_y = np.squeeze(np.asarray(mat['gt']))
    if raw_y.ndim != 1:
        raise ValueError(
            "MSRC-v1 labels must squeeze to shape [N]; found " + str(raw_y.shape)
        )
    if raw_y.shape[0] != 210:
        raise ValueError(
            "MSRC-v1 must contain 210 labels; found " + str(raw_y.shape[0])
        )
    if not np.isfinite(raw_y).all():
        raise ValueError("MSRC-v1 labels contain NaN or Inf")
    _, y = np.unique(raw_y, return_inverse=True)
    y = y.astype(np.int64, copy=False)
    if not np.array_equal(np.unique(y), np.arange(7)):
        raise ValueError("MSRC-v1 must contain exactly 7 label classes")

    # The inspected MSRC_v1.mat stores precomputed features in `fea`. The
    # requested legacy MFLVC loader is unavailable on this server, so preserve
    # those feature values without adding normalization; only orient each view
    # to [N, D] and convert it to MVCAN's float32 input dtype.
    X_list = []
    for view_index, raw_view in enumerate(raw_views.ravel(), start=1):
        view = np.asarray(raw_view)
        if view.ndim != 2:
            raise ValueError(
                "MSRC-v1 view " + str(view_index)
                + " must be a matrix; found shape " + str(view.shape)
            )
        if view.shape[0] == y.shape[0]:
            oriented_view = view
        elif view.shape[1] == y.shape[0] and view.shape[0] != y.shape[0]:
            oriented_view = view.T
        else:
            raise ValueError(
                "MSRC-v1 view " + str(view_index)
                + " has no unambiguous sample axis of length 210; found shape "
                + str(view.shape)
            )
        oriented_view = oriented_view.astype(np.float32, copy=False)
        if not np.isfinite(oriented_view).all():
            raise ValueError(
                "MSRC-v1 view " + str(view_index) + " contains NaN or Inf"
            )
        X_list.append(np.ascontiguousarray(oriented_view))

    return X_list, [y]


def load_data(config):
    """
       Load multi-view data, different datasets may need different data pre-processing methods,
       e.g., normalization, regularization, cleaning labels to [0, 1, 2 ... K-1], etc.
    """
    data_name = config['dataset']
    X_list = []
    Y_list = []

    if data_name in ['DIGIT']:
        mat = sio.loadmat("./data/DIGIT2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyDIGIT']:
        mat = sio.loadmat("./data/DIGIT2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['Amazon']:
        mat = sio.loadmat("./data/Amazon3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyAmazon']:
        mat = sio.loadmat("./data/Amazon3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        X_list.append(mat['X4'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['COIL']:
        mat = sio.loadmat("./data/COIL3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyCOIL']:
        mat = sio.loadmat("./data/COIL3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        X_list.append(mat['X4'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['BDGP']:
        mat = sio.loadmat("./data/BDGP2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyBDGP']:
        mat = sio.loadmat("./data/BDGP2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['DHA']:
        mat = sio.loadmat("./data/DHA.mat")
        X_list.append(min_max_scaler.fit_transform(mat['X1'].astype('float32')))
        X_list.append(min_max_scaler.fit_transform(mat['X2'].astype('float32')))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['Caltech-6V']:
        mat = sio.loadmat("./data/Caltech.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(min_max_scaler.fit_transform(mat['X3'].astype('float32')))
        X_list.append(mat['X4'].astype('float32'))
        X_list.append(mat['X5'].astype('float32'))
        X_list.append(mat['X6'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        # print(y[0:100])       # 7 class labels are [6  2  3  3  4  1  4  95  6  2  4  2  4  1  4  6  0 ...]
        for i in range(len(y)):
            if y[i] == 95:
                y[i] = 5        # cleaning labels to [0, 1, 2 ... K-1] for visualization
        Y_list.append(y)
        # print(y[0:100])
    elif data_name in ['RGB-D']:
        mat = sio.loadmat("./data/RGB-D.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['YoutubeVideo']:
        mat = sio.loadmat("./data/Video-3V.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y - 1)    # cleaning labels to [0, 1, 2 ... K-1] for visualization
    elif data_name in ['MSRC-v1']:
        X_list, Y_list = _load_msrcv1()

    return X_list, Y_list
