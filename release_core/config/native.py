"""Frozen dataset-native backbone configurations."""


_TRAINING = {
    "Caltech-6V": (5, 0.01),
    "MSRC-v1": (20, 0.01),
    "BDGP": (1, 10),
}


def get_native_config(data_name):
    """Return a fresh copy of the frozen native configuration."""
    if data_name not in _TRAINING:
        raise ValueError("unsupported dataset: " + str(data_name))
    seed, lambda1 = _TRAINING[data_name]
    return {
        "Autoencoder": {
            "arch": [10],
            "channal": [1],
            "activations": "relu",
            "batchnorm": False,
            "FCN": True,
        },
        "training": {
            "seed": seed,
            "batch_size": 256,
            "init_epoch": 200,
            "T_1": 2,
            "T_2": 100,
            "epoch": 1000,
            "lr": 0.0001,
            "lambda1": lambda1,
        },
    }
