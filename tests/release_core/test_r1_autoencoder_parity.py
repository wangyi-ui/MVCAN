import torch

from model import Autoencoder as LegacyAutoencoder
from release_core.backbone.autoencoder import Autoencoder
from release_core.backbone.multiview import MultiViewBackbone
from release_core.config.native import get_native_config


def _models(seed=314159):
    arguments = ([7, 500, 500, 2000, 10], "relu", False, 4, True, [1])
    torch.manual_seed(seed)
    legacy = LegacyAutoencoder(*arguments)
    torch.manual_seed(seed)
    clean = Autoencoder(*arguments)
    return legacy, clean


def test_autoencoder_initialization_exact():
    legacy, clean = _models()
    legacy_state = legacy.state_dict()
    clean_state = clean.state_dict()
    assert tuple(legacy_state) == tuple(clean_state)
    assert sum(value.numel() for value in legacy.parameters()) == sum(
        value.numel() for value in clean.parameters()
    )
    for key in legacy_state:
        assert legacy_state[key].shape == clean_state[key].shape
        assert legacy_state[key].dtype == clean_state[key].dtype
        assert torch.equal(legacy_state[key], clean_state[key])


def test_autoencoder_forward_exact_cpu_float32():
    legacy, clean = _models()
    values = torch.arange(35, dtype=torch.float32).reshape(5, 7) / 17.0
    with torch.no_grad():
        legacy_outputs = legacy(values)
        clean_outputs = clean(values)
    for old, new in zip(legacy_outputs, clean_outputs):
        assert old.shape == new.shape
        assert old.dtype == new.dtype == torch.float32
        assert torch.equal(old, new)


def test_multiview_container_preserves_per_view_initialization():
    config = get_native_config("Caltech-6V")
    clean = MultiViewBackbone(config, 2, [7, 9], n_clusters=4, seed=23)
    for index, dimension in enumerate((7, 9)):
        torch.manual_seed(23)
        torch.cuda.manual_seed(23)
        legacy = LegacyAutoencoder(
            [dimension, 500, 500, 2000, 10], "relu", False, 4, True, [1]
        )
        for key, value in legacy.state_dict().items():
            assert torch.equal(value, clean.autoencoders[index].state_dict()[key])
