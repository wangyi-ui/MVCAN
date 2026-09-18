from configure import get_default_config

from release_core.config.native import get_native_config


def test_native_configs_are_exact():
    for name in ("Caltech-6V", "MSRC-v1", "BDGP"):
        assert get_native_config(name) == get_default_config(name)


def test_native_configs_are_fresh_records():
    first = get_native_config("BDGP")
    first["Autoencoder"]["arch"].append(99)
    assert get_native_config("BDGP") == get_default_config("BDGP")
