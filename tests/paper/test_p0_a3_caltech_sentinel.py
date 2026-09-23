from experiments.paper.diagnostics import caltech_seed20_exact_sentinel as sentinel
from experiments.paper.diagnostics import p0_a3_protocol as protocol


def test_caltech_sentinel_is_frozen_to_one_seed_and_expected_result():
    assert protocol.CALTECH_EXPECTED_PREDICTION == (
        "b3a40a90c5069a2fe6926b48efe406dc51c670b204ccc1bd53c4d322e830d504"
    )
    assert protocol.CALTECH_EXPECTED_METRICS == {
        "acc": 0.8614285714285714,
        "nmi": 0.7734181958700173,
        "ari": 0.753482395557614,
    }
    args = sentinel.parse_args(["--output-dir", "/tmp/not-executed"])
    assert args.device == "cuda:0"
    assert not hasattr(args, "training_seed")

