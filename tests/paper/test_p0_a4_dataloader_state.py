import torch

from experiments.paper.diagnostics.msrc_native_stagewise_parity import (
    preview_first_batch_ids,
)


def test_preview_matches_dataloader_and_does_not_advance_live_generator():
    dataset = torch.utils.data.TensorDataset(torch.arange(210, dtype=torch.int64))
    generator = torch.Generator(device="cpu").manual_seed(20)
    before = generator.get_state().clone()
    preview = preview_first_batch_ids(generator, len(dataset), 256)
    assert torch.equal(before, generator.get_state())

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, drop_last=False,
        generator=generator,
    )
    actual = next(iter(loader))[0].tolist()[:32]
    assert preview == actual


def test_selected_epoch_preview_is_seed_and_trajectory_sensitive():
    left = torch.Generator(device="cpu").manual_seed(20)
    right = torch.Generator(device="cpu").manual_seed(21)
    assert preview_first_batch_ids(left, 210, 256) != preview_first_batch_ids(
        right, 210, 256
    )
