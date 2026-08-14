"""Tensor-level hashing helpers for the B3 backbone protection audit."""

import hashlib
import struct

import torch


def _update_length_prefixed(digest, value):
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def hash_state_dict(state_dict):
    """Hash sorted tensor keys, dtypes, shapes, and contiguous raw bytes."""
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise TypeError("state_dict value for " + str(key) + " is not a tensor")

        contiguous = tensor.detach().cpu().contiguous()
        byte_view = contiguous.reshape(-1).view(torch.uint8)
        raw_bytes = byte_view.numpy().tobytes(order="C")
        shape = ",".join(str(int(size)) for size in contiguous.shape)

        _update_length_prefixed(digest, str(key).encode("utf-8"))
        _update_length_prefixed(digest, str(contiguous.dtype).encode("ascii"))
        _update_length_prefixed(digest, shape.encode("ascii"))
        _update_length_prefixed(digest, raw_bytes)
    return digest.hexdigest()


def _aggregate_hash(per_view):
    digest = hashlib.sha256()
    for view_idx, view_hash in enumerate(per_view):
        _update_length_prefixed(digest, str(view_idx).encode("ascii"))
        _update_length_prefixed(digest, view_hash.encode("ascii"))
    return digest.hexdigest()


def hash_backbone(autoencoders):
    """Hash only Native MVCAN autoencoder state, separately for each view."""
    per_view = [hash_state_dict(module.state_dict()) for module in autoencoders]
    return {
        "per_view": per_view,
        "aggregate": _aggregate_hash(per_view),
    }


def hash_semantic_heads(semantic_heads):
    """Hash semantic head state independently from the Native backbone."""
    if semantic_heads is None:
        return None
    per_view = [
        hash_state_dict(head.state_dict()) for head in semantic_heads.heads
    ]
    return {
        "per_view": per_view,
        "aggregate": _aggregate_hash(per_view),
    }
