"""Canonical diagnostic-only hashing for P0-A4 stagewise state snapshots."""

import hashlib
import json
import random
import struct

import numpy as np
import torch


def _framed_sha256(components):
    digest = hashlib.sha256()
    for component in components:
        value = bytes(component)
        digest.update(struct.pack(">Q", len(value)))
        digest.update(value)
    return digest.hexdigest()


def array_logical_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    return _framed_sha256((
        str(array.dtype).encode("ascii"),
        ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ))


def named_hash(records):
    components = []
    for name, value in records:
        components.extend((str(name).encode("utf-8"), str(value).encode("ascii")))
    return _framed_sha256(components)


def _tensor_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def module_state_hash(module, prefix=""):
    records = []
    for name, value in module.state_dict().items():
        records.append((prefix + name, array_logical_sha256(_tensor_array(value))))
    return named_hash(records), dict(records)


def model_state_snapshot(model):
    per_view = []
    aggregate_records = []
    for view_id, autoencoder in enumerate(model.autoencoders):
        view_hash, view_records = module_state_hash(autoencoder)
        encoder_hash, _ = module_state_hash(autoencoder._encoder)
        decoder_hash, _ = module_state_hash(autoencoder._decoder)
        cluster_hash = array_logical_sha256(_tensor_array(autoencoder._cluster_layer))
        per_view.append({
            "view_id": view_id,
            "model_hash": view_hash,
            "encoder_parameter_hash": encoder_hash,
            "decoder_parameter_hash": decoder_hash,
            "cluster_centers_hash": cluster_hash,
            "parameter_hashes": view_records,
            "dtype": str(next(autoencoder.parameters()).dtype),
            "device": str(next(autoencoder.parameters()).device),
        })
        aggregate_records.append(("view_%d" % view_id, view_hash))
    return {
        "aggregate_hash": named_hash(aggregate_records),
        "per_view": per_view,
    }


def parameter_name_map(model):
    return {
        id(parameter): "view_%d.%s" % (view_id, name)
        for view_id, autoencoder in enumerate(model.autoencoders)
        for name, parameter in autoencoder.named_parameters()
    }


def _json_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _framed_sha256((payload,))


def optimizer_state_snapshot(optimizers, model):
    names = parameter_name_map(model)
    per_optimizer = []
    aggregate = []
    for view_id, optimizer in enumerate(optimizers):
        states = []
        for group_id, group in enumerate(optimizer.param_groups):
            hyperparameters = {
                key: value for key, value in group.items() if key != "params"
            }
            normalized = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in hyperparameters.items()
            }
            for parameter in group["params"]:
                parameter_name = names[id(parameter)]
                state = optimizer.state.get(parameter, {})
                tensor_fields = {}
                scalar_fields = {}
                for key, value in sorted(state.items()):
                    if isinstance(value, torch.Tensor):
                        tensor_fields[key] = array_logical_sha256(_tensor_array(value))
                        if value.numel() == 1:
                            scalar_fields[key] = float(value.detach().cpu().item())
                    elif isinstance(value, (int, float, bool)):
                        scalar_fields[key] = value
                    else:
                        scalar_fields[key] = repr(value)
                states.append({
                    "group_id": group_id,
                    "parameter": parameter_name,
                    "hyperparameters": normalized,
                    "state_tensor_hashes": tensor_fields,
                    "state_scalars": scalar_fields,
                })
        optimizer_hash = _json_hash(states)
        per_optimizer.append({
            "view_id": view_id,
            "logical_hash": optimizer_hash,
            "parameters": states,
        })
        aggregate.append(("view_%d" % view_id, optimizer_hash))
    return {
        "aggregate_hash": named_hash(aggregate),
        "per_view": per_optimizer,
    }


def generator_state_hash(generator):
    if generator is None:
        return None
    return array_logical_sha256(_tensor_array(generator.get_state()))


def rng_state_snapshot(generator=None):
    numpy_state = np.random.get_state()
    numpy_hash = named_hash((
        ("algorithm", _framed_sha256((numpy_state[0].encode("ascii"),))),
        ("keys", array_logical_sha256(numpy_state[1])),
        ("position", _framed_sha256((str(numpy_state[2]).encode("ascii"),))),
        ("has_gauss", _framed_sha256((str(numpy_state[3]).encode("ascii"),))),
        ("cached_gaussian", _framed_sha256((repr(numpy_state[4]).encode("ascii"),))),
    ))
    cuda_hashes = []
    if torch.cuda.is_available():
        cuda_hashes = [
            array_logical_sha256(_tensor_array(value))
            for value in torch.cuda.get_rng_state_all()
        ]
    return {
        "python_random_hash": _json_hash(repr(random.getstate())),
        "numpy_rng_hash": numpy_hash,
        "torch_cpu_rng_hash": array_logical_sha256(_tensor_array(torch.get_rng_state())),
        "torch_cuda_rng_hash": named_hash([
            ("device_%d" % index, value) for index, value in enumerate(cuda_hashes)
        ]) if cuda_hashes else None,
        "dataloader_generator_state_hash": generator_state_hash(generator),
    }


@torch.no_grad()
def representation_snapshot(model, full_views):
    latent_hashes = []
    q_hashes = []
    statistics = {"latent": [], "q_local": []}
    for view_id, (autoencoder, values) in enumerate(
        zip(model.autoencoders, full_views)
    ):
        latent = autoencoder.encoder(values)
        q_local = autoencoder.clustering(latent)
        latent_array = _tensor_array(latent)
        q_array = _tensor_array(q_local)
        latent_hashes.append(array_logical_sha256(latent_array))
        q_hashes.append(array_logical_sha256(q_array))
        statistics["latent"].append(array_statistics(latent_array, view_id))
        statistics["q_local"].append(array_statistics(q_array, view_id))
    return {
        "latent_per_view_hash": latent_hashes,
        "q_local_per_view_hash": q_hashes,
        "latent_aggregate_hash": named_hash([
            ("view_%d" % index, value) for index, value in enumerate(latent_hashes)
        ]),
        "q_local_aggregate_hash": named_hash([
            ("view_%d" % index, value) for index, value in enumerate(q_hashes)
        ]),
        "_statistics": statistics,
    }


def gradient_snapshot(model):
    per_view = []
    for view_id, autoencoder in enumerate(model.autoencoders):
        records = []
        for name, parameter in autoencoder.named_parameters():
            digest = None if parameter.grad is None else array_logical_sha256(
                _tensor_array(parameter.grad)
            )
            records.append((name, digest or "NONE"))
        per_view.append({
            "view_id": view_id,
            "aggregate_hash": named_hash(records),
            "parameter_gradient_hashes": dict(records),
        })
    return per_view


def array_statistics(value, identifier=None):
    array = _tensor_array(value).astype(np.float64, copy=False)
    return {
        "id": identifier,
        "shape": list(array.shape),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "std": float(array.std()) if array.size else None,
        "l2": float(np.linalg.norm(array.ravel())) if array.size else 0.0,
    }


def stage_snapshot(stage, model, optimizers, generator, full_views, **extra):
    representation = representation_snapshot(model, full_views)
    statistics = representation.pop("_statistics")
    result = {
        "stage": stage,
        "model": model_state_snapshot(model),
        "optimizer": optimizer_state_snapshot(optimizers, model),
        "rng": rng_state_snapshot(generator),
        "representation": representation,
    }
    result.update(extra)
    result["_statistics"] = statistics
    return result

