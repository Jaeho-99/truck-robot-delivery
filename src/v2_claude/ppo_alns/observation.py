"""CPU transport for the graph-free PPO observation (num_nodes=0, g_t)."""

import numpy as np
import torch
from torch_geometric.data import Data


OBSERVATION_CODECS = ("direct", "numpy")
_G_DIM = 9


def _check_codec(codec):
    if codec not in OBSERVATION_CODECS:
        raise ValueError(f"unknown observation codec: {codec!r}")


def validate_observation(data):
    """Reject unsupported attributes instead of silently dropping information."""
    if not isinstance(data, Data):
        raise TypeError("PPO observation must be a Data object")
    if set(data.to_dict()) != {"num_nodes", "g"}:
        raise ValueError("PPO observation must contain exactly num_nodes and g")
    if type(data.num_nodes) is not int or data.num_nodes != 0:
        raise ValueError("graph-free PPO observation must have num_nodes=0")
    g = data.g
    if not isinstance(g, torch.Tensor):
        raise TypeError("g must be a Torch tensor")
    if (g.device.type != "cpu" or g.dtype != torch.float32
            or tuple(g.shape) != (1, _G_DIM)
            or g.layout != torch.strided or g.requires_grad):
        raise ValueError("g must be a dense CPU float32 (1, 9) observation")
    return data


def encode_observation(data, codec="direct"):
    _check_codec(codec)
    validate_observation(data)
    if codec == "direct":
        return data
    return {"version": 1, "num_nodes": 0,
            "g": data.g.detach().numpy().copy(order="C")}


def decode_observation(payload, codec="direct"):
    _check_codec(codec)
    if codec == "direct":
        return validate_observation(payload)
    if not isinstance(payload, dict) or set(payload) != {"version", "num_nodes", "g"}:
        raise ValueError("invalid NumPy observation payload")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported observation payload version")
    if type(payload["num_nodes"]) is not int or payload["num_nodes"] != 0:
        raise ValueError("NumPy observation must have num_nodes=0")
    array = payload["g"]
    if (not isinstance(array, np.ndarray) or array.dtype != np.float32
            or array.shape != (1, _G_DIM)):
        raise ValueError("NumPy g must have dtype float32 and shape (1, 9)")
    data = Data(num_nodes=0)
    data.g = torch.from_numpy(array.copy(order="C"))
    return validate_observation(data)
