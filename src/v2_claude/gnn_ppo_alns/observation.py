"""CPU observation transport with explicit graph-schema preservation.

The default codec leaves HeteroData intact.  The optional NumPy codec never puts
Torch tensors in its payload, allowing resource comparisons without changing the
environment or policy.  Both reject unsupported attributes instead of silently
discarding part of an observation.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

from v2_claude.gnn_ppo_alns.gnn import EDGE_DIMS, EDGE_TYPES, G_DIM, NODE_DIMS


OBSERVATION_CODECS = ("direct", "numpy")
_PAYLOAD_VERSION = 1


def _check_codec(codec):
    if codec not in OBSERVATION_CODECS:
        raise ValueError(f"unknown observation codec {codec!r}; "
                         f"expected one of {OBSERVATION_CODECS}")


def _require_keys(value, keys, label):
    if set(value) != set(keys):
        raise ValueError(f"{label} must have exactly these keys: {tuple(keys)!r}")


def _cpu_tensor(value, dtype, shape, label):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a Torch tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{label} must remain on CPU, got {value.device}")
    if value.layout != torch.strided or value.requires_grad:
        raise ValueError(f"{label} must be a dense observation without gradients")
    if value.dtype != dtype:
        raise ValueError(f"{label} must have dtype {dtype}, got {value.dtype}")
    if value.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape)):
        raise ValueError(f"{label} must have shape {shape}, got {tuple(value.shape)}")


def validate_observation(data):
    """Validate one unbatched graph without mutating its store insertion order.

    GraphBuilder creates present edge stores in route order before padding the
    remaining edge types.  Enforcing EDGE_TYPES *order* here would reject valid
    serial observations; the exact key set is what must match the schema.
    """
    if not isinstance(data, HeteroData):
        raise TypeError("GNN observation must be HeteroData")
    _require_keys(data.node_types, NODE_DIMS, "node types")
    _require_keys(data.edge_types, EDGE_TYPES, "edge types")
    stores = data.to_dict()
    _require_keys(stores, ["_global_store", *NODE_DIMS, *EDGE_TYPES], "graph stores")
    _require_keys(stores["_global_store"], ("g",), "global store")
    _cpu_tensor(data.g, torch.float32, (1, G_DIM), "g")
    for node_type, dimension in NODE_DIMS.items():
        _require_keys(stores[node_type], ("x",), f"node {node_type!r}")
        _cpu_tensor(data[node_type].x, torch.float32, (None, dimension),
                    f"{node_type}.x")
    for edge_type in EDGE_TYPES:
        store = data[edge_type]
        _require_keys(stores[edge_type], ("edge_index", "edge_attr"),
                      f"edge {edge_type!r}")
        _cpu_tensor(store.edge_index, torch.int64, (2, None),
                    f"{edge_type}.edge_index")
        _cpu_tensor(store.edge_attr, torch.float32,
                    (store.edge_index.shape[1], EDGE_DIMS[edge_type[1]]),
                    f"{edge_type}.edge_attr")
    return data


def encode_observation(data, codec="direct"):
    """Encode a CPU graph; NumPy payloads own snapshots of every tensor's data."""
    _check_codec(codec)
    validate_observation(data)
    if codec == "direct":
        # The current builder allocates fresh tensors at every step/reset and
        # workers must not mutate a returned graph after sending it.
        return data

    def snapshot(tensor):
        return tensor.detach().numpy().copy(order="C")

    return {
        "version": _PAYLOAD_VERSION,
        "node_order": tuple(data.node_types),
        "edge_order": tuple(data.edge_types),
        "nodes": {node: snapshot(data[node].x) for node in NODE_DIMS},
        "edges": {
            edge: (snapshot(data[edge].edge_index), snapshot(data[edge].edge_attr))
            for edge in EDGE_TYPES
        },
        "g": snapshot(data.g),
    }


def _require_order(order, expected, label):
    if not isinstance(order, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    if len(order) != len(expected) or set(order) != set(expected):
        raise ValueError(f"{label} must contain each expected store exactly once")


def _numpy_tensor(array, dtype, shape, label):
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype):
        raise TypeError(f"{label} must be a NumPy array with dtype {dtype}")
    if array.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape)):
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    # Own writable storage: neither the caller reusing its payload nor a
    # read-only NumPy array can invalidate or alter an earlier observation.
    return torch.from_numpy(array.copy(order="C"))


def decode_observation(payload, codec="direct"):
    """Restore values and original store order; reject malformed payloads early."""
    _check_codec(codec)
    if codec == "direct":
        return validate_observation(payload)
    if not isinstance(payload, dict):
        raise TypeError("NumPy observation payload must be a dict")
    _require_keys(payload, ("version", "node_order", "edge_order", "nodes",
                            "edges", "g"), "NumPy observation payload")
    if type(payload["version"]) is not int or payload["version"] != _PAYLOAD_VERSION:
        raise ValueError("unsupported NumPy observation payload version")
    if not isinstance(payload["nodes"], dict) or not isinstance(payload["edges"], dict):
        raise TypeError("NumPy nodes and edges must be dictionaries")
    _require_keys(payload["nodes"], NODE_DIMS, "NumPy nodes")
    _require_keys(payload["edges"], EDGE_TYPES, "NumPy edges")
    _require_order(payload["node_order"], NODE_DIMS, "node_order")
    _require_order(payload["edge_order"], EDGE_TYPES, "edge_order")
    data = HeteroData()
    for node in payload["node_order"]:
        data[node].x = _numpy_tensor(payload["nodes"][node], np.float32,
                                     (None, NODE_DIMS[node]), f"{node}.x")
    for edge in payload["edge_order"]:
        arrays = payload["edges"][edge]
        if not isinstance(arrays, (tuple, list)) or len(arrays) != 2:
            raise ValueError(f"edge {edge!r} must contain index and attribute arrays")
        index, attr = arrays
        data[edge].edge_index = _numpy_tensor(index, np.int64, (2, None),
                                             f"{edge}.edge_index")
        data[edge].edge_attr = _numpy_tensor(
            attr, np.float32,
            (data[edge].edge_index.shape[1], EDGE_DIMS[edge[1]]),
            f"{edge}.edge_attr")
    data.g = _numpy_tensor(payload["g"], np.float32, (1, G_DIM), "g")
    return validate_observation(data)
