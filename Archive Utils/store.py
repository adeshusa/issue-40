"""
Storage helpers for VizFold inference trace archives.

This module currently defines:
- Method 3: single-representation layer storage
- Method 4: pair-representation layer storage

The storage layout follows the VizFold Zarr format specification:

run.vizfold.zarr/
├── metadata/
├── representations/
│   ├── single/
│   │   ├── layer_00
│   │   └── ...
│   └── pair/
│       ├── layer_00
│       └── ...
├── attention/
│   └── triangle_start/
└── structure/
"""

import numpy as np
import zarr

STORAGE_DTYPE = np.float32
_SINGLE_TOKEN_CHUNK = 1024
_PAIR_TOKEN_CHUNK = 128


def _open_archive_root(path):
    """
    Open (or create) a Zarr v2 archive root.

    Falls back to plain open_group for older zarr versions that do not
    accept the zarr_version keyword.
    """
    try:
        return zarr.open_group(path, mode="a", zarr_version=2)
    except TypeError:
        return zarr.open_group(path, mode="a")


def _ensure_archive_layout(root):
    """
    Ensure canonical top-level groups and required subgroups exist.
    """
    root.require_group("metadata")
    representations_group = root.require_group("representations")
    representations_group.require_group("single")
    representations_group.require_group("pair")
    attention_group = root.require_group("attention")
    attention_group.require_group("triangle_start")
    root.require_group("structure")


def _normalize_numeric_array(array, *, array_name, min_ndim):
    """
    Validate and normalize an array for storage.
    """
    array = np.asarray(array)

    if array.ndim < min_ndim:
        raise ValueError(
            f"{array_name} must have at least {min_ndim} dimensions; "
            f"received shape {array.shape}"
        )

    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(
            f"{array_name} must contain numeric values; "
            f"received dtype {array.dtype}"
        )

    if any(dim <= 0 for dim in array.shape):
        raise ValueError(
            f"{array_name} dimensions must all be positive; "
            f"received shape {array.shape}"
        )

    return array.astype(STORAGE_DTYPE, copy=False)


def _layer_name(layer_index):
    """
    Build canonical layer key names such as layer_00, layer_01, ... .
    """
    if not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer")
    if layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    return f"layer_{layer_index:02d}"


def _single_chunks(shape):
    """
    Recommended chunking for per-residue (single) representations.
    """
    return (max(1, min(shape[0], _SINGLE_TOKEN_CHUNK)),) + shape[1:]


def _pair_chunks(shape):
    """
    Recommended chunking for pair representations.
    """
    return (
        max(1, min(shape[0], _PAIR_TOKEN_CHUNK)),
        max(1, min(shape[1], _PAIR_TOKEN_CHUNK)),
    ) + shape[2:]


# ============================================================
# METHOD 3
# ============================================================

def store_layer_activation(path, layer_index, activation_array):
    """
    Store a per-layer single representation tensor.

    Typical shape:
        (num_residues, channel_dim)
    but any N-D numeric array with ndim >= 2 is accepted.

    Archive layout:
        representations/single/layer_XX

    Storage behavior:
        Each layer is stored as one named dataset. Writing to the same
        layer index overwrites that layer dataset.

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    layer_index : int
        Layer index used to build the key layer_XX.

    activation_array : numpy.ndarray
        Single-representation tensor for one layer.

    Returns
    -------
    None
    """
    layer_key = _layer_name(layer_index)
    activation_array = _normalize_numeric_array(
        activation_array,
        array_name="activation_array",
        min_ndim=2,
    )

    root = _open_archive_root(path)
    _ensure_archive_layout(root)

    single_group = root["representations"]["single"]
    single_group.create_dataset(
        layer_key,
        data=activation_array,
        shape=activation_array.shape,
        chunks=_single_chunks(activation_array.shape),
        dtype=STORAGE_DTYPE,
        overwrite=True,
    )

    ds = single_group[layer_key]
    ds.attrs["layer_index"] = layer_index
    ds.attrs["representation_type"] = "single"
    ds.attrs["storage_dtype"] = np.dtype(STORAGE_DTYPE).name


# ============================================================
# METHOD 4
# ============================================================

def store_pair_representation(path, pair_array, layer_index=0):
    """
    Store a per-layer pair representation tensor.

    Typical shape:
        (num_residues, num_residues, pair_dim)
    but any N-D numeric array with ndim >= 2 is accepted.

    Archive layout:
        representations/pair/layer_XX

    Storage behavior:
        Each layer is stored as one named dataset. Writing to the same
        layer index overwrites that layer dataset.

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    pair_array : numpy.ndarray
        Pair representation tensor for one layer.

    layer_index : int, optional
        Layer index used to build the key layer_XX.
        Defaults to 0 for backward compatibility with older calls.

    Returns
    -------
    None
    """
    layer_key = _layer_name(layer_index)
    pair_array = _normalize_numeric_array(
        pair_array,
        array_name="pair_array",
        min_ndim=2,
    )

    if pair_array.shape[0] != pair_array.shape[1]:
        raise ValueError(
            "pair_array first two dimensions must match (num_residues x num_residues); "
            f"received shape {pair_array.shape}"
        )

    root = _open_archive_root(path)
    _ensure_archive_layout(root)

    pair_group = root["representations"]["pair"]
    pair_group.create_dataset(
        layer_key,
        data=pair_array,
        shape=pair_array.shape,
        chunks=_pair_chunks(pair_array.shape),
        dtype=STORAGE_DTYPE,
        overwrite=True,
    )

    ds = pair_group[layer_key]
    ds.attrs["layer_index"] = layer_index
    ds.attrs["representation_type"] = "pair"
    ds.attrs["storage_dtype"] = np.dtype(STORAGE_DTYPE).name
