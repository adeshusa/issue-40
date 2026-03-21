"""
Storage helpers for VizFold archive outputs.

This module currently defines scaffolding for:
- Method 3: layer activation storage
- Method 4: pair representation storage
"""

import numpy as np
import zarr

STORAGE_DTYPE = np.float32


def _normalize_embedding_array(array, expected_ndim, array_name):
    """
    Validate an embedding-like array and normalize it for storage.
    """
    array = np.asarray(array)

    if array.ndim != expected_ndim:
        raise ValueError(
            f"{array_name} must have {expected_ndim} dimensions; "
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

    return array.astype(STORAGE_DTYPE, copy=False)


# ============================================================
# METHOD 3
# ============================================================

def store_layer_activation(path, layer_index, activation_array):
    """
    Store a transformer layer activation inside the archive.

    Layer activations represent the hidden state outputs from a
    transformer block.

    Typical shape:
        (tokens, hidden_dimension)

    Archive layout:
        layers/{layer_index}/activation

    Storage behavior:
        Each call appends one activation matrix to the dataset,
        so stored arrays have shape:
        (num_writes, tokens, hidden_dimension)

    Responsibilities:
    -----------------
    - Ensure the correct archive group exists
    - Validate activation shape
    - Write the activation array to the appropriate location

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    layer_index : int
        Index of the transformer layer.

    activation_array : numpy.ndarray
        Activation tensor with shape (tokens, hidden_dim).

    Returns
    -------
    None
    """

    if not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer")
    if layer_index < 0:
        raise ValueError("layer_index must be nonnegative")

    activation_array = _normalize_embedding_array(
        activation_array,
        expected_ndim=2,
        array_name="activation_array",
    )
    tokens, hidden_dim = activation_array.shape

    root = zarr.open_group(path, mode="a")
    layers_group = root.require_group("layers")
    layer_group = layers_group.require_group(str(layer_index))
    layer_group.attrs["layer_index"] = layer_index

    # Append each write as a new leading entry.
    if "activation" in layer_group:
        activation_ds = layer_group["activation"]

        if activation_ds.ndim != 3:
            raise ValueError(
                "Existing activation dataset must be 3D with shape "
                "(num_writes, tokens, hidden_dim)"
            )
        if activation_ds.shape[1] != tokens:
            raise ValueError(
                "Token dimension mismatch: "
                f"existing={activation_ds.shape[1]}, new={tokens}"
            )
        if activation_ds.shape[2] != hidden_dim:
            raise ValueError(
                "Hidden dimension mismatch: "
                f"existing={activation_ds.shape[2]}, new={hidden_dim}"
            )

        old_writes = activation_ds.shape[0]
        activation_ds.resize((old_writes + 1, tokens, hidden_dim))
        activation_ds[old_writes, :, :] = activation_array
    else:
        token_chunk = max(1, min(tokens, 1024))
        layer_group.create_dataset(
            "activation",
            data=activation_array[np.newaxis, :, :],
            shape=(1, tokens, hidden_dim),
            chunks=(1, token_chunk, hidden_dim),
            dtype=STORAGE_DTYPE,
            overwrite=False,
        )

    activation_ds = layer_group["activation"]
    activation_ds.attrs["write_policy"] = "append"
    activation_ds.attrs["storage_dtype"] = np.dtype(STORAGE_DTYPE).name
    activation_ds.attrs["tokens"] = tokens
    activation_ds.attrs["hidden_dim"] = hidden_dim
    activation_ds.attrs["num_writes"] = activation_ds.shape[0]


# ============================================================
# METHOD 4
# ============================================================

def store_pair_representation(path, pair_array):
    """
    Store pair representation embeddings.

    Pair representations capture relationships between residues
    or tokens in the model and are commonly used in protein
    structure prediction models like OpenFold.

    Typical shape:
        (tokens, tokens, pair_dimension)

    Archive layout:
        representations/pair

    Storage behavior:
        Each call appends one pair tensor to the dataset,
        so stored arrays have shape:
        (num_writes, tokens, tokens, pair_dimension)

    Responsibilities:
    -----------------
    - Validate input shape
    - Create representations group if needed
    - Store pair representation array

    Parameters
    ----------
    path : str
        Root path to the Zarr archive.

    pair_array : numpy.ndarray
        Pair representation tensor.

    Returns
    -------
    None
    """
    pair_array = _normalize_embedding_array(
        pair_array,
        expected_ndim=3,
        array_name="pair_array",
    )

    tokens_i, tokens_j, pair_dim = pair_array.shape
    if tokens_i != tokens_j:
        raise ValueError(
            "pair_array first two dimensions must match (tokens x tokens)"
        )

    root = zarr.open_group(path, mode="a")
    representations_group = root.require_group("representations")

    if "pair" in representations_group:
        pair_ds = representations_group["pair"]
        if pair_ds.ndim != 4:
            raise ValueError(
                "Existing pair dataset must be 4D with shape "
                "(num_writes, tokens, tokens, pair_dim)"
            )
        if pair_ds.shape[1] != tokens_i or pair_ds.shape[2] != tokens_j:
            raise ValueError(
                "Token dimension mismatch: "
                f"existing=({pair_ds.shape[1]}, {pair_ds.shape[2]}), "
                f"new=({tokens_i}, {tokens_j})"
            )
        if pair_ds.shape[3] != pair_dim:
            raise ValueError(
                "Pair dimension mismatch: "
                f"existing={pair_ds.shape[3]}, new={pair_dim}"
            )

        old_writes = pair_ds.shape[0]
        pair_ds.resize((old_writes + 1, tokens_i, tokens_j, pair_dim))
        pair_ds[old_writes, :, :, :] = pair_array
    else:
        token_chunk = max(1, min(tokens_i, 128))
        representations_group.create_dataset(
            "pair",
            data=pair_array[np.newaxis, :, :, :],
            shape=(1, tokens_i, tokens_j, pair_dim),
            chunks=(1, token_chunk, token_chunk, pair_dim),
            dtype=STORAGE_DTYPE,
            overwrite=False,
        )

    pair_ds = representations_group["pair"]
    pair_ds.attrs["write_policy"] = "append"
    pair_ds.attrs["storage_dtype"] = np.dtype(STORAGE_DTYPE).name
    pair_ds.attrs["tokens"] = tokens_i
    pair_ds.attrs["pair_dim"] = pair_dim
    pair_ds.attrs["num_writes"] = pair_ds.shape[0]
