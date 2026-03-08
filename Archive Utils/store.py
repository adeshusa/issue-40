"""
Storage helpers for VizFold archive outputs.

This module currently defines scaffolding for:
- Method 3: layer activation storage
- Method 4: pair representation storage
"""

import numpy as np
import zarr


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
    # Basic input checks for this method's scope only.
    if not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer")

    activation_array = np.asarray(activation_array)
    if activation_array.ndim != 2:
        raise ValueError(
            "activation_array must have shape (tokens, hidden_dim)"
        )

    tokens, hidden_dim = activation_array.shape

    # Ensure archive structure: layers/{layer_index}/
    root = zarr.open_group(path, mode="a")
    layers_group = root.require_group("layers")
    layer_group = layers_group.require_group(str(layer_index))

    # Incremental behavior:
    # - If dataset exists, append on token axis.
    # - If missing, create it with initial data.
    if "activation" in layer_group:
        activation_ds = layer_group["activation"]

        if activation_ds.ndim != 2:
            raise ValueError(
                "Existing activation dataset must be 2D"
            )
        if activation_ds.shape[1] != hidden_dim:
            raise ValueError(
                "Hidden dimension mismatch: "
                f"existing={activation_ds.shape[1]}, new={hidden_dim}"
            )

        old_tokens = activation_ds.shape[0]
        activation_ds.resize((old_tokens + tokens, hidden_dim))
        activation_ds[old_tokens:old_tokens + tokens, :] = activation_array
    else:
        # Chunking is token-major so later appends and reads stay simple.
        token_chunk = max(1, min(tokens, 1024))
        layer_group.create_dataset(
            "activation",
            data=activation_array,
            shape=(tokens, hidden_dim),
            chunks=(token_chunk, hidden_dim),
            dtype=activation_array.dtype,
            overwrite=False,
        )


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
    pair_array = np.asarray(pair_array)
    if pair_array.ndim != 3:
        raise ValueError(
            "pair_array must have shape (tokens, tokens, pair_dim)"
        )

    tokens_i, tokens_j, pair_dim = pair_array.shape
    if tokens_i != tokens_j:
        raise ValueError(
            "pair_array first two dimensions must match (tokens x tokens)"
        )

    root = zarr.open_group(path, mode="a")
    representations_group = root.require_group("representations")

    # If pair already exists with matching shape, update in place.
    # Otherwise, replace it with the new tensor.
    if "pair" in representations_group:
        pair_ds = representations_group["pair"]
        if pair_ds.shape == pair_array.shape:
            pair_ds[:] = pair_array
            return

    token_chunk = max(1, min(tokens_i, 128))
    representations_group.create_dataset(
        "pair",
        data=pair_array,
        shape=(tokens_i, tokens_j, pair_dim),
        chunks=(token_chunk, token_chunk, pair_dim),
        dtype=pair_array.dtype,
        overwrite=True,
    )
