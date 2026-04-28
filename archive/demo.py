"""
Small demo for reading and incrementally updating a VizFold Zarr archive.

Run from repo root:
    python archive/demo.py

Run from the archive directory:
    python demo.py
"""

import sys
from pathlib import Path

import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive import (  # noqa: E402
    load_attention_head,
    load_pair_representation,
    load_single_representation,
    store_attention,
    store_pair_representation,
    store_single_representation,
)


ARCHIVE_PATH = Path(__file__).resolve().parent / "test_1UBQ.zarr"
ARCHIVE = str(ARCHIVE_PATH)
ATTENTION_TYPE = "triangle_start"


if not ARCHIVE_PATH.exists():
    raise FileNotFoundError(
        f"Demo archive not found: {ARCHIVE_PATH}\n"
        "Create it first with: python archive/test_archive.py"
    )

root = zarr.open(ARCHIVE_PATH, mode="r")
single_group = root["representations"]["single"]
pair_group = root["representations"]["pair"]
attention_group = root["attention"][ATTENTION_TYPE]

existing_layers = sorted(
    int(name.removeprefix("layer_"))
    for name in single_group.keys()
    if name.startswith("layer_")
)
source_layer = existing_layers[-1]
new_layer = source_layer + 1

print(f"Using archive: {ARCHIVE_PATH}")
print("Before add:")
print(f"  single layers: {sorted(single_group.keys())}")
print(f"  pair layers: {sorted(pair_group.keys())}")
print(f"  {ATTENTION_TYPE} layers: {sorted(attention_group.keys())}")

# Read existing data through the archive utilities.
single = load_single_representation(ARCHIVE, source_layer)
pair = load_pair_representation(ARCHIVE, source_layer)
attention = np.asarray(attention_group[f"layer_{source_layer:02d}"])
attention_head = load_attention_head(ARCHIVE, ATTENTION_TYPE, source_layer, 0)

print(f"\nRead source layer_{source_layer:02d}:")
print(f"  single shape: {single.shape}")
print(f"  pair shape: {pair.shape}")
print(f"  attention shape: {attention.shape}")
print(f"  attention head 0 shape: {attention_head.shape}")

# Add one incremental layer. The offset makes it easy to verify this is new data.
offset = np.float32(new_layer * 0.01)
store_single_representation(ARCHIVE, new_layer, single + offset)
store_pair_representation(ARCHIVE, new_layer, pair + offset)
store_attention(ARCHIVE, ATTENTION_TYPE, new_layer, attention + offset)

# Re-open and read the new data back.
root = zarr.open(ARCHIVE_PATH, mode="r")
new_single = load_single_representation(ARCHIVE, new_layer)
new_pair = load_pair_representation(ARCHIVE, new_layer)
new_attention_head = load_attention_head(ARCHIVE, ATTENTION_TYPE, new_layer, 0)

print(f"\nAdded and read back layer_{new_layer:02d}:")
print(f"  single shape: {new_single.shape}, mean delta: {float((new_single - single).mean()):.6f}")
print(f"  pair shape: {new_pair.shape}, mean delta: {float((new_pair - pair).mean()):.6f}")
print(
    "  attention head 0 shape: "
    f"{new_attention_head.shape}, mean delta: {float((new_attention_head - attention_head).mean()):.6f}"
)

print("\nAfter add:")
print(f"  single layers: {sorted(root['representations']['single'].keys())}")
print(f"  pair layers: {sorted(root['representations']['pair'].keys())}")
print(f"  {ATTENTION_TYPE} layers: {sorted(root['attention'][ATTENTION_TYPE].keys())}")
