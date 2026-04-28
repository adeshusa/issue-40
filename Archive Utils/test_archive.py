"""
Test script for VizFold Archive Utilities using real OpenFold output data.
"""
import sys
import os
import pickle
import shutil
import importlib.util

import numpy as np

# Load the outline module (which doesn't have .py extension)
outline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outline")
spec = importlib.util.spec_from_loader("outline", loader=None, origin=outline_path)
outline = importlib.util.module_from_spec(spec)
with open(outline_path, "r") as f:
    code = f.read()
exec(compile(code, outline_path, "exec"), outline.__dict__)

# Import functions from the loaded module
tensor_to_numpy = outline.tensor_to_numpy
store_single_representation = outline.store_single_representation
store_pair_representation = outline.store_pair_representation
store_attention = outline.store_attention
store_structure_coordinates = outline.store_structure_coordinates
load_single_representation = outline.load_single_representation
load_pair_representation = outline.load_pair_representation
load_attention_head = outline.load_attention_head
validate_archive = outline.validate_archive

# Path to a test pickle file
PICKLE_FILE = "../1UBQ_result_model_1_ptm.pickle copy"
TEST_ARCHIVE = "test_1UBQ.zarr"


def load_pickle_data(pkl_path):
    """Load data from OpenFold output pickle."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def test_store_and_load_single_representation(data, archive_path):
    """Test storing and loading single representation."""
    print("\n=== Testing single representation ===")
    
    single = data["representations"]["single"]
    print(f"Original shape: {single.shape}, dtype: {single.dtype}")
    
    # Store it (treating as layer 0)
    store_single_representation(archive_path, layer_index=0, single_array=single, overwrite=True)
    print("Stored to archive")
    
    # Load it back
    loaded = load_single_representation(archive_path, layer_index=0)
    print(f"Loaded shape: {loaded.shape}, dtype: {loaded.dtype}")
    
    # Verify
    if np.allclose(single, loaded, rtol=1e-3):
        print("✓ Single representation: PASS")
        return True
    else:
        print("✗ Single representation: FAIL - data mismatch")
        return False


def test_store_and_load_pair_representation(data, archive_path):
    """Test storing and loading pair representation."""
    print("\n=== Testing pair representation ===")
    
    pair = data["representations"]["pair"]
    print(f"Original shape: {pair.shape}, dtype: {pair.dtype}")
    
    # Store it (treating as layer 0)
    store_pair_representation(archive_path, layer_index=0, pair_array=pair, overwrite=True)
    print("Stored to archive")
    
    # Load it back
    loaded = load_pair_representation(archive_path, layer_index=0)
    print(f"Loaded shape: {loaded.shape}, dtype: {loaded.dtype}")
    
    # Verify
    if np.allclose(pair, loaded, rtol=1e-3):
        print("✓ Pair representation: PASS")
        return True
    else:
        print("✗ Pair representation: FAIL - data mismatch")
        return False


def test_store_structure(data, archive_path):
    """Test storing structure data."""
    print("\n=== Testing structure storage ===")
    
    sm = data["structure_module"]
    atom_positions = sm["final_atom_positions"]
    atom_mask = sm["final_atom_mask"]
    ptm = float(data["ptm"])
    
    print(f"atom_positions shape: {atom_positions.shape}")
    print(f"atom_mask shape: {atom_mask.shape}")
    print(f"ptm: {ptm}")
    
    # Store structure
    store_structure_coordinates(archive_path, atom_positions, atom_mask=atom_mask, ptm=ptm, overwrite=True)
    print("Stored to archive")
    
    # Verify by opening archive
    import zarr
    root = zarr.open(archive_path, mode="r")
    
    if "structure" in root:
        struct = root["structure"]
        if "atom_positions" in struct:
            loaded_pos = np.asarray(struct["atom_positions"])
            print(f"Loaded atom_positions shape: {loaded_pos.shape}")
            if np.allclose(atom_positions, loaded_pos, rtol=1e-3):
                print("✓ Structure storage: PASS")
                return True
    
    print("✗ Structure storage: FAIL")
    return False


def test_store_attention(data, archive_path):
    """Test storing attention data (synthetic since not in pickle)."""
    print("\n=== Testing attention storage ===")
    
    # Create synthetic attention data based on sequence length
    num_residues = data["representations"]["single"].shape[0]
    num_heads = 8
    
    # Shape: (num_heads, num_residues, num_residues)
    fake_attention = np.random.rand(num_heads, num_residues, num_residues).astype(np.float32)
    print(f"Synthetic attention shape: {fake_attention.shape}")
    
    # Store it
    store_attention(archive_path, attention_type="triangle_start", layer_index=0, 
                   attention_array=fake_attention, overwrite=True)
    print("Stored to archive")
    
    # Load single head
    loaded_head = load_attention_head(archive_path, attention_type="triangle_start", 
                                      layer_index=0, head_index=3)
    print(f"Loaded head shape: {loaded_head.shape}")
    
    # Verify
    if np.allclose(fake_attention[3], loaded_head, rtol=1e-5):
        print("✓ Attention storage: PASS")
        return True
    else:
        print("✗ Attention storage: FAIL")
        return False


def test_validate_archive(archive_path):
    """Test archive validation."""
    print("\n=== Testing archive validation ===")
    
    report = validate_archive(archive_path, strict=False)
    print(f"Valid: {report['valid']}")
    print(f"Components found: {report['components_found']}")
    
    if report["errors"]:
        print(f"Errors: {report['errors']}")
    if report["warnings"]:
        print(f"Warnings: {report['warnings']}")
    
    if report["valid"]:
        print("✓ Archive validation: PASS")
        return True
    else:
        print("✗ Archive validation: FAIL")
        return False


def main():
    print("=" * 60)
    print("VizFold Archive Utilities - Test with Real Data")
    print("=" * 60)
    
    # Check pickle file exists
    pkl_path = os.path.join(os.path.dirname(__file__), PICKLE_FILE)
    if not os.path.exists(pkl_path):
        print(f"ERROR: Pickle file not found: {pkl_path}")
        return 1
    
    print(f"\nLoading data from: {pkl_path}")
    data = load_pickle_data(pkl_path)
    
    archive_path = os.path.join(os.path.dirname(__file__), TEST_ARCHIVE)
    print(f"Test archive: {archive_path}")
    
    # Clean up old test archive
    if os.path.exists(archive_path):
        shutil.rmtree(archive_path)
    
    # Run tests
    results = []
    results.append(test_store_and_load_single_representation(data, archive_path))
    results.append(test_store_and_load_pair_representation(data, archive_path))
    results.append(test_store_structure(data, archive_path))
    results.append(test_store_attention(data, archive_path))
    results.append(test_validate_archive(archive_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
