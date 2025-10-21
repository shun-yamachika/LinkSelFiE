#!/usr/bin/env python3
"""
Convert pickle files in outputs/ directory to JSON format.
"""

import pickle
import json
import os
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def convert_pickle_to_json(pickle_file_path, json_file_path):
    """
    Convert a single pickle file to JSON format.

    Args:
        pickle_file_path: Path to the input pickle file
        json_file_path: Path to the output JSON file
    """
    try:
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print(f"✓ Converted: {pickle_file_path} -> {json_file_path}")
        return True
    except Exception as e:
        print(f"✗ Error converting {pickle_file_path}: {e}")
        return False


def main():
    """
    Convert all pickle files in outputs/ directory to JSON format.
    """
    # Define directories
    input_dir = Path('outputs')
    output_dir = Path('outputs')

    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # Find all pickle files
    pickle_files = list(input_dir.glob('*.pickle'))

    if not pickle_files:
        print(f"No pickle files found in '{input_dir}' directory.")
        return

    print(f"Found {len(pickle_files)} pickle file(s) to convert.\n")

    # Convert each pickle file
    success_count = 0
    for pickle_file in pickle_files:
        json_file = output_dir / f"{pickle_file.stem}.json"
        if convert_pickle_to_json(pickle_file, json_file):
            success_count += 1

    print(f"\nConversion complete: {success_count}/{len(pickle_files)} files converted successfully.")


if __name__ == '__main__':
    main()
