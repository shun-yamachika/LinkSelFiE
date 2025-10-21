#!/usr/bin/env python3
"""
Convert old JSON format to new format with metadata
"""
import json
import os

def convert_old_to_new_format(input_file, output_file):
    """Convert old JSON format (without metadata) to new format (with metadata)"""

    with open(input_file, 'r') as f:
        old_data = json.load(f)

    # Create new format with metadata
    new_data = {
        'metadata': {
            'noise_model': 'Depolar',  # Extracted from filename
            'repeat': 20,  # Default value based on code
            'bounces': [1, 2, 3, 4],
            'sample_times': {
                '1': 200,
                '2': 200,
                '3': 200,
                '4': 200
            },
            'gap': 0.04
        },
        'data': {}
    }

    # Convert old data format to new format
    for algorithm_name, algo_data in old_data.items():
        if isinstance(algo_data, list) and len(algo_data) == 2:
            # Old format: [path_num_list, costs_per_path_num]
            path_num_list = algo_data[0]
            costs_per_path_num = algo_data[1]

            # Generate fidelity lists for each path_num
            fidelity_lists = []
            for path_num in path_num_list:
                fidelity_list = []
                fidelity = 1.0
                gap = 0.04
                for _ in range(path_num):
                    fidelity_list.append(fidelity)
                    fidelity -= gap
                fidelity_lists.append(fidelity_list)

            # Create new format
            new_data['data'][algorithm_name] = {
                'path_num_list': path_num_list,
                'costs_per_path_num': costs_per_path_num,
                'fidelity_lists': fidelity_lists,
                'raw_results': []  # Empty as old format doesn't have this
            }

    # Write new format
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"Converted {input_file} -> {output_file}")

if __name__ == '__main__':
    input_file = 'outputs1/outputs/plot_cost_vs_path_num_Depolar.json'
    output_file = 'outputs1/outputs/plot_cost_vs_path_num_Depolar_new.json'

    if os.path.exists(input_file):
        convert_old_to_new_format(input_file, output_file)
        print(f"\nNew file created: {output_file}")
        print("You can now replace the old file with the new one if needed.")
    else:
        print(f"Error: {input_file} not found")
