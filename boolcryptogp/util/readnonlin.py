import os
import pandas as pd
import numpy as np

# Folder containing the CSV files
folder_path = "results_1"

# Initialize a dictionary to store the results
results = {}

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("best-pseudobooleanfunctionsGPNSGA2"):  # Process only CSV files
        # Parse parameters from the filename
        try:
            num_bits = int(file_name.split('-')[2].replace("bit", ""))
            dataset_type = next(filter(lambda x: x.startswith("datasettype_"), file_name.split('-'))).replace("datasettype_", "")
        except Exception as e:
            print(f"Error parsing parameters from {file_name}: {e}")
            continue

        # Read the CSV file
        file_path = os.path.join(folder_path, file_name)
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Ensure 'NonLinearity' column exists
        if 'NonLinearity' not in data.columns:
            print(f"'NonLinearity' column missing in {file_path}. Skipping this file.")
            continue

        # Extract the Seed value from the filename
        try:
            seed = int(next(filter(lambda x: x.startswith("SEED"), file_name.split('-'))).replace("SEED", "").replace(".csv", ""))
        except Exception as e:
            print(f"Error parsing SEED from {file_name}: {e}")
            continue

        # Compute the maximum absolute value of the NonLinearity column
        max_abs_nonlinearity = data['NonLinearity'].abs().max()

        # Store the result in the dictionary
        key = (num_bits, dataset_type)
        if key not in results:
            results[key] = {}
        results[key][seed] = max_abs_nonlinearity

# Compute the median for each dataset type and number of bits
final_results = {}
for (num_bits, dataset_type), seeds_dict in results.items():
    if len(seeds_dict) != 30:
        print(f"Warning: Found {len(seeds_dict)} seeds for {num_bits}-bit {dataset_type}. Expected 30.")
    medians = np.median(list(seeds_dict.values()))
    final_results[(num_bits, dataset_type)] = medians

# Print the results
print("\nMedian of highest absolute NonLinearity values for each dataset type and number of bits:")
for (num_bits, dataset_type), median_value in sorted(final_results.items()):
    print(f"{num_bits}-bit {dataset_type}: {median_value}")

