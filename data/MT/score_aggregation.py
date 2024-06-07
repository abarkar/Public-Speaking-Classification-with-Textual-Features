import pandas as pd
import numpy as np

# Load the CSV file
file_path = './MT_Labels.csv'  # Update this with the actual path
data = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')

# Define aggregation functions
def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def harmmean(x):
    return len(x) / np.sum(1.0 / x)

# Initialize an empty DataFrame to store the results
results = []

# Create dictionary for clip names mapping
clip_map = {'full':'full', 'beginning':'beg', 'middle':'mid', 'end':'end'}

# Iterate over each unique ID and clip
for name in data['Input.name'].unique():
    for clip in data['clip'].unique():
        subset = data[(data['Input.name'] == name) & (data['clip'] == clip)]
        for method, func in zip(['rms', 'mean', 'harmmean'], [rms, np.mean, harmmean]):
            result = {
                'ID': name,
                'clip': clip_map[clip],
                'aggregationMethod': method,
                'SelfConfidence': func(subset['Answer.Competence']),
                'Persuasiveness': func(subset['Answer.Persuasiveness']),
                'Engagement': func(subset['Answer.Engagement']),
                'Global': func(subset['Answer.Global']),
            }
            results.append(result)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_file_path = './MT_aggregated_ratings.csv'  # Update this with the desired output path
results_df.to_csv(output_file_path, index=False)
