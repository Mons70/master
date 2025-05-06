import os
import json
import numpy as np
from sklearn import preprocessing
# Directory containing the .json files
directory = '/home/mons/dev/private/master/saved_trajectories/normalized_dataset'

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        # Open and load the JSON file
        with open(file_path, 'r') as file:
            print(file_path)
            data = json.load(file)
        
        # Check if "states" exists and is a listgit 
        if "states" in data and isinstance(data["states"], list):
            arr = np.round(preprocessing.minmax_scale(data["states"],feature_range=(-1,1),axis=0),2) # Normalize all the state to be between -1 and 1
            data["states"] = arr.tolist()
            # Save the updated JSON back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file)