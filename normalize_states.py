import os
import json
import numpy as np
from sklearn import preprocessing
from glob import glob

# Iterate through all files in the directory
def normalize_states(directory, min=-1, max=1):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                print(file_path)
                data = json.load(file)
            
            # Check if "states" exists and is a listgit 
            if "states" in data and isinstance(data["states"], list):
                arr = np.round(preprocessing.minmax_scale(data["states"],feature_range=(min,max),axis=0),2) # Normalize all the state to be between -1 and 1
                data["states"] = arr.tolist()
                # Save the updated JSON back to the file
                with open(file_path, 'w') as file:
                    json.dump(data, file)

def get_state_scaling(directory: str):
    ep_paths = os.path.join(directory, "ep_*.json")
    state_min = 0
    state_max = 0
    for ep_file in sorted(glob(ep_paths)):
        with open(ep_file, 'r') as f:
            data_dictionary = json.load(f)
        # state_values = np.array([ai['states'] for ai in data_dictionary['states']])
        state_values = np.array(state for state in data_dictionary['states'])
        curr_min = np.min(state_values)
        if curr_min < state_min:
            state_min = curr_min

        curr_max = np.max(state_values)
        if curr_max > state_max:
            state_max = curr_max        

    return state_min,state_max

def set_state_scalings(states, state_min, state_max):
    return np.round(preprocessing.minmax_scale(states,feature_range=(state_min,state_max),axis=0),2)


def denormalize_states(directory):
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                print(file_path)
                data = json.load(file)
            
            # Check if "states" exists and is a listgit 
            if "states" in data and isinstance(data["states"], list):
                arr = np.round(preprocessing.minmax_scale(data["states"],feature_range=(min,max),axis=0),2) # Normalize all the state to be between -1 and 1
                data["states"] = arr.tolist()
                # Save the updated JSON back to the file
                with open(file_path, 'w') as file:
                    json.dump(data, file)


if __name__ == "__main__":

    # Directory containing the .json files
    directory = '/home/mons/dev/private/master/saved_trajectories/normalized_dataset'
    # normalize_states(directory)