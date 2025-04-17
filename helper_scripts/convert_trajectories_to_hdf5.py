import os
import time
import argparse
import datetime
import h5py
import json
import csv
import ast
from array import array
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn import preprocessing

def normalize_to_minus_one_one(arr):
    X = np.array(arr)
    # arr_min, arr_max = arr.min(), arr.max()
    # print(arr_min)
    # print(arr_max)
    # return ((2*(arr - arr_min))/(arr_max - arr_min)) - 1
    return np.round(preprocessing.minmax_scale(arr,feature_range=(-1,1),axis=0),2)

def convert_csv_hdf5(directory, output_dir, env_info):
    """
    Parameters:
    -------------------------------------
    directory (str): Path to the directory containing raw demonstrations.

    output_dir (str): Path to where to store the hdf5 file.

    ####env_info (str): JSON-encoded string containing environment information,
    ####                including controller and robot info
    -------------------------------------

    Returns:
    -------------------------------------
    hdf5 file containing dataset of robosuite compatible RL demonstrations.

    The strucure of the hdf5 file is as follows.

        data (group)
            date (attribute) - date of collection
            time (attribute) - time of collection
            repository_version (attribute) - repository version used during collection
            env (attribute) - environment name on which demos were collected

            demo1 (group) - every demonstration has a group
                model_file (attribute) - model xml string for demonstration
                states (dataset) - flattened mujoco states
                actions (dataset) - actions applied during demonstration

            demo2 (group)
            ...
    """

    try:
        path = os.path.join(output_dir, "demo.hdf5")
    except:
        pass
    f = h5py.File(path, "w")

    #stores some metadata for the demos as attributes in a group
    data_grp = f.create_group("data")

    env_name = 'Push' # will get populated at some point

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    data_grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    data_grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    data_grp.attrs["repository_version"] = "0"
    data_grp.attrs["env"] = env_name
    data_grp.attrs["env_info"] = env_info

    num_eps = 0

    # for ep_directory in os.listdir(directory):

    ep_paths = os.path.join(directory, "ep_*.json")
    states = []
    actions = []
    rewards = []
    total_reward = []

    for ep_file in sorted(glob(ep_paths)):
        with open(ep_file, 'r') as f:
            data_dictionary = json.load(f)

        states = np.array(data_dictionary['states'])

        actions = np.array([ai['actions'] for ai in data_dictionary['actions']])
        print(actions.shape)
        actions = normalize_to_minus_one_one(actions)
        print(actions.shape)
        # print(actions[10])

        rewards = np.array(data_dictionary['rewards'])
        total_reward = np.array(data_dictionary['total_reward'])

        assert len(states) == len(actions)

        # print(states.shape)
        # print(actions.shape)

        num_eps += 1
        ep_data_grp = data_grp.create_group('demo_{}'.format(num_eps))
        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        #ep_data_grp.create_dataset("goal", data=np.array(goal))


    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join("/tmp/", "demonstrations"),
    )
    args = parser.parse_args()

    # Create argument configuration
    config = {
        "env_name": "Hill",
        "robots": "Quadruped",
        "controller_configs": None,
    }

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # make a new timestamped directory
    try:
        new_dir = os.path.join(args.directory, "demo")
        os.makedirs(new_dir)
    except:
        new_dir = os.path.join(args.directory, "demo")

    convert_csv_hdf5(args.directory, new_dir, env_info)
