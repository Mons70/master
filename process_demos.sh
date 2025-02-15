#!/bin/bash

source ./master_venv/bin/activate
# variables
demoDir=$(realpath "$1") # absolute path to dataset

# make a hdf5 file from all the trajectories
python3 -m convert_trajectories_to_hdf5 --directory $demoDir

# data post-processing (90-10 validation split)
cd robomimic/robomimic/scripts/conversion
fileName="${demoDir}/demo/demo.hdf5"
python3 -m convert_robosuite --dataset $fileName

# convert states to observations
cd .. # script folder
newFileName="low_dim.hdf5"
python3 dataset_states_to_obs.py --dataset $fileName --output_name $newFileName --done_mode 1

# check data structure
datasetName="${demoDir}/demo/${newFileName}"
python3 get_dataset_info.py --dataset $datasetName
