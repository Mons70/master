#!/bin/bash

source ./master_venv/bin/activate

obs_size=$1
# variables
demoDir=$(realpath "$2") # absolute path to dataset

# make a hdf5 file from all the trajectories
python ./helper_scripts/convert_trajectories_to_hdf5.py --directory $demoDir

# data post-processing (90-10 validation split)
cd robomimic/robomimic/scripts/conversion
fileName="${demoDir}/demo/demo.hdf5"
python -m convert_robosuite --dataset $fileName

# convert states to observations
cd .. # script folder
newFileName="low_dim.hdf5"
python dataset_states_to_obs.py --dataset $fileName --output_name $newFileName --done_mode 1 --obs_size $obs_size

# check data structure
datasetName="${demoDir}/demo/${newFileName}"
python get_dataset_info.py --dataset $datasetName
