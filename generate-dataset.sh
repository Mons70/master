#!/bin/bash

#Usage ./generate-dataset.sh [max-mpc-demo-duration] [num-mpc-demos] [path-to-directory-with-mpc-demos]
#Example: ./generate-dataset.sh 3000 1000 ./saved_trajectories

source ./master_venv/bin/activate

#Generate MPC demonstrations
python run_planner.py --max_demo_duration $1 --num_demonstrations $2

# variables
demoDir=$(realpath "$3") # absolute path to dataset

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