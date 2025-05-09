# Master_thesis
All source code and documents for my master thesis


# Training Real-world robust offline Reinforcement learning (RL) policies using Model Predictive Control (MPC) trajectories

## Project progess steps and status:

- 1: Wrapper for Mujoco MPC (Mjpc) to run and extract MPC trajectories
    - Status: Done (Might need updates w/regards to dataset conversion and formatting)
- 2: Formatting/Conversion of Mjpc trajectories to Robomimic compatible hdf5 dataset
    - Status: WIP, currently underway by using solution devloped by shin for faking pybullet to appear as robosuite in the Robomimic script for making robosuite datasets
      compatible with Robomimic. This is slightly modified to suit my specific task (currently the cartpole task). I think this should part should be implemented as it's own Mjpc
      wrapper rather than tweaking it to work with the robosuite wrapper. This would improve ease of use and avoid having to dig into the wrapper code to e.g. change the state
      formats. And it should should ideally enable the user to train an RL policy on any Mjpc task by just changing the specifications and dataset in some config.
- 3: Run X amount of Mjpc runs of the cartpole task (e.g. 10000) and extract all trajectories, convert to Robomimic dataset and train RL policy
    - Status: To be done.
- 4: Review results of the trained cartpole policy. Hopefully get some results showing the promise of the approach. If so, increase task complexity.
- 5: Repeat step 3, but for a more complex Mjpc task, e.g. quadruped walking. (Can be a custom defined task (Evalute at time of arrival on step 5 if it's worth it/feasible to
    expect within deadline 15th of May).
    - Status: To be done.

In parallel:
- Write actual thesis.

TODO:
- Implement stopping when reached certain cost/reward threshold for mjpc trajectory collection to avoid too short/unfinished or unnecessarily long trajectories
    NEEDS REDOING: stops after set threshold, but i think the trajectories are
    initialized as length = max time horizon, meaning they contain x amount of null
    samples after ended MPC planning.
- Modularize/generalize code further to enable easier switching between mjpc tasks (i.e. not having to edit source code to adjust which trajectories a state 
  and actions consists, should only be specified in a config). 
- Implement a new robosuite environment/wrapper for mujoco/mujoco MPC to allow for compatibility with robomimic without having to alter its source code/fake
  the robosuite environment to allow training a model.
