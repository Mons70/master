import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import colorcet as cc
import json
import os

def fetch_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_goal_pos(goal):
    keyframes = {
        "home": [-0.125, -0.625, 0.25],
        1: [-0.25, -1.0, 0.25],
        2: [-0.352465, -1.59862, 0.25],
        3: [-0.521426, -2.13349, 0.25],
        4: [-1.01012, -2.46729, 0.25],
        5: [-2.01529, -2.4747, 0.25],
        6: [-2.70571, -2.30555, 0.25],
        7: [-3.33213, -1.54147, 0.25],
        8: [-4.3052, -1.01978, 0.25],
        9: [-4.65979, -0.304562, 0.25],
        10: [-4.74108, 0.688841, 0.282658],
        11: [-4.72309, 2.03176, 0.332065],
        12: [-4.261, 2.18683, 0.482171],
        13: [-3.82627, 2.35773, 0.664309],
        14: [-3.42159, 2.52583, 0.74358],
        15: [-2.97588, 2.51519, 0.668477],
        16: [-2.12638, 2.40482, 0.294929],
        17: [-0.520779, 1.66374, 0.294929],
        18: [-0.0844852, 0.465219, 0.294929],
        19: [1.93695, -0.13810, 0.45],
        20: [-4.98318, -3.95614, 0.25],
        21: [0.79131, -4.68012, 0.25],
        22: [-0.02520, 3.68893, 0.9],
        23: [3.74955, -1.58986, 0.27],
        24: [4.48456, 2.75455, 0.30],
        25: [4.58215, -3.42454, 0.25],
        26: [2.43022, 4.54289, 0.25],
        27: [1.12297, 1.19466, 0.25],
        28: [-3.49162, -0.55558, 0.25],
        29: [-0.94072, 4.81276, 0.25],
        30: [2.14062, 0.77786, 0.92],
        31: [-1.30249, 0.09238, 0.47],
        32: [2.50175, 3.22686, 0.25],
        33: [4.69978, -4.66257, 0.25],
        34: [1.22319, -2.47120, 0.25],
        35: [-3.12424, 0.40762, 0.55],
        36: [4.77988, -0.56363, 0.25],
        37: [-3.55986, 4.69313, 0.25],
        38: [0.28912, -4.63385, 0.25],
        39: [1.65215, 4.04531, 0.33],
        40: [-0.87880, 3.33669, 0.25]
    }

    return keyframes[goal]

def plot_total_reward(policy_trajectories, time_horizon, show:bool = False):
    avg_policy_reward = {}
    for policy in policy_trajectories.keys():
        avg_policy_reward[policy] = {}
        #  For the current policy iterate through all goal tasks, and find mean distance from goal for all the runs from each goal task
        for goal_state in policy_trajectories[policy].keys():
            run_rewards = []
            for run in policy_trajectories[policy][goal_state]:
                # print(run.keys())
                total_rewards = np.array(run['total_reward'][0])
                goal = run['goal_state']
                print(total_rewards.shape)
                if len(total_rewards) < time_horizon:
                    total_rewards = np.pad(total_rewards, ((0, time_horizon - len(total_rewards))), mode='constant', constant_values=np.nan)
                    # print(total_rewards)
                elif len(total_rewards) > time_horizon:
                    total_rewards = total_rewards[:time_horizon]
                # The distances from goal per timestep for a single run is appended to list
                run_rewards.append(total_rewards)
            # print(len(run_rewards))

            # The mean distance from a specific goal across all the runs per timestep is added to the dictionary
            avg_policy_reward[policy][goal] = np.mean(run_rewards, axis=0)
    
    # print(avg_policy_reward)

    time = np.arange(0, time_horizon)

    for goal in avg_policy_reward['MPC'].keys():
        fig = plt.figure()
        for policy in avg_policy_reward.keys():
            print(goal)
            plt.plot(time[:], avg_policy_reward[policy][goal], label=policy)

        plt.legend()
        plt.xlabel("Timesteps")
        plt.ylabel("Total reward")
        plt.title(f'Averge Total reward for goal task {goal}')
        if show:
            plt.show()

def plot_control(policy_trajectories, time_horizon, show:bool = False):


    # ctrl trajectories are shape 12xtime_horizon
    avg_policy_ctrl = {}
    for policy in policy_trajectories.keys():
        avg_policy_ctrl[policy] = {}
        #  For the current policy iterate through all goal tasks, and find mean distance from goal for all the runs from each goal task
        for goal_state in policy_trajectories[policy].keys():
            run_controls = []
            for run in policy_trajectories[policy][goal_state]:
                # print(run.keys())
                controls = np.array(run['ctrl'])
                padded_controls = np.zeros((len(controls), time_horizon))
                # print(controls.shape)
                goal = run['goal_state']
                # print(controls.shape)
                for i in range(len(controls)):
                    print(len(controls[i]))
                    padded_controls[i] = np.pad(controls[i], ((0, time_horizon - len(controls[i]))), mode='constant', constant_values=np.nan)
                    # print(controls)
                # The distances from goal per timestep for a single run is appended to list
                run_controls.append(padded_controls)
        
            print(np.array(run_controls).shape)

            # The mean distance from a specific goal across all the runs per timestep is added to the dictionary
            print(len(run_controls))
            avg_policy_ctrl[policy][goal] = np.mean(run_controls, axis=0)


    time = np.arange(0, time_horizon)

    colors = cc.glasbey_light[:len(avg_policy_ctrl[list(avg_policy_ctrl.keys())[0]][1])]
    labels = ["FR_hip_joint","FR_thigh_joint","FR_calf_joint","FL_hip_joint","FL_thigh_joint","FL_calf_joint","RR_hip_joint","RR_thigh_joint","RR_calf_joint","RL_hip_joint","RL_thigh_joint","RL_calf_joint"]
    
    for goal in avg_policy_ctrl[list(avg_policy_ctrl.keys())[0]].keys():
        fig, axs = plt.subplots(3,1)
        for i, policy in enumerate(avg_policy_ctrl.keys()):
            print(goal)
            axs[i].set_title(f'Control signals for {policy}, Goal task {goal}')
            for signal in range(len(avg_policy_ctrl[policy][goal])):
                print(signal)
                print(avg_policy_ctrl[policy][goal][signal])
                axs[i].plot(time[:], avg_policy_ctrl[policy][goal][signal], label=labels[signal])

            axs[i].set_xlabel("Timesteps")
            axs[i].set_ylabel("Signal strength")

        fig.legend()
        if show:
            plt.show()

def plot_body_height(policy_trajectories, time_horizon, show:bool = False):
    avg_policy_body_height = {}
    for policy in policy_trajectories.keys():
        avg_policy_body_height[policy] = {}
        #  For the current policy iterate through all goal tasks, and find mean distance from goal for all the runs from each goal task
        for goal_state in policy_trajectories[policy].keys():
            run_body_heights = []
            for run in policy_trajectories[policy][goal_state]:
                # print(run.keys())
                body_heights = np.array(run['body_height'])
                goal = run['goal_state']
                print(body_heights.shape)
                if len(body_heights) < time_horizon:
                    body_heights = np.pad(body_heights, ((0, time_horizon - len(body_heights))), mode='constant', constant_values=np.nan)
                    # print(body_heights)
                # The distances from goal per timestep for a single run is appended to list
                run_body_heights.append(body_heights)
            # print(len(run_body_heights))

            # The mean distance from a specific goal across all the runs per timestep is added to the dictionary
            avg_policy_body_height[policy][goal] = np.mean(run_body_heights, axis=0)
    
    # print(avg_policy_body_height)

    time = np.arange(0, time_horizon)

    for goal in avg_policy_body_height['MPC'].keys():
        fig = plt.figure()
        for policy in avg_policy_body_height.keys():
            print(goal)
            plt.plot(time[:], avg_policy_body_height[policy][goal], label=policy)

        plt.legend()
        plt.xlabel("Timesteps")
        plt.ylabel("Height(m)")
        plt.hlines(0,0, time_horizon, colors='black', linestyles='solid', lw=1)
        plt.hlines(0.25,0, time_horizon, colors='black', linestyles='dashed')
        plt.title("Averge body height above feet")
        if show:
            plt.show()

def plot_mean_distance_to_goal(policy_trajectories, time_horizon, show:bool = False):
    avg_policy_distance = {}
    for policy in policy_trajectories.keys():
        avg_policy_distance[policy] = {}
        #  For the current policy iterate through all goal tasks, and find mean distance from goal for all the runs from each goal task
        for goal_state in policy_trajectories[policy].keys():
            run_distances = []
            for run in policy_trajectories[policy][goal_state]:
                # print(run.keys())
                states = run['states']
                goal = run['goal_state']
                positions = np.array(states)[:, :3]
                goal_pos = get_goal_pos(int(goal))
                # print(np.array(positions).shape)
                if len(positions) < time_horizon:
                    positions = np.pad(positions, ((0, time_horizon - len(positions)), (0, 0)), mode='constant', constant_values=np.nan)
                    # print(positions)
                # The distances from goal per timestep for a single run is appended to list
                run_distances.append([np.linalg.norm(position - goal_pos) for position in positions])
            # print(len(run_distances))
            # print(run_distances)
            # The mean distance from a specific goal across all the runs per timestep is added to the dictionary
            avg_policy_distance[policy][goal] = np.mean(run_distances, axis=0)

    time = np.arange(0, time_horizon)


    for goal in avg_policy_distance['MPC'].keys():
        fig = plt.figure()
        for policy in avg_policy_distance.keys():
            print(goal)
            policies = list(avg_policy_distance.keys())
            plt.plot(time[:], avg_policy_distance[policy][goal], label=policy)

        plt.legend()
        plt.xlabel("Timesteps")
        plt.ylabel("Distance(m)")
        plt.title("Mean distance from goal")
        if show:
            plt.show()

if __name__ == "__main__":

    policy_trajectories = fetch_data('/home/mons/dev/private/master/tests/test_data.json')

    print(len(policy_trajectories['MPC']['1'][0]['ctrl'][0]))
    # Plot reward (average pr timestep)
    # plot_total_reward(policy_trajectories, 599, True)

    #plot control signals ( pr timestep)
    plot_control(policy_trajectories, 200, True)

    # Plot body height (average pr timestep)
    # plot_body_height(policy_trajectories, 600, True)

    # Plot distance to goal (average pr timestep)
    # plot_mean_distance_to_goal(policy_trajectories, 600, True)