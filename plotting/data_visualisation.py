import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import colorcet as cc
import json
import os


def distance_from_home(stage_name):

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



    home_pos = np.array(keyframes["home"])
    stage_pos = np.array(keyframes[stage_name])
    return np.linalg.norm(home_pos - stage_pos)

def total_percent_reached_goals(directory, time_limit):
    goals_reached = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            if len(data["states"]) < time_limit:
                goals_reached.append(1)
            else:
                goals_reached.append(0)
    
    num_reached = np.sum(goals_reached)
    num_goals = len(goals_reached)

    return (num_reached/num_goals)*100, num_reached, num_goals #Return percent reached, number of reached goals and total number of goals/trajectories in the dataset


def plot_amount_reached_pr_goal_state(directory, time_limit):
    """
    Plots the amount of successfully reached goals within the timeframe for each goal state.
    Each bar is numbered sequentially on the x-axis.
    """
    goals_reached = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            goal_state = data["goal_state"]

            if goal_state not in goals_reached:
                goals_reached[goal_state] = []

            if len(data["states"]) < time_limit:
                goals_reached[goal_state].append(1)
            else:
                goals_reached[goal_state].append(0)

    goals_reached_sorted = dict(sorted(goals_reached.items(), key=lambda item:distance_from_home(item[0])))

    goal_keys = list(goals_reached_sorted.keys())
    num_goals = len(goal_keys)

    colors = cc.glasbey_light[:num_goals]
    # pale_colors = [to_rgba(c = color, alpha = 0.3) for color in colors]
    pale_colors = [(0.77734375, 0.77734375, 0.77734375) for _ in range(num_goals)]

    fig, ax = plt.subplots()

    x_val = 0  # 1-based x-axis
    x_vals = []

    for i, key in zip(range(num_goals),goal_keys):
        values = goals_reached_sorted[key]
        reached = sum(values)
        possible = len(values)
        percent = reached / possible * 100 if possible > 0 else 0
        if percent <= 100:
            x_val += 1
            x_vals.append(key)
            ax.bar(x_val, possible, color=pale_colors[i], edgecolor='black', label='Possible' if i == 0 else "", ls='--', hatch='/')
            ax.bar(x_val, reached, color=colors[i], edgecolor='black', label='Reached' if i == 0 else "")

            ax.text(
                x_val,
                reached / 2,
                f"{percent:.0f}%",
                ha='center', va='center',
                color='black', fontsize=9, weight='demibold', rotation=340
            )

    ax.set_xlabel("Goal Number")
    ax.set_ylabel("Number of runs")
    ax.set_title("Reached vs Possible Goals before time limit")
    ax.set_xticks(range(1, len(x_vals)+1))
    ax.set_xticklabels(x_vals)  # Optional: use goal_keys instead if you prefer
    # Create top x-axis
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # match bottom x-axis limits
    ax_top.set_xticks(range(1, len(x_vals)+1))
    ax_top.set_xticklabels(np.round([distance_from_home(x) for x in x_vals], 2))  # custom labels
    ax_top.set_xlabel("Distance from start(m)")

    plt.tight_layout()

    plt.savefig("/home/mons/dev/private/thesis-paper/figures/percentage_reached_pr_goal_state.pgf")




def plot_average_reward_pr_goal_state(directory, time_limit):
    """
    Plots the average reward for each timestep for each goal state.
    Assumes each .json contains a 'goal_state' and a list 'total_reward'.
    Pads shorter reward lists with np.nan to handle varying lengths.
    """
    goal_rewards = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file)

            goal_state = str(data["goal_state"])  # Ensure string keys
            rewards = data["total_reward"][0]

            if goal_state not in goal_rewards:
                goal_rewards[goal_state] = []

            goal_rewards[goal_state].append(rewards)

    average_goal_rewards = {}
    for goal in goal_rewards:
        padded = [r + [np.nan] * (time_limit - len(r)) if len(r) < time_limit else r[:time_limit] for r in goal_rewards[goal]]
        average = np.nanmean(padded, axis=0)
        average_goal_rewards[goal] = average

    
    sorted_average_reward_dict = dict(sorted(average_goal_rewards.items(), key=lambda item: distance_from_home(int(item[0]))))
    distance_list = [distance_from_home(int(goal)) for goal in sorted_average_reward_dict.keys()]

    num_goals = len(sorted_average_reward_dict.keys())
    colors = cc.glasbey_light[:num_goals]

    fig, ax = plt.subplots()
    for i, (goal, avg_rewards) in zip(range(num_goals), sorted_average_reward_dict.items()):
        ax.plot(avg_rewards, label=f"Stage {goal}: {distance_list[i]:.2f} m", color=colors[i], linewidth=2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward per Timestep per Goal State")

    # Place legend outside the plot (right side)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Goal States")

    plt.tight_layout()
    plt.savefig("/home/mons/dev/private/thesis-paper/figures/avg_reward_pr_timestep.pgf")



def plot_distance_from_start_pr_goal_state(num_goals):
    """
    Plots the length from start for each goal state
    """

    distance_from_start = {}
    for i in range(1, num_goals+1):
        distance_from_start[i] = distance_from_home(i)
    
    colors = cc.glasbey_light[:num_goals]
    # Plotting
    fig, ax = plt.subplots()
    for i, key in enumerate(distance_from_start.keys()):
        # Plot the average termination reward for each goal state
        ax.bar(key, distance_from_start[key], color=colors[i], edgecolor='black', label=f"Goal {key}")
    
    ax.set_xlabel("Goal state")
    ax.set_xticks(range(1, num_goals+1))
    ax.set_xticklabels(distance_from_start.keys())
    ax.set_ylabel("Distance from start (m)")
    ax.set_title("Distance from start to goal for each Goal State")
 
    plt.tight_layout()
    plt.savefig("/home/mons/dev/private/thesis-paper/figures/distance_from_start.pgf")



def plot_cumulative_reward_pr_goal_state(directory, time_limit):
    """
    Plots the cumulative reward for each goal state
    """
    goal_cumulative_rewards = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            goal_state = data["goal_state"]
            if goal_state not in goal_cumulative_rewards:
                goal_cumulative_rewards[goal_state] = []

            cumulative_reward = sum(data["total_reward"][0])
            goal_cumulative_rewards[goal_state].append(cumulative_reward)
    
    sorted_cumulative_reward_dict = dict(sorted(goal_cumulative_rewards.items(), key=lambda item: distance_from_home(int(item[0]))))
    distance_list = [distance_from_home(int(goal)) for goal in sorted_cumulative_reward_dict.keys()]

    goal_keys = sorted_cumulative_reward_dict.keys()
    num_goals = len(goal_keys)

    colors = cc.glasbey_light[:num_goals]
    # cmap = matplotlib.colormaps.get_cmap('tab10')  # or try 'viridis', 'plasma', etc.
    # colors = cmap(np.linspace(0, 1, num_goals))

    # Plotting
    fig, ax = plt.subplots()
    for key, i in zip(sorted_cumulative_reward_dict.keys(), range(num_goals)):
        # Plot the cumulative reward for each goal state
        ax.bar((i+1), np.mean(goal_cumulative_rewards[key]), color=colors[i], edgecolor='black', label=f"Goal {key}")
    ax.set_xlabel("Goal state")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward for Goal States")
    ax.set_xticks(range(1, num_goals + 1))
    ax.set_xticklabels(goal_keys)  # Use goal keys for x-tick labels
    # Create top x-axis
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # match bottom x-axis limits
    ax_top.set_xticks(range(1, num_goals + 1))
    ax_top.set_xticklabels(np.round(distance_list, 2), rotation = 310)  # custom labels
    ax_top.set_xlabel("Distance from start(m)")
    plt.tight_layout()
    plt.savefig("/home/mons/dev/private/thesis-paper/figures/avg_cumulative_reward.pgf")
    # plt.show()


def main():
    """
    Main function to run the plotting functions
    """
    directory = "/home/mons/dev/private/master/saved_trajectories/new_dataset"
    time_limit = 600  # Set your time limit here

    matplotlib.use('pgf')
    matplotlib.rcParams.update(
        {
            'pgf.texsystem': 'pdflatex',
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        }
    )

    # plot_amount_reached_pr_goal_state(directory, time_limit)
    # plot_average_reward_pr_goal_state(directory, time_limit)
    # percentage, num_reached, total_goals = total_percent_reached_goals(directory, time_limit)
    # print(percentage, num_reached, total_goals)
    plot_distance_from_start_pr_goal_state(18)
    # plot_cumulative_reward_pr_goal_state(directory, time_limit)

if __name__ == "__main__":
    main()