import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os

def plot_amount_reached_pr_goal_state(directory, time_limit):
    """
    Plots the amount of successfully reached goals within the timeframe for each goal state
    """
    goals_reached = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            goal_state = data["goal_state"]

            if goal_state not in goals_reached:
                goals_reached[goal_state] = []

            if len(data["states"]) < time_limit:
                goals_reached[goal_state].append(1)
            else:
                goals_reached[goal_state].append(0)
    # Count the number of successful goal states
    goals_reached = {}

    num_goals = len(goals_reached.keys())
    cmap = cm.get_cmap("tab20", num_goals)
    colors = [cmap(i) for i in range(num_goals)]
    pale_colors = [(r, g, b, 0.3) for r, g, b, _ in colors]

    for goal in goals_reached:
        if goal not in goals_reached:
                goals_reached[goal] = []

        goals_reached[goal] = [sum(goal), len(goal)]
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in goals_reached.keys():
        # Plot the number of reached and possible finished for a given goal_state
        reached = goals_reached[key][0]
        possible = goals_reached[key][1]
        percent = reached / possible * 100 if possible > 0 else 0
        ax.bar(key, possible, color=pale_colors[key], edgecolor='black', label='Possible')
        ax.bar(key, reached, color=colors[key], edgecolor='black', label='Reached')
        # Add percentage text
        ax.text(
            key,                         # x
            reached / 2,                 # y (center of bar)
            f"{percent:.0f}%",           # text
            ha='center', va='center',    # center alignment
            color='white', fontsize=10   # styling)
        )

    # Labels
    ax.set_xlabel("Task index")
    ax.set_ylabel("Number of goals")
    ax.set_title("Reached vs Possible Goals")
    ax.legend()

    plt.tight_layout()
    plt.show()



def plot_average_reward_pr_goal_state(directory, time_limit):
    """
    Plots the average reward for each timestep for each goal state
    """
    goal_rewards = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            goal_state = data["goal_state"]
            rewards = data["total_reward"]

            if goal_state not in goal_rewards:
                goal_rewards[goal_state] = []

            goal_rewards[goal_state].append(rewards)
    
    average_goal_rewards = {} 
    # Calculate average rewards
    for goal in goal_rewards:
        if goal_state not in average_goal_rewards:
                average_goal_rewards[goal_state] = []
        average_goal_rewards[goal_state] = np.mean(goal_rewards[goal], axis=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in average_goal_rewards.keys():
        # Plot the average reward for each goal state
        ax.plot(average_goal_rewards[key], label=f"Goal {key}")
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward for Goal States pr timestep")

    ax.legend()
    plt.tight_layout()
    plt.show()



def plot_average_termination_reward_pr_goal_state(directory, time_limit):
    """
    Plots the average termination reward for each goal state
    """
    goal_termination_rewards = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            goal_state = data["goal_state"]
            termination_reward = data["total_reward"][-1]

            if goal_state not in goal_termination_rewards:
                goal_termination_rewards[goal_state] = []
            goal_termination_rewards[goal_state].append(termination_reward)
    # Calculate average termination rewards
    average_goal_termination_rewards = {}

    for goal in goal_termination_rewards:
        if goal_state not in average_goal_termination_rewards:
                average_goal_termination_rewards[goal_state] = []
        average_goal_termination_rewards[goal_state] = np.mean(goal_termination_rewards[goal], axis=0)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in average_goal_termination_rewards.keys():
        # Plot the average termination reward for each goal state
        ax.bar(key, average_goal_termination_rewards[key], label=f"Goal {key}")
    
    ax.set_xlabel("Goal state")
    ax.set_ylabel("Average Termination Reward")
    ax.set_title("Average Termination Reward for Goal States")
    ax.legend()
    plt.tight_layout()
    plt.show()



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
            cumulative_reward = sum(data["total_reward"])/len(data["total_reward"])
            
            goal_cumulative_rewards[goal_state] = cumulative_reward
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in goal_cumulative_rewards.keys():
        # Plot the cumulative reward for each goal state
        ax.bar(key, goal_cumulative_rewards[key], label=f"Goal {key}")
    ax.set_xlabel("Goal state")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward for Goal States")
    ax.legend()
    plt.tight_layout()
    plt.show()