import gc
from mjpc_agent_wrapper import *
from tqdm import tqdm
import os
import psutil


def collect_demonstration(ep_number: int, time_horizon: int, plot: bool, render: bool):
    # Initialize model, agent, data and renderer for simulation
    model, agent, data, renderer = init_model_task("/home/mons/dev/private/master/mujoco_mpc/build/mjpc/tasks/cartpole/task.xml", "Cartpole", render)
    # Set the agents' weights
    agent.set_cost_weights({"Velocity": 0.15})
    print("Cost weights:", agent.get_cost_weights())

    # Set the task parameters, e.g. goal
    #Set goal position to be random rather than in -1 all the time
    # agent.set_task_parameter("Goal",  ((np.random.rand(1)*2) - 1.0)[0])

    print("Parameters:", agent.get_task_parameters())

    # Run planner
    #qpos, qvel, ctrl, cost_terms, cost_total = run_planner(model, agent, data, renderer, T, True, False, savepath = f'./saved_trajectories/trajectories_model_{i}.csv')
    np.random.seed(ep_number)
    states, actions, rewards, total_reward = run_planner(model, agent, data, renderer, time_horizon, True, True, savepath = f'./saved_trajectories/ep_{ep_number}.json')
    
    if plot:
        # plot states
        plot_states(states)
        # plot actions
        plot_actions(actions)
        # plot costs
        plot_rewards(agent, rewards, total_reward, show = True)

    mem_info = psutil.virtual_memory()
    print(f"Total Memory: {mem_info.total / 1024**2:.2f} MB")
    print(f"Used Memory: {mem_info.used / 1024**2:.2f} MB")
    print(f"Available Memory: {mem_info.available / 1024**2:.2f} MB")


if __name__ == "__main__":
    # rollout horizon
    T = 1000
    num_eps = 500
    for i in tqdm(range(300,num_eps)):
        if i < num_eps-1:
            collect_demonstration(i, T, False, False)
        else:
            collect_demonstration(i, T, True, True)
