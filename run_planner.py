import gc
from mjpc_agent_wrapper import *
from tqdm import tqdm
# rollout horizon
T = 1000
num_eps = 60
for i in tqdm(range(385, num_eps + 385)):
    # Initialize model, agent, data and renderer for simulation
    model, agent, data, renderer = init_model_task("/home/mons/dev/private/master/mujoco_mpc/build/mjpc/tasks/cartpole/task.xml", "Cartpole", render=True)
    # Set the agents' weights
    agent.set_cost_weights({"Velocity": 0.15})
    print("Cost weights:", agent.get_cost_weights())

    # Set the task parameters, e.g. goal
    agent.set_task_parameter("Goal", -1.0)
    print("Parameters:", agent.get_task_parameters())

    # Run planner
    #qpos, qvel, ctrl, cost_terms, cost_total = run_planner(model, agent, data, renderer, T, True, False, savepath = f'./saved_trajectories/trajectories_model_{i}.csv')

    states, actions, rewards, total_reward = run_planner(model, agent, data, renderer, T, True, True, savepath = f'./saved_trajectories/ep_{i}.json')
    
    if i % 25 == 0:
        gc.collect()

# plot states
plot_states(states)
# plot actions
plot_actions(actions)
# plot costs
plot_rewards(agent, rewards, total_reward, show = True)
