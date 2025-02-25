# %%
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import torch
import numpy as np
import pathlib
import cv2
import csv
import json
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from mujoco_mpc import agent as agent_lib

def plot_states(states, time_horizon, show:bool = False):
    #State shape = timesteps x dim(state space)
    fig = plt.figure()
    time = np.arange(0, time_horizon)
    states[:,2] = states[:,2] % 2*np.pi
    plt.plot(time, states, label=['Horizontal position','Radial position','Vel_1','Vel_2'])
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("State values")
    if show:
       plt.show()


def plot_actions(actions, time_horizon, show:bool = False):
    fig = plt.figure()
    time = np.arange(0, time_horizon)
    actions_list = []
    for dict in actions:
        actions_list.append(*dict.values())
    print(actions_list)
    plt.plot(time, actions_list)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Control")
    if show:
       plt.show()

def plot_rewards(agent, rewards, total_reward, time_horizon, show:bool = False):
    fig = plt.figure()
    time = np.arange(0, time_horizon)
    for i, c in enumerate(agent.get_cost_term_values().items()):
        plt.plot(time[:], rewards[i, :], label=c[0])

    plt.plot(time[:], total_reward[0,:], label="Total (weighted)", color="black")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Rewards")
    if show:
        plt.show()

def render_video(frames, framerate, playback_speed):
    # Define the codec and create VideoWriter object
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
    out = cv2.VideoWriter("./policy-inference-simulations/policy_inference.mp4", fourcc, np.round(playback_speed*framerate), (width, height))
    assert out.isOpened()

    for frame in frames:
        out.write(frame)

    out.release()

def init_model_task(model:str, task_id:str, render:bool = False):
    model_path = (
        pathlib.Path(model)
            )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # data
    data = mujoco.MjData(model)

    # renderer
    if render:
        renderer = mujoco.Renderer(model)
    else:
        renderer = None

    # agent
    agent = agent_lib.Agent(task_id=task_id, model=model)

    return model, agent, data, renderer


def init_trajectories(model, time_horizon):
    #Set number of timesteps T
    # TODO: Implement actual number of timesteps T from time horizon / timestep variable
    T = time_horizon

    #trajectories
    qpos = np.zeros((model.nq, T)) # model.nq = number of generalized coordinates in the model = dim(qpos)
    qvel = np.zeros((model.nv, T)) # model.nv = number of degrees of freedom in the model = dim(qvel)
    ctrl = np.zeros((model.nu, T)) # model.nu = number of actuators/controls = dim(ctrl)
    time = np.zeros(T) #number of timesteps in defined unit, unit defined in the model .xml file
    return qpos, qvel, ctrl, time


def set_initial_state(model, data, qpos,qvel,time):
    # rollout
    mujoco.mj_resetData(model, data)
    # cache initial state
    qpos[:, 0] = data.qpos
    qvel[:, 0] = data.qvel
    time[0] = data.time

    return qpos, qvel, time


def set_random_initial_state(model,data,qpos,qvel,time):
    # rollout
    mujoco.mj_resetData(model, data)
    horizontal_pos = (np.random.rand(1)*1.8) - 0.9
    vertical_pos = (np.random.rand(1)*2*np.pi) - np.pi
    print(horizontal_pos)
    print(vertical_pos)
    data.qpos = [horizontal_pos[0], vertical_pos[0]]
    # cache initial state
    qpos[:, 0] = data.qpos
    qvel[:, 0] = data.qvel
    print(qpos[:,0])
    print(qvel[:,0])
    time[0] = data.time

    return qpos, qvel, time


def init_agent_cost_terms(agent, time_horizon):
    #Set number of timesteps T
    # TODO: Implement actual number of timesteps T from time horizon / timestep variable
    T = time_horizon

    # costs
    cost_total = np.zeros((1,T))
    cost_terms = np.zeros((len(agent.get_cost_term_values()), T))
    return cost_terms, cost_total


def run_policy(model, agent, data, renderer, time_horizon, random_initial_state:bool = False, policy_path:str = "~/dev/private/master/"):
    T = time_horizon

    # restore policy
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=policy_path, device=device, verbose=True)
    policy.start_episode()

    #trajectories
    qpos, qvel, ctrl, time = init_trajectories(model, T)

    # costs
    cost_terms, cost_total = init_agent_cost_terms(agent, T)

    # If a renderer is specified, initialize array to store frames
    if renderer == None:
        render = False
    else:
        render =True
        frames = []

    if random_initial_state:
        qpos, qvel, time = set_random_initial_state(model, data, qpos,qvel,time)
    else:
        qpos, qvel, time = set_initial_state(model, data, qpos,qvel,time)

    # simulate
    for t in range(T-1):
        if t % 100 == 0:
            print("\rt = ", t)

        # set planner state
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,               #Insert policy action here?
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        # Fetch current state as an observation
        obs = {"pos": np.array(data.qpos),"vel": np.array(data.qvel)}
        # print(obs)

        # set ctrl from RL model policy
        data.ctrl = policy(ob=obs)
        ctrl[:, t] = data.ctrl

        # get costs
        cost_total[0][t] = agent.get_total_cost()
        for i, c in enumerate(agent.get_cost_term_values().items()):
            cost_terms[i, t] = c[1]

        # step
        mujoco.mj_step(model, data)

        # cache
        qpos[:, t + 1] = data.qpos
        qvel[:, t + 1] = data.qvel
        time[t + 1] = data.time

        # If a renderer was specified, render and save frames
        if render:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

        if cost_total[0][t] < 0.005 and data.qvel[:] < 0.1:
            break
    # reset
    agent.reset()

    # If a renderer was specified, render video from frames and write to file
    if render:
        # display video
        FPS = 1.0 / model.opt.timestep #timestep is defined as an option in the model .xml file
        SLOWDOWN = 0.5
        render_video(frames, FPS, SLOWDOWN)
        renderer.close()

    #format trajectories to be compatible for robosuite hdf5 conversion:
    states = np.column_stack((np.transpose(qpos), np.transpose(qvel)))
    # print(states.shape)

    actions = []
    for t in range(T):
        actions.append({'actions': ctrl[:,t].tolist()})
    actions = np.array(actions)

    # print(actions.shape)

    rewards = cost_terms * -1
    total_reward = cost_total * -1

#    return qpos, qvel, ctrl, cost_terms, cost_total
    return states, actions, rewards, total_reward

def main():
    T = 1000
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

    states, actions, rewards, total_reward = run_policy(model, agent, data, renderer, T, True, policy_path="/home/mons/dev/private/master/robomimic/robomimic/./bc_trained_models/bc/20250215141548/models/model_epoch_8000.pth")
    print("STAAAAAAAAAAAAAAAAAAAAAAATES")
    print(states[-10:])
    # plot states
    plot_states(states, len(states))
    # plot actions
    plot_actions(actions, len(actions))
    # plot costs
    plot_rewards(agent, rewards, total_reward, len(total_reward[0]), show = True)

main()
