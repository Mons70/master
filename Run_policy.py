# %%
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pathlib
import cv2
import os
import json
import argparse
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from glob import glob
from sklearn import preprocessing
from mujoco_mpc import agent as agent_lib

def plot_states(states, time_horizon, show:bool = False):
    #State shape = timesteps x dim(state space)
    fig = plt.figure()
    time = np.arange(0, time_horizon)
    states[:,2] = states[:,2] % 2*np.pi
    plt.plot(time, states)
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

def render_video(frames, framerate, playback_speed, name):
    # Define the codec and create VideoWriter object
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
    out = cv2.VideoWriter(f"./policy-inference-simulations/{name}.mp4", fourcc, np.round(playback_speed*framerate), (width, height))
    assert out.isOpened()

    for frame in frames:
        out.write(frame)

    out.release()

def init_model_task(model:str, task_id:str, render:bool = False, render_resolution: tuple = None):
    model_path = (
        pathlib.Path(model)
            )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # data
    data = mujoco.MjData(model)

    # renderer
    if render:
        renderer = mujoco.Renderer(model, render_resolution[0], render_resolution[1])
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


def set_random_goal_state(agent, data):
    available_states = agent.get_all_modes()
    rand_idx = np.random.randint(1,len(available_states)-1)
    random_state = available_states[rand_idx]
    agent.set_mode(random_state)
    goal_pos = agent.model.key_mpos[rand_idx -1]
    goal_quat = agent.model.key_mquat[rand_idx -1]

    data.mocap_pos = goal_pos
    data.mocap_quat = goal_quat
    goal_dict_list = data.qpos.copy()
    # print("Initial state: ", goal_dict_list)
    # print(goal_pos)
    # print(goal_quat)
    mocap_goal = [*goal_pos, *goal_quat]
    # print("Mocap goal: ", mocap_goal)

    for i in range(7): #7 for the world + quat pos given in the mocap_pos
        goal_dict_list[i] = mocap_goal[i]
    
    goal_dict = {"pos": np.array(goal_dict_list)}
    # print("Goal state: ", goal_dict_list)
    return agent,data, goal_dict

def set_goal_state(state: int, agent, data):
    available_states = agent.get_all_modes()
    agent.set_mode(available_states[state])

    goal_pos = agent.model.key_mpos[state -1]
    goal_quat = agent.model.key_mquat[state -1]
    
    data.mocap_pos = goal_pos
    data.mocap_quat = goal_quat
    goal_dict_list = data.qpos.copy()
   
    mocap_goal = [*goal_pos, *goal_quat]

    for i in range(7): #7 first indxs for the world + quat pos given in the mocap_pos
        goal_dict_list[i] = mocap_goal[i]
    
    goal_dict = {"pos": np.array(goal_dict_list)}
    
    return agent,data, goal_dict


def get_action_scaling(directory: str):
    ep_paths = os.path.join(directory, "ep_*.json")
    action_min = 0
    action_max = 0
    for ep_file in sorted(glob(ep_paths)):
        with open(ep_file, 'r') as f:
            data_dictionary = json.load(f)
        action_values = np.array([ai['actions'] for ai in data_dictionary['actions']])
        curr_min = np.min(action_values)
        if curr_min < action_min:
            action_min = curr_min

        curr_max = np.max(action_values)
        if curr_max > action_max:
            action_max = curr_max        

    return action_min,action_max

def set_action_scalings(actions, action_min, action_max):
    return np.round(preprocessing.minmax_scale(actions,feature_range=(action_min,action_max),axis=0),2)


def set_camera_view(position: tuple = (0,0,0), elevation: int = -90, azimuth: int = 90, xml_cam_id: str = None):
    # Good tracking camera view for quadruped in xml format:
    # <camera name="robot_cam" pos="0 2 2" xyaxes="-1 0 0 0 -0.707 0.707" mode="trackcom"/>
    if xml_cam_id != None:
        camera = xml_cam_id
    else:    
        camera = mujoco.MjvCamera()
        camera.elevation = elevation
        camera.azimuth = azimuth
        camera.lookat = np.array(position)
    
    return camera

def run_policy(model, agent, data, renderer, time_horizon, random_initial_state:bool = False, goal_state = None, camera_id= None, policy_path:str = "~/dev/private/master/"):
    T = time_horizon

    render_video_name = policy_path.split("/")[-4:]
    render_video_name[-1] = render_video_name[-1].strip(".pth")
    render_video_name = '-'.join(render_video_name)
    print(render_video_name)

    # restore policy
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=policy_path, device=device, verbose=True)
    policy.start_episode()

    #trajectories
    qpos, qvel, ctrl, time = init_trajectories(model, T)

    # costs
    cost_terms, cost_total = init_agent_cost_terms(agent, T)

    # get max and min action values from collected data to scale policy actions back to environment corrected actions
    print("Fetching max- and min-action values for scaling ...")
    action_min, action_max = get_action_scaling("/home/mons/dev/private/master/saved_trajectories/1000_quads")


    if camera_id == None:
        camera = None
    else:
        camera = set_camera_view(xml_cam_id=camera_id)
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
    
    if goal_state != None:
        agent, data, goal_dict = set_goal_state(goal_state,agent, data)
    else:
        agent, data, goal_dict = set_random_goal_state(agent=agent, data=data)


    # simulate
    for t in range(T-1):
        if t % 100 == 0:
            print("\rt = ", t)

        # set planner state
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        # Fetch current state as an observation
        obs = {"pos": np.array(data.qpos),"vel": np.array(data.qvel)}
        print("OBS: ", obs)
        print("GOAL_OBS: ", goal_dict)

        # set ctrl from RL model policy
        data.ctrl = set_action_scalings(policy(ob=obs, goal=goal_dict), action_min, action_max)
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
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            frames.append(pixels)

        if cost_total[0][t] < 0.25:
            qpos = qpos[:, :t+1]
            qvel = qvel[:, :t+1]
            ctrl = ctrl[:, :t+1]
            cost_terms = cost_terms[:,:t+1]
            cost_total = cost_total[:,:t+1]
            break
    # reset
    agent.close()

    # If a renderer was specified, render video from frames and write to file
    if render:
        # display video
        FPS = 1.0 / model.opt.timestep #timestep is defined as an option in the model .xml file
        SLOWDOWN = 0.5
        render_video(frames, FPS, SLOWDOWN, render_video_name)
        renderer.close()

    #format trajectories to be compatible for robosuite hdf5 conversion:
    states = np.column_stack((np.transpose(qpos), np.transpose(qvel)))
    # print(states.shape)

    actions = []
    for t in range(len(states)):
        actions.append({'actions': ctrl[:,t].tolist()})
    actions = np.array(actions)

    # print(actions.shape)

    rewards = cost_terms * -1
    total_reward = cost_total * -1

#    return qpos, qvel, ctrl, cost_terms, cost_total
    return states, actions, rewards, total_reward

def main(policy_path):
    T = 1000
    # Initialize model, agent, data and renderer for simulation
    model, agent, data, renderer = init_model_task("/home/mons/dev/private/master/mujoco_mpc/build/mjpc/tasks/quadruped/task_hill.xml", "Quadruped Hill", render=True, render_resolution=(480, 640))

    # Set the agents' weights
    # agent.set_cost_weights({"Velocity": 0.15})
    print("Cost weights:", agent.get_cost_weights())

    # Set the task parameters, e.g. goal
    # agent.set_task_parameter("Goal", -1.0)
    print("Parameters:", agent.get_task_parameters())

    # Run planner
    #qpos, qvel, ctrl, cost_terms, cost_total = run_planner(model, agent, data, renderer, T, True, False, savepath = f'./saved_trajectories/trajectories_model_{i}.csv')

    states, actions, rewards, total_reward = run_policy(model, agent, data, renderer, T, random_initial_state=False, goal_state=3, camera_id="robot_cam", 
                                                        policy_path=str(policy_path))
    print("STAAAAAAAAAAAAAAAAAAAAAAATES")
    print(states[-10:])
    # plot states
    plot_states(states, len(states))
    # plot actions
    plot_actions(actions, len(actions))
    # plot costs
    # plot_rewards(agent, rewards, total_reward, len(total_reward[0]), show = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--policy_path",
        help="Path to the policy checkpoint .pth file",
    )
    args = parser.parse_args()
    main(args.policy_path)

# NOTE Currently best:
# td3_bc_quadruped_hill_hyper_search_batch_1024_actor_lr_0.0007_critic_lr_0.0007_actor_dims_1024_1024_critic_dims_1024_1024_steps_pr_epoch_2500_steps_pr_val_epoch_250 - 1000 epochs
# td3_bc_quadruped_hill_hyper_search_batch_1024_actor_lr_0.0007_critic_lr_0.0007_actor_dims_1024_1024_critic_dims_1024_1024_steps_pr_epoch_500_steps_pr_val_epoch_50 - 1000 epochs
# td3_bc_quadruped_hill_hyper_search_batch_1024_actor_lr_0.0007_critic_lr_0.0007_actor_dims_1024_1024_critic_dims_1024_1024_steps_pr_epoch_1000_steps_pr_val_epoch_100 - 1000 epochs
# td3_bc_quadruped_hill_hyper_search_batch_1024_actor_lr_0.0007_critic_lr_0.0007_actor_dims_1024_1024_critic_dims_1024_1024_steps_pr_epoch_100_steps_pr_val_epoch_10 - 1000 epochs

# TODO:
# BIGGER BATCH SIZE: pong typically performs well with 1000 in batch size (yikes) according to this: https://andyljones.com/posts/rl-debugging.html
# Retry with 3x512 networks: singularities are more easily avoided/overcome with bigger batches
# Learning rate shceduling

# Bigger batch sizes: (Steps pr epoch is ish dataset divided by batch size, and val steps per epoch is 1/10 of that. E.g. batch 8192: steps pr epoch 30 (23x xxx samples in data set)
#                       learning rate used: 0.0005, no learning rate scheduling, 1000 epochs.)
# 2048:
# 512x3: Moves, more chaotic movements, but able to cover some space, flings itself to cover more space sometimes
# 1024x2: Similar to 512x3, but more "dead spider" legs, arguably worse than 512x3
# 4096:
# 512x3:  Does some movements for the first moments, then stuck in singularity/doesn't move
# 1024x2: Pretty good! tries to walk, tips over, tries to walk when on its back, doesn't cover much ground: Check tensorboard, think more/further training can be done here

# 8192:
# 512x3: Best yet? Similar to 4096, 1024x2, but "stronger" movements, first smooth Train/critic/q_target curve, both actor and critic validation loss decreasing with 
#        training, very spiky loss curves still though, especially training loss
# 1024x2: Singularity :( get's stuck/doesn't move

# 12288: 
# 512x3: Not very good, more similar to 2048, 512x3 again, little less chaotic, but less prounounced walking patterns in the legs, also lays on back 
# 1024x2: Not very good, straigther legs than 512x3, less chaotic, but legs in wrong directions compared to each other as stuff.

# MOST PROMISING SO FAR:
# Batch 4096: 1024x2
# Batch 8192: 512x3
# TODO: Try further training here, more epochs, other steps_pr_epoch, learning rate, learning rate scheduling
# INVESTIGATE singularities in 4096: 512x3 and 8192: 1024x2


# NOTE: Smaller networks seem promising!! Double layers <= 512 tried. Will add extra reward at last obs over every trajectory in dataset now, and see if that helps.
# ALSO try to lower learning rate, to e.g. 0.0003 as they do in the paper, this should/could help with the singularties and mega spikes in the critic loss
