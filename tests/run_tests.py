# %%
import mujoco
import pathlib
import cv2
import json
import os
import argparse
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils
import matplotlib.pyplot as plt
import numpy as np
from mujoco_mpc import agent as agent_lib
from sklearn import preprocessing
from tqdm import tqdm
from glob import glob

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
    return action_min, action_max




class MUJOCO_POLICY_AGENT():
    def __init__(self, task_path: str = None, task_id: str = None, time_horizon: int = 1000, render: bool = False):

        model_path = (
            pathlib.Path(task_path)
                )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))

        # data
        self.data = mujoco.MjData(self.model)

        # agent
        self.agent = agent_lib.Agent(task_id=task_id, model=self.model)
        
        # renderer
        if render:
            # model.cam(data.cam("robot_cam").id).fovy[0] = 20
            self.renderer = mujoco.Renderer(self.model, 480, 640)
            # renderer.camera = mujoco.MjsCamera(model, data, "robot_cam")
            # renderer.update_scene(data, camera=cam)
        else:
            self.renderer = None

        # Time horizon
        self.T = time_horizon
        self.time_stop = None
        self.camera = None
        
        print("Timestep:", self.model.opt.timestep)

    def _render_video(frames, framerate, playback_speed, name):
        # Define the codec and create VideoWriter object
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
        out = cv2.VideoWriter(f"./{name}.mp4", fourcc, np.round(playback_speed*framerate), (width, height))
        assert out.isOpened()

        for frame in frames:
            out.write(frame)

        out.release()

    def _save_trajectories(self, filename,states, actions, rewards, total_reward, goal_state):
        data = {}
        data['states'] = states
        data['actions'] = actions
        data['rewards'] = rewards
        data['total_reward'] = total_reward
        data['goal_state'] = goal_state
        with open(filename, 'w') as f:
            json.dump(data, f)

    def _init_trajectories(self):
        #Set number of timesteps T
        # TODO: Implement actual number of timesteps T from time horizon / timestep variable

        #trajectories
        self.qpos = np.zeros((self.model.nq, self.T)) # model.nq = number of generalized coordinates in the model = dim(qpos)
        self.qvel = np.zeros((self.model.nv, self.T)) # model.nv = number of degrees of freedom in the model = dim(qvel)
        self.ctrl = np.zeros((self.model.nu, self.T)) # model.nu = number of actuators/controls = dim(ctrl)
        self.time = np.zeros(self.T) #number of timesteps in defined unit, unit defined in the model .xml file

    def _init_agent_cost_terms(self):
        #Set number of timesteps T
        # TODO: Implement actual number of timesteps T from time horizon / timestep variable

        # costs
        self.cost_total = np.zeros((1,self.T))
        self.cost_terms = np.zeros((len(self.agent.get_cost_term_values()), self.T))


    def _set_initial_state(self):
        # rollout
        mujoco.mj_resetData(self.model, self.data)
        # cache initial state
        self.qpos[:, 0] = self.data.qpos
        self.qvel[:, 0] = self.data.qvel
        self.time[0] = self.data.time

        print("Initital position: " + str(self.qpos[:,0]))
        print("Initial velocity: " + str(self.qvel[:,0]))


    def _set_random_initial_state(self, body: str):
        # rollout
        mujoco.mj_resetData(self.model, self.data)
        for i in range(len(self.data.qpos)):
            self.data.qpos[i] = (np.random.rand(1)*10) 
        
        # cache initial state
        self.qpos[:, 0] = self.data.qpos
        self.qvel[:, 0] = self.data.qvel
        self.time[0] = self.data.time
        
        print("Initital position: " + str(self.qpos[:,0]))
        print("Initial velocity: " + str(self.qvel[:,0]))
    
    def _set_random_goal_state(self):
        available_states = self.agent.get_all_modes()
        rand_idx = np.random.randint(1,len(available_states)-1)
        random_state = available_states[rand_idx]
        self.agent.set_mode(random_state)
        self.goal_pos = self.agent.model.key_mpos[rand_idx -1]
        self.goal_quat = self.agent.model.key_mquat[rand_idx -1]
        
        self.data.mocap_pos = self.goal_pos
        self.data.mocap_quat = self.goal_quat

        goal_dict_list = self.data.qpos.copy()
        # print("Initial state: ", goal_dict_list)
        # print(goal_pos)
        # print(goal_quat)
        mocap_goal = [*self.goal_pos, *self.goal_quat]
        # print("Mocap goal: ", mocap_goal)

        for i in range(7): #7 for the world + quat pos given in the mocap_pos
            goal_dict_list[i] = mocap_goal[i]
        
        self.goal_dict = {"pos": np.array(goal_dict_list)}

    def _set_goal_state(self, goal_state):
        available_states = self.agent.get_all_modes()
        self.agent.set_mode(available_states[goal_state])

        self.goal_pos = self.agent.model.key_mpos[goal_state -1]
        self.goal_quat = self.agent.model.key_mquat[goal_state -1]
        
        self.data.mocap_pos = self.goal_pos
        self.data.mocap_quat = self.goal_quat
        goal_dict_list = self.data.qpos.copy()
   
        mocap_goal = [*self.goal_pos, *self.goal_quat]

        for i in range(7): #7 first indxs for the world + quat pos given in the mocap_pos
            goal_dict_list[i] = mocap_goal[i]
        
        self.goal_dict = {"pos": np.array(goal_dict_list)}


    # def _get_action_scaling(self, directory: str):
    #     ep_paths = os.path.join(directory, "ep_*.json")
    #     self.action_min = 0
    #     self.action_max = 0
    #     for ep_file in sorted(glob(ep_paths)):
    #         with open(ep_file, 'r') as f:
    #             data_dictionary = json.load(f)
    #         action_values = np.array([ai['actions'] for ai in data_dictionary['actions']])
    #         curr_min = np.min(action_values)
    #         if curr_min < self.action_min:
    #             self.action_min = curr_min

    #         curr_max = np.max(action_values)
    #         if curr_max > self.action_max:
    #             self.action_max = curr_max    
    #     print("Action min: ", self.action_min)
    def _populate_action_scaling_values(self, action_min, action_max):
        self.action_min = action_min
        self.action_max = action_max


    def _set_action_scalings(self, actions):
        return np.round(preprocessing.minmax_scale(actions,feature_range=(self.action_min,self.action_max),axis=0),2)

    def set_camera_view(self, position: tuple = (0,0,0), elevation: int = -90, azimuth: int = 90, xml_cam_id: str = None):
        # Good tracking camera view for quadruped in xml format:
        # <camera name="robot_cam" pos="0 2 2" xyaxes="-1 0 0 0 -0.707 0.707" mode="trackcom"/>
        if xml_cam_id != None:
            self.camera = xml_cam_id
        else:    
            self.camera = mujoco.MjvCamera()
            self.camera.elevation = elevation
            self.camera.azimuth = azimuth
            self.camera.lookat = np.array(position)


    def run_policy(self, random_initial_state:bool = False, goal_state = None, camera_id=None, policy_path:str = None, disturbance:bool = False):
        #time horizon
        if policy_path != "mpc" and policy_path != None:
            try:
                render_video_name = policy_path.split("/")[-4:]
                render_video_name[-1] = render_video_name[-1].strip(".pth")
                render_video_name = '-'.join(render_video_name)
                print(render_video_name)

                # restore policy
                device = TorchUtils.get_torch_device(try_to_use_cuda=True)
                policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=policy_path, device=device, verbose=True)
                policy.start_episode()
                
                # action_min, action_max = self.get_action_scaling("/home/mons/dev/private/master/saved_trajectories/1000_quads")
                self.mpc = False
            except FileNotFoundError:
                print("Policy not found. Please check the path.")
                return
        elif policy_path == "mpc":
            self.mpc = True
        else:
            raise ValueError("Please provide a valid policy path or set policy to 'mpc'.")

        #trajectories
        self._init_trajectories()

        # costs
        self._init_agent_cost_terms()

        # get max and min action values from collected data to scale policy actions back to environment corrected actions
        print("Fetching max- and min-action values for scaling ...")


        if camera_id == None:
            camera = None
        else:
            camera = self.set_camera_view(xml_cam_id=camera_id)

        # If a renderer is specified, initialize array to store frames
        if self.renderer == None:
            render = False
        else:
            render =True
            frames = []

        if disturbance:
            terrain_id = self.model.geom_name2id("terrain")
            self.model.geom_friction = [0.05, 0.005, 0.0001]

        if random_initial_state:
            self._set_random_initial_state("Quadruped Terrain")
        else:
            self._set_initial_state()

        if goal_state != None:
            self._set_goal_state(goal_state)
        else:
            self._set_random_goal_state()
        
        if self.agent.get_mode() == 'Loop':
            self.goal_state = 'Loop'
        else:
            self.goal_state = int("".join([x if x.isdigit() else "" for x in list(self.agent.get_mode())]))

        print("Goal state:", self.goal_state)

        self.body_height = []
        
        # simulate
        for t in tqdm(range(self.T)):
            # if t % 100 == 0:
            #     print("\rt = ", t)

            # set planner state
            self.agent.set_state(
                time=self.data.time,
                qpos=self.data.qpos,
                qvel=self.data.qvel,
                act=self.data.act,
                mocap_pos=self.data.mocap_pos,
                mocap_quat=self.data.mocap_quat,
                userdata=self.data.userdata,
            )

            # The residual 'Stand' is the difference in height between the body and the average height of the feet minus the specified height goal, so adding the height goal here gives the height of the body relative to the average height of the legs
            # Default height goal for the quadruped task is 0.25(m)
            self.body_height.append(float(self.agent.get_residuals()['Stand'][0]) + float(self.agent.get_task_parameters()["Height Goal"]))

            # Fetch current state as an observation
            obs = {"pos": np.array(self.data.qpos),"vel": np.array(self.data.qvel)}

            # run planner for num_steps
            num_steps = 10
            for _ in range(num_steps):
                self.agent.planner_step()

            # set ctrl from agent policy
            if self.mpc:
                self.data.ctrl = self.agent.get_action()
            else:
                self.data.ctrl = self._set_action_scalings(policy(ob=obs, goal=self.goal_dict))
            self.ctrl[:, t] = self.data.ctrl

            # get costs
            self.cost_total[0][t] = self.agent.get_total_cost()
            for i, c in enumerate(self.agent.get_cost_term_values().items()):
                self.cost_terms[i, t] = c[1]
                self.cost_term_labels = c[0]

            # step
            mujoco.mj_step(self.model, self.data)

            # cache
            if t < self.T-1:
                self.qpos[:, t + 1] = self.data.qpos
                self.qvel[:, t + 1] = self.data.qvel
                self.time[t + 1] = self.data.time

            # If a renderer was specified, render and save frames
            if render:
                if self.camera == None:
                    self.renderer.update_scene(self.data)
                else:
                    self.renderer.update_scene(self.data, camera=self.camera)
                    frames.append(self.renderer.render())
            
            if self.cost_total[0][t] < 0.25:
                self.qpos = self.qpos[:, :t+1]
                self.qvel = self.qvel[:, :t+1]
                self.ctrl = self.ctrl[:, :t+1]
                self.cost_terms = self.cost_terms[:,:t+1]
                self.cost_total = self.cost_total[:,:t+1]
                break

        # close agent after finished run
        self.agent.close()

        # If a renderer was specified, render video from frames and write to file
        if render:
            # display video
            FPS = 1.0 / self.model.opt.timestep #timestep is defined as an option in the model .xml file
            SLOWDOWN = 0.5
            self._render_video(frames, FPS, SLOWDOWN)
            self.renderer.close()

        #format trajectories to be compatible for robosuite hdf5 conversion:
        self.states = np.column_stack((np.transpose(self.qpos), np.transpose(self.qvel)))
        print("States shape: " + str(self.states.shape))

        self.actions = []
        for t in range(len(self.states)):
            self.actions.append({'actions': self.ctrl[:,t].tolist()})
        self.actions = np.array(self.actions)

        print("Actions shape: " + str(self.actions.shape))

        self.rewards = 1/(1 + self.cost_terms)
        self.total_reward = 1/(1 + self.cost_total)
        # self.total_reward = self.cost_total * -1

        # if save_trajectory:
        #     self._save_trajectories(savepath,self.states.tolist(),self.actions.tolist(),self.rewards.tolist(),self.total_reward.tolist(), self.goal_state)
        # print(self.qpos.shape[0])
        
        return self.states.tolist(), self.actions.tolist(), self.rewards.tolist(), self.total_reward.tolist(), self.ctrl.tolist(), self.body_height, self.goal_state
    
def plot_states(states, time_horizon, show:bool = False):
    #State shape = timesteps x dim(state space)
    fig = plt.figure()
    time = np.arange(0, time_horizon)
    states[:,2] = states[:,2] % 2*np.pi
    plt.plot(time, states)
    plt.legend()
    plt.xlabel("Timesteps")
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
    plt.xlabel("Timesteps")
    plt.ylabel("Control")
    if show:
       plt.show()

def plot_rewards(total_reward, time_horizon, show:bool = False):
    fig = plt.figure()
    time = np.arange(0, time_horizon)
    # for i, c in enumerate(agent.get_cost_term_values().items()):
    #     plt.plot(time[:], rewards[i, :], label=c[0])

    plt.plot(time[:], total_reward[0,:], label="Total (weighted)", color="black")
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    if show:
        plt.show()


def main(rl_policy_path: str, bc_policy_path: str):
    T = 100
    policies = ['mpc', bc_policy_path, rl_policy_path]
    policy_names = ['MPC', 'Behavioral cloning', 'Offline RL']
    goals = [1,2] #, 8, 11]#, 14, 17]
    runs_pr_goal = 2
    policy_trajectories = {}
    for current_policy, policy_name in zip(policies, policy_names):
        if current_policy != "mpc":
            action_min, action_max = get_action_scaling("/home/mons/dev/private/master/saved_trajectories/1000_quads")
        print(f'Running {policy_name}:')
        goal_trajectories = {}
        
        for goal in goals:
            print(f'Goal: {goal}')
            goal_trajectories[goal] = []

            for i in range(runs_pr_goal):
                mujoco_agent = MUJOCO_POLICY_AGENT(task_path="/home/mons/dev/private/master/mujoco_mpc/build/mjpc/tasks/quadruped/task_hill.xml", task_id="Quadruped Hill", 
                                        time_horizon= T, render=False)
                if current_policy != "mpc":
                    mujoco_agent._populate_action_scaling_values(action_min, action_max)
                print(f'\rRun {i+1}/{runs_pr_goal}')
                # Set the agents' weights
                # agent.set_cost_weights({"Velocity": 0.15})
                print("Cost weights:", mujoco_agent.agent.get_cost_weights())

                # Set the task parameters, e.g. goal
                # agent.set_task_parameter("Goal", -1.0)
                print("Parameters:", mujoco_agent.agent.get_task_parameters())

                # Run planner
                #qpos, qvel, ctrl, cost_terms, cost_total = run_planner(model, agent, data, renderer, T, True, False, savepath = f'./saved_trajectories/trajectories_model_{i}.csv')

                states, actions, rewards, total_reward, ctrl, body_height, goal_state = mujoco_agent.run_policy(random_initial_state=False, goal_state=goal, camera_id="robot_cam", 
                                                                                                        policy_path=str(current_policy), disturbance=True)
                goal_trajectories[goal].append({'states': states, 'actions': actions, 'rewards': rewards, 'total_reward': total_reward,
                                                'ctrl': ctrl, 'body_height': body_height, 'goal_state': goal_state})
        policy_trajectories[policy_name] = goal_trajectories
    
    print(policy_trajectories)
    with open('./tests/test_data.json', 'w') as f:
        json.dump(policy_trajectories, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--rl_policy",
        help="Path to the offline rl policy .pth file",
    )

    parser.add_argument(
        "--bc_policy",
        help="Path to the behaviour cloning policy .pth file",
    )
    args = parser.parse_args()
    main(args.rl_policy, args.bc_policy)