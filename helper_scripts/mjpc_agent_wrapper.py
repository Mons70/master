# %%
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import pathlib
import cv2
import csv
import json
from mujoco_mpc import agent as agent_lib
from mujoco_mpc import mjpc_parameters
from mujoco_mpc.proto import agent_pb2
from typing import Mapping
from tqdm import tqdm


class MJPC_AGENT():
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

    def _render_video(self, frames, framerate, playback_speed):
        # Define the codec and create VideoWriter object
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
        out = cv2.VideoWriter("mpc_demo.mp4", fourcc, np.round(playback_speed*framerate), (width, height))
        assert out.isOpened()

        for frame in frames:
            out.write(frame)

        out.release()

    def _save_trajectories(self, filename,states, actions, rewards,total_reward):
        data = {}
        data['states'] = states
        data['actions'] = actions
        data['rewards'] = rewards
        data['total_reward'] = total_reward
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


    def _init_agent_cost_terms(self):
        #Set number of timesteps T
        # TODO: Implement actual number of timesteps T from time horizon / timestep variable

        # costs
        self.cost_total = np.zeros((1,self.T))
        self.cost_terms = np.zeros((len(self.agent.get_cost_term_values()), self.T))


    def run_planner(self, random_initial_state:bool = False, random_goal_state:bool = False, save_trajectory:bool = False, savepath:str = "../saved_trajectories/trajectory.json"):
        #time horizon

        #trajectories
        self._init_trajectories()

        # costs
        self._init_agent_cost_terms()

        # If a renderer is specified, initialize array to store frames
        if self.renderer == None:
            render = False
        else:
            render =True
            frames = []

        if random_initial_state:
            self._set_random_initial_state("Quadruped Terrain")
        else:
            self._set_initial_state()

        if random_goal_state:
            self._set_random_goal_state()
            print("Goal state:", self.agent.get_mode())

        # simulate
        for t in tqdm(range(self.T-1)):
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

            # run planner for num_steps
            num_steps = 10
            for _ in range(num_steps):
                self.agent.planner_step()

            # set ctrl from agent policy
            self.data.ctrl = self.agent.get_action()
            self.ctrl[:, t] = self.data.ctrl

            # get costs
            self.cost_total[0][t] = self.agent.get_total_cost()
            for i, c in enumerate(self.agent.get_cost_term_values().items()):
                self.cost_terms[i, t] = c[1]
                self.cost_term_labels = c[0]

            # step
            mujoco.mj_step(self.model, self.data)

            # cache
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

        self.rewards = self.cost_terms * -1
        self.total_reward = self.cost_total * -1

        if save_trajectory:
            self._save_trajectories(savepath,self.states.tolist(),self.actions.tolist(),self.rewards.tolist(),self.total_reward.tolist())
        print(self.qpos.shape[0])
        
        return self.qpos.shape[0]

    
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
    
    def plot_states(self, show:bool = False):
        #State shape = timesteps x dim(state space)
        fig = plt.figure()
        time = np.arange(0, len(self.states))
        plt.plot(time, self.states)
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("State values")
        if show:
            plt.show()


    def plot_actions(self, show:bool = False):
        fig = plt.figure()
        time = np.arange(0, len(self.actions))
        actions_list = []
        for dict in self.actions:
            actions_list.append(*dict.values())
        plt.plot(time, actions_list)
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Control")
        if show:
            plt.show()

    def plot_rewards(self, show:bool = False):
        fig = plt.figure()
        time = np.arange(0, len(self.rewards[0]))
        agent = self.agent
        for i,c in zip(range(len(self.rewards)), self.cost_term_labels):
            plt.plot(time[:], self.rewards[i, :], label=c)

        plt.plot(time[:], self.total_reward[0,:], label="Total (weighted)", color="black")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Costs")
        if show:
            plt.show()
