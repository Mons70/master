import sys
import os
import argparse
from mjpc_agent_wrapper import *

def main(args):
    T = args.max_demo_duration
    num_demos = args.num_demonstrations
    terminal_width = os.get_terminal_size().columns 

    for i in range(5000, 5000 + num_demos):
        print('#' * terminal_width)
        demo_string = f'Calculating demo {i+1} out of {num_demos}...'
        print(f'{demo_string}')
        print('-' * len(demo_string))
        # Initialize mpc agent
        mpc_agent = MJPC_AGENT(task_path="/home/mons/dev/private/master/mujoco_mpc/build/mjpc/tasks/quadruped/task_hill.xml", task_id="Quadruped Hill", 
                            time_horizon= T, render=True)

        # Get the task parameters, e.g. goal
        print("Parameters:", mpc_agent.agent.get_task_parameters())

        mpc_agent.set_camera_view(xml_cam_id="robot_cam")

        # Run MPC planner, return qpos observation size, qvel is the remaining half of a state
        obs_size = mpc_agent.run_planner(random_initial_state=False, goal_state=None, save_trajectory=True, savepath = f'./saved_trajectories/ep_{i}.json')
        print('#' * terminal_width)
        # # plot states, don't show
        # mpc_agent.plot_states()
        # # plot actions, don't show
        # mpc_agent.plot_actions()
        # # plot costs
        # mpc_agent.plot_rewards(show=True)
    # Exit using obs size as exit code to return it for use in dataset generation
    print(obs_size)
    sys.exit(obs_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max_demo_duration",
        type=int,
        help="Max amount of time in milliseconds a single MPC demonstration can be before terminating",
    )

    parser.add_argument(
        "--num_demonstrations",
        type=int,
        help="Number of MPC demonstrations/trajectories that should be included in the dataset",
    )

    args = parser.parse_args()
    sys.stdout.write(str(main(args)))

    #             <!-- New goals -->
    # <!--Stage 2-->
    # <!-- <key mpos="1.93695 -0.13810 0.45" mquat="0.266032 0 0 -0.963964" /> 
    # <key mpos="-4.98318 -3.95614 0.25" mquat="0.996152 0 0 -0.087639" />
    # <key mpos="0.79131 -4.68012 0.25" mquat="0.673138 0 0 0.739517" />
    # <key mpos="-0.02520 3.68893 0.9" mquat="0.889288 0 0 -0.457348" />
    # <key mpos="3.74955 -1.58986 0.27" mquat="0.762615 0 0 0.646853" />
    # <key mpos="4.48456 2.75455 0.30" mquat="0 -0.4 0.71258 -0.701591" />
    # <key mpos="4.58215 -3.42454 0.25" mquat="0.607969 0 0 0.793961" />
    # <key mpos="2.43022 4.54289 0.25" mquat="0.574809 0 0 -0.818287" />
    # <key mpos="1.12297 1.19466 0.25" mquat="0.0176800 0 0 -0.984247" />
    # <key mpos="-3.49162 -0.55558 0.25" mquat="0.849291 0 0 0.527925" />
    # <key mpos="-0.94072 4.81276 0.25" mquat="0 0 0 -0.402998" />
    # <key mpos="2.14062 0.77786 0.92" mquat="0.125345 0 0 -0.992113" />
    # <key mpos="-1.30249 0.09238 0.47" mquat="0.949181 0 0 0.314732" />
    # <key mpos="2.50175 3.22686 0.25" mquat="0.799433 0 0 -0.600756" />
    # <key mpos="4.69978 -4.66257 0.25" mquat="0.277517 0 0 0.960721" />
    # <key mpos="1.22319 -2.47120 0.25" mquat="0.288297 0 0 0.957541" />
    # <key mpos="-3.12424 0.40762 0.55" mquat="0.996404 0 0 -0.084734" />
    # <key mpos="4.77988 -0.56363 0.25" mquat="0.285845 0 0 0.958276" />
    # <key mpos="-3.55986 4.69313 0.25" mquat="0.835779 0 0 -0.549066" />
    # <key mpos="0.28912 -4.63385 0.25" mquat="0.761761 0 0 0.647859" />
    # <key mpos="1.65215 4.04531 0.33" mquat="0.136072 0 0 -0.990699" />
    # <key mpos="-0.87880 3.33669 0.25" mquat="0.917958 0 0 -0.396678" /> -->
    # <!--Stage 23-->
