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
        obs_size = mpc_agent.run_planner(random_initial_state=False, random_goal_state=True, save_trajectory=True, savepath = f'./saved_trajectories/ep_{i}.json')
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
