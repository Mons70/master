import argparse
from mjpc_agent_wrapper import *
from tqdm import tqdm

def main(args):
    T = args.max_demo_duration
    num_demos = args.num_demonstrations

    for i in tqdm(range(0, num_demos)):
        # Initialize mpc agent
        mpc_agent = MJPC_AGENT(task_path="/home/mons/dev/private/master/mujoco_mpc/build/mjpc/tasks/quadruped/task_flat.xml", task_id="Quadruped Flat", 
                            time_horizon= T, render=True)

        # Get the task parameters, e.g. goal
        print("Parameters:", mpc_agent.agent.get_task_parameters())

        mpc_agent.set_camera_view(xml_cam_id="robot_cam")

        # Run MPC planner
        mpc_agent.run_planner(random_initial_state=False, save_trajectory=True, savepath = f'./saved_trajectories/ep_{i}.json')

        # # plot states, don't show
        # mpc_agent.plot_states()
        # # plot actions, don't show
        # mpc_agent.plot_actions()
        # # plot costs
        # mpc_agent.plot_rewards(show=True)

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
    main(args)
