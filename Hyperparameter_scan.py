import argparse
import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils


def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )
    
    # next: set and sweep over hyperparameters
    generator.add_param(...) # set / sweep hp1
    generator.add_param(...) # set / sweep hp2
    generator.add_param(...) # set / sweep hp3
    ...
    
    return generator

def main(args):

    # make config generator
    generator = make_generator(
      config_file=args.config, # base config file from step 1
      script_file=args.script  # explained later in step 4
    )

    # generate jsons and script
    generator.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config",
        type=str,
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
    )

    args = parser.parse_args()
    main(args)