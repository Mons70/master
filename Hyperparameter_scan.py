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
    generator.add_param( # set / sweep batch size
        key="train.batch_size",
        name="batch",
        group=0,
        values=[1024,2048]
    )

    generator.add_param(
        key="algo.optim_params.actor.learning_rate.initial",
        name="actor_lr",
        group=1,
        values=[0.0003, 0.0007]
    ) # set / sweep actor learning rate

    generator.add_param(
        key="algo.optim_params.critic.learning_rate.initial",
        name="critic_lr",
        group=1,
        values=[0.0003,0.0007]
    ) # set / sweep critic learning rate

    generator.add_param(
        key="algo.actor.layer_dims",
        name="actor_dims",
        group=2,
        values=[[512,512,512],[1024,1024]]
    ) # set / sweep actor layer dims

    generator.add_param(
        key="algo.critic.layer_dims",
        name="critic_dims",
        group=2,
        values=[[512,512,512],[1024,1024]]
    ) # set / sweep actor layer dims

    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="steps_pr_epoch",
        group=3,
        values=[100,500,1000,2500]
    )

    generator.add_param(
        key="experiment.validation_epoch_every_n_steps",
        name="steps_pr_val_epoch",
        group=3,
        values=[10,50,100,250]
    )
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