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
    # generator.add_param( # set / sweep batch size
    #     key="train.batch_size",
    #     name="batch",
    #     group=0,
    #     values=[4096, 8192] # [2048, 4096, 8192, 12288]
    # )

    # generator.add_param(
    #     key="experiment.epoch_every_n_steps",
    #     name="steps_pr_epoch",
    #     group=0,
    #     values=[60,30] # [120,60,30,20]
    # )

    # generator.add_param(
    #     key="experiment.validation_epoch_every_n_steps",
    #     name="steps_pr_val_epoch",
    #     group=0,
    #     values=[6,3] # [12,5,3,2]
    # )

    # generator.add_param(
    #     key="algo.actor.layer_dims",
    #     name="actor_dims",
    #     group=1,
    #     values=[[512,512],[300,400], [256,256]]
    # ) # set / sweep actor layer dims

    # generator.add_param(
    #     key="algo.critic.layer_dims",
    #     name="critic_dims",
    #     group=1,
    #     values=[[512,512],[300,400], [256,256]]
    # ) # set / sweep actor layer dims

    # generator.add_param(
    #     key="algo.optim_params.actor.learning_rate.initial",
    #     name="actor_lr",
    #     group=2,
    #     values=[0.0003, 0.0001]
    # ) # set / sweep actor learning rate

    # generator.add_param(
    #     key="algo.optim_params.critic.learning_rate.initial",
    #     name="critic_lr",
    #     group=2,
    #     values=[0.0003, 0.0001]
    # ) # set / sweep critic learning rate

    generator.add_param(
        key="algo.alpha",
        name="alpha",
        group=0,
        values=[2.5, 2.75, 3.0, 2.5]
    )

    generator.add_param(
        key="algo.optim_params.actor.learning_rate.epoch_schedule",
        name="actor_schedule",
        group=0,
        values=[[],[],[],[1000, 1500]]
    ) # set / sweep actor learning rate

    generator.add_param(
        key="algo.optim_params.critic.learning_rate.epoch_schedule",
        name="critic_schedule",
        group=0,
        values=[[],[],[],[1000, 1500]]
    ) # set / sweep critic learning rate


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