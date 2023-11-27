from adbcr.utils import load_data, get_representation, reformat_output
from adbcr.ADBCR_base import *
from adbcr.DANNCR_base import *
import numpy as np
import torch
from ray import tune
import ray
from ray.tune.schedulers import ASHAScheduler
from functools import  partial
import shutil
from adbcr.utils import get_best_model, get_best_DANNCR_model
import argparse

### Select Number of Representations (NUM_TRIALS) and METHOD (ADBCR, UADBCR, DANNCR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select method')
    parser.add_argument('--method', type=str, default="ADBCR")
    parser.add_argument('--num_rep', type=int, default=100)
    args = dict(vars(parser.parse_args()))

    if not os.path.isfile('Data/ihdp_npci_1-1000.train.npz'):
        print("Download the IHDP-1000 Data from https://www.fredjo.com/")
    Data_all = load_data('Data/ihdp_npci_1-1000.train.npz')
    if args["method"] == "UADBCR":
        Data_all_test = load_data('Data/ihdp_npci_1-1000.test.npz')

    for i in range(0, args["num_rep"]):
        config = {
            "Method": args["method"],
            "trial_name": 'IHDP_t{}'.format(i),                                 # name of your trial
            "result_dir": './ray_results',                                      # will be created to store results
            "val_set_fraction": 0.3,                                            # Size of the validation Set
            "num_covariates": 25,                                               # Number of covariates
            "shared_layer": tune.grid_search([[50,50], [20, 20], [10,10]]),     # Layer of shared representation
            "individual_layer": tune.grid_search([[50,50], [20,20], [10]]),     # Layer of outcome heads
            "lr": tune.loguniform(1e-5, 1e-2),                                  # Learning rate
            "dropout": tune.choice([0.1, 0.3, 0.5]),                            # Dropout rate
            "weight_decay": tune.choice([1, 0.1, 0.01, 0.001]),                 # Weight decay
            "batch_size": tune.choice([100, 250, 500]),                         # Batch size
            "number_adversarial_steps": tune.grid_search([1, 2, 3]),            # Num adversarial steps
            "epochs": 5000,                                                     # Max number of epochs
            "grace_period": 100,                                                # Early stopping
            "gpus_per_trial": 0.19,                                             # Num GPUs (here 5 trials per gpu)
            "cpus_per_trial": 6,                                                # CPUs used per trial
            "num_samples": 30,                                                  # Num trials per grid search element
            "number_treatments": 2,                                             # Number of treatments (tested for 2)
            "batch_norm": False,                                                # Use batch norm layers
            "seed": 42,                                                         # random seed
            "data_path": '/mnt/Data/IHDP_DATA'                                  # Data path (absolute)
        }

        Data=get_representation(Data_all, index=i)
        Yf, _ = reformat_output(Data["yf"],Data["ycf"],Data["t"])

        ray.init(object_store_memory=100000000)
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=10000,
            grace_period=config["grace_period"],
            reduction_factor=2)

        if args["method"] == "ADBCR":
            torch.save([Data["x"], Yf], config["data_path"])
            result = tune.run(
                partial(fit_ADBCR),
                name=config["Method"] + '_' + config["trial_name"],
                resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
                config=config,
                num_samples=config["num_samples"],
                scheduler=scheduler,
                checkpoint_at_end=False,
                local_dir=config["result_dir"])
            ray.shutdown()

            ### Get best model of trial
            model, config = get_best_model('./ray_results/' + config["Method"] + '_' + config["trial_name"])
            torch.save(model, 'saved_models/model_' + config["Method"] + '_' + config["trial_name"])
            torch.save(config, 'saved_models/config_' + config["Method"] + '_' + config["trial_name"])

            ### Delete temporary trial data
            shutil.rmtree('./ray_results')

        elif args["method"] == "UADBCR":
            Data_test = get_representation(Data_all_test, index=i)
            torch.save([Data["x"], Yf, Data_test["x"]], config["data_path"])

            result = tune.run(
                partial(fit_ADBCR),
                name=config["Method"] + '_' + config["trial_name"],
                resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
                config=config,
                num_samples=config["num_samples"],
                scheduler=scheduler,
                checkpoint_at_end=False,
                local_dir=config["result_dir"])
            ray.shutdown()

            ### Get best model of trial
            model, config = get_best_model('./ray_results/' + config["Method"] + '_' + config["trial_name"])
            torch.save(model, 'saved_models/model_' + config["Method"] + '_' + config["trial_name"])
            torch.save(config, 'saved_models/config_' + config["Method"] + '_' + config["trial_name"])

            ### Delete temporary trial data
            shutil.rmtree('./ray_results')

        elif args["method"] == "DANNCR":
            torch.save([Data["x"], Yf], config["data_path"])
            config["number_adversarial_steps"] = tune.grid_search([1]) # Not needed for DANNCR
            config["dc_layer"] = tune.grid_search([[50, 50], [20, 20], [10]])

            result = tune.run(
                partial(fit_DANNCR),
                name=config["Method"] + '_' + config["trial_name"],
                resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
                config=config,
                num_samples=config["num_samples"],
                scheduler=scheduler,
                checkpoint_at_end=False,
                local_dir=config["result_dir"])
            ray.shutdown()

            ### Get best model of trial
            model, config = get_best_DANNCR_model('./ray_results/' + config["Method"] + '_' + config["trial_name"])
            torch.save(model, 'saved_models/model_' + config["Method"] + '_' + config["trial_name"])
            torch.save(config, 'saved_models/config_' + config["Method"] + '_' + config["trial_name"])

            ### Delete temporary trial data
            shutil.rmtree('./ray_results')

        else:
            print("METHOD: "+args["method"]+" not implemented yet!" )

