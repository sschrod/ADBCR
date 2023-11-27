import torch
import numpy as np
from adbcr.utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select method')
    parser.add_argument('--method', type=str, default="ADBCR")
    parser.add_argument('--num_rep', type=int, default=100)
    args = dict(vars(parser.parse_args()))


    ### Arrays to save the results
    results_pehe = np.zeros(args["num_rep"])
    results_ate = np.zeros(args["num_rep"])
    results_ate_train = np.zeros(args["num_rep"])
    results_pehe_train = np.zeros(args["num_rep"])

    ### Load Data
    Data_all_train = load_data('Data/ihdp_npci_1-1000.train.npz')
    Data_all_test = load_data('Data/ihdp_npci_1-1000.test.npz')

    ### Evaluate the saved models
    for i in range(0, args["num_rep"]):
        ### Load trial data
        Data_test = get_representation(Data_all_test, index=i)
        Data_train = get_representation(Data_all_train, index=i)
        X_test = Data_test["x"]
        X_train = Data_train["x"]
        Yf_test, Ycf_test = reformat_output(Data_test["yf"], Data_test["ycf"], Data_test["t"])
        Yf_train, Ycf_train = reformat_output(Data_train["yf"], Data_train["ycf"], Data_train["t"])

        ### Load trial model and predict results
        model = torch.load('saved_models/model_'+args["method"]+'_IHDP_t{}'.format(i))
        config = torch.load('saved_models/config_'+args["method"]+'_IHDP_t{}'.format(i))
        pred_test = model.predict_numpy(X_test)
        pred_train = model.predict_numpy(X_train)

        ### Predict PEHE and ATE
        results_pehe[i], results_ate[i] = analyse(pred_test,Yf_test,Ycf_test, Data_test)
        results_pehe_train[i], results_ate_train[i] = analyse(pred_train,Yf_train,Ycf_train, Data_train)
        print('\n Representation {}'.format(i+1))
        print('PEHE train: {}, eATE train: {}'.format(results_pehe_train[i], results_ate_train[i]))
        print('PEHE test: {}, eATE test: {},'.format(results_pehe[i], results_ate[i]))

    print('-----------------------')
    print('Mean pehe: {}, std pehe: {}'.format(np.mean(results_pehe),np.std(results_pehe)/np.sqrt(args["num_rep"])))
    print('Mean eATE: {}, std eATE: {}'.format(np.mean(results_ate),np.std(results_ate)/np.sqrt(args["num_rep"])))
    print('Mean pehe in sample: {}, std pehe in sample: {}'.format(np.mean(results_pehe_train),
                                                                   np.std(results_pehe_train)/np.sqrt(args["num_rep"])))
    print('Mean eATE in sample: {}, std eATE in sample: {}'.format(np.mean(results_ate_train),
                                                                   np.std(results_ate_train)/np.sqrt(args["num_rep"])))



