from ray.tune import Analysis
from adbcr.ADBCR_base import ADBCR
from adbcr.DANNCR_base import DANNCR
import torch
import os
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split


def get_best_model(path_to_experiment="./ray_results/test_hydra"):
    """Returns the best model and used config obtained by a Ray[Tune] trial"""
    analysis = Analysis(path_to_experiment, default_metric="val_loss", default_mode="min")
    best_config = analysis.get_best_config()
    best_config["dataframe"] = analysis.dataframe().values
    best_config["dataframe_names"] = analysis.dataframe().columns
    best_checkpoint_dir = analysis.get_best_checkpoint(analysis.get_best_logdir())

    best_net = ADBCR(in_features=best_config["num_covariates"], num_nodes_shared=best_config["shared_layer"],num_nodes_indiv=best_config["individual_layer"],
                     batch_norm=best_config["batch_norm"], out_features=1, dropout=best_config["dropout"],n_treatments=best_config["number_treatments"])

    model_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"), map_location=torch.device('cpu'))

    best_net.load_state_dict(model_state)
    return best_net, best_config

def get_best_DANNCR_model(path_to_experiment="./ray_results/test_hydra"):
    """Returns the best model and used config obtained by a Ray[Tune] trial"""
    analysis = Analysis(path_to_experiment, default_metric="val_loss", default_mode="min")
    best_config = analysis.get_best_config()
    best_config["dataframe"] = analysis.dataframe().values
    best_config["dataframe_names"] = analysis.dataframe().columns
    best_checkpoint_dir = analysis.get_best_checkpoint(analysis.get_best_logdir())

    best_net = DANNCR(in_features=best_config["num_covariates"], num_nodes_shared=best_config["shared_layer"],num_nodes_indiv=best_config["individual_layer"], num_nodes_dc=best_config["dc_layer"],
                     batch_norm=best_config["batch_norm"], out_features=1, dropout=best_config["dropout"],n_treatments=best_config["number_treatments"])

    model_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"), map_location=torch.device('cpu'))

    best_net.load_state_dict(model_state)
    return best_net, best_config

def load_data(datapath):
    """ Load dataset (from https://github.com/clinicalml/cfrnet/blob/master/cfr/loader.py)"""
    arr = np.load(datapath)
    xs = arr['x']

    HAVE_TRUTH = False
    SPARSE = False

    if len(xs.shape)==1:
        SPARSE = True

    ts = arr['t']
    yfs = arr['yf']
    try:
        es = arr['e']
    except:
        es = None
    try:
        ate = np.mean(arr['ate'])
    except:
        ate = None
    try:
        ymul = arr['ymul'][0,0]
        yadd = arr['yadd'][0,0]
    except:
        ymul = 1
        yadd = 0
    try:
        ycfs = arr['ycf']
        mu0s = arr['mu0']
        mu1s = arr['mu1']
        HAVE_TRUTH = True
    except:
        ycfs = None; mu0s = None; mu1s = None

    data = {'x':xs, 't':ts, 'e':es, 'yf':yfs, 'ycf':ycfs,'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul,'YADD': yadd, 'ATE': ate.tolist(), 'HAVE_TRUTH': HAVE_TRUTH,'SPARSE': SPARSE}

    return data

def get_representation(Data, index=0):
    """Get a single representation for IHDP"""
    x = Data["x"][:, :, index]
    t = Data["t"][:, index]
    yf = Data["yf"][:, index]
    if Data["HAVE_TRUTH"]:
        ycf = Data["ycf"][:, index]
        mu0 = Data["mu0"][:, index]
        mu1 = Data["mu1"][:, index]
    else:
        ycf = None
        mu0 = None
        mu1 = None

    try:
        es = Data['e'][:,index]
    except:
        es = None
    return {'x': x, 't': t, "yf": yf, "ycf": ycf, "mu0": mu0, "mu1": mu1, 'e': es}


def reformat_output(yf,ycf,t):
    """reformat independent treatment and outcome variables into array[num_samples,num_treatments]"""
    Y, Y_cf = np.zeros((len(t), 2)), np.zeros((len(t), 2))
    Y[:], Y_cf[:] = np.nan, np.nan
    t=t.astype('int')
    for i, t_i in enumerate(t):
        Y[i, t_i] = yf[i]
        if ycf is not None:
            Y_cf[i, 1-t_i] = ycf[i]
    return Y,Y_cf


def load_sparse(fname):
    """ Load sparse data set (from https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py)"""
    E = np.loadtxt(open(fname,"rb"),delimiter=",")
    H = E[0,:]
    n = int(H[0])
    d = int(H[1])
    E = E[1:,:]
    S = coo_matrix((E[:,2],(E[:,0]-1,E[:,1]-1)),shape=(n,d))
    S = S.todense()

    return np.array(S)


def load_News(fname='Data/topic_doc_mean_n5000_k3477_seed_1.csv'):
    """Load the first realization of the NEWS dataset"""
    data_in = np.loadtxt(open(fname + '.y', "rb"), delimiter=",")
    x = load_sparse(fname + '.x')
    ind_train, ind_test = train_test_split(np.arange(0,x.shape[0],1),test_size=0.1,random_state=42,shuffle=True)
    data = {'x': x[ind_train,:], 't' : data_in[ind_train,0:1], 'yf' : data_in[ind_train,1:2], 'ycf' : data_in[ind_train,2:3],'mu0':data_in[ind_train,3:4],'mu1': data_in[ind_train,4:5]}
    data_test = {'x': x[ind_test,:], 't' : data_in[ind_test,0:1], 'yf' : data_in[ind_test,1:2], 'ycf' : data_in[ind_test,2:3],'mu0':data_in[ind_test,3:4],'mu1': data_in[ind_test,4:5]}
    return data, data_test


def analyse(pred_test, Yf_test, Ycf_test, Data_test):
    """analyse the predictions and return the PEHE and ATE"""
    f_mask, cf_mask = ~np.isnan(Yf_test), np.isnan(Yf_test)

    rmse_fact = np.sqrt(np.mean(np.square((pred_test - Yf_test)[f_mask])))
    rmse_cfact = np.sqrt(np.mean(np.square((pred_test - Ycf_test)[cf_mask])))

    eff = Data_test['mu1'] - Data_test['mu0']
    eff_pred = pred_test[:, 1] - pred_test[:, 0]

    pehe = np.sqrt(np.mean(np.square(eff_pred - eff)))

    ate_pred = np.mean(eff_pred)
    bias_ate = np.abs(ate_pred - np.mean(eff))
    return pehe, bias_ate

def analyse_NEWS(pred_test, Yf_test, Ycf_test, Data_test):
    """analyse the predictions and return the PEHE and ATE"""
    f_mask, cf_mask = ~np.isnan(Yf_test), np.isnan(Yf_test)

    rmse_fact = np.sqrt(np.mean(np.square((pred_test - Yf_test)[f_mask])))
    rmse_cfact = np.sqrt(np.mean(np.square((pred_test - Ycf_test)[cf_mask])))

    eff = (Data_test['mu1'] - Data_test['mu0']).flatten()
    eff_pred = pred_test[:, 1] - pred_test[:, 0]

    pehe = np.sqrt(np.mean(np.square(eff_pred - eff)))

    ate_pred = np.mean(eff_pred)
    bias_ate = np.abs(ate_pred - np.mean(eff))
    return pehe, bias_ate
