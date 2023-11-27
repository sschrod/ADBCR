import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
from ray import tune
import os
import copy
from adbcr.Network_base import network_block


class ADBCR(nn.Module):
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, out_features,
                 n_treatments=2, batch_norm=False, dropout=None, random_seed=42):
        super().__init__()
        torch.manual_seed(random_seed)
        self.shared_net = network_block(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm=batch_norm, dropout=dropout, activation=nn.ELU, output_activation=nn.ELU, output_batch_n=False,
            output_dropout=True)

        self.first_rep_net = torch.nn.ModuleList()
        for i in range(n_treatments):
            torch.manual_seed((random_seed+i) * 2)
            net = network_block(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout, output_bias=True, activation=nn.ELU)
            self.first_rep_net.append(net)

        self.second_rep_net = torch.nn.ModuleList()
        for i in range(n_treatments):
            torch.manual_seed(random_seed + i)
            net = network_block(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout, output_bias=True,activation=nn.ELU)
            self.second_rep_net.append(net)

        torch.manual_seed(random_seed)

    def forward(self, input: Tensor):
        if torch.cuda.is_available():
            device = input.get_device()
            y0 = torch.zeros((input.size()[0], len(self.first_rep_net)), device=device)
            y1 = torch.zeros((input.size()[0], len(self.first_rep_net)), device=device)
        else:
            y0 = torch.zeros((input.size()[0], len(self.first_rep_net)))
            y1 = torch.zeros((input.size()[0], len(self.first_rep_net)))
        out = self.shared_net(input)

        for i, net in enumerate(self.first_rep_net):
            out_head = net(out)
            y0[:, i] = out_head.flatten()

        for i, net in enumerate(self.second_rep_net):
            out_head = net(out)
            y1[:, i] = out_head.flatten()

        return y0, y1, out

    def predict(self, X):
        self.eval()
        pred = self(X)
        self.train()
        return pred[0], pred[1], pred[2]

    def predict_numpy(self, X):
        self.eval()
        X = torch.Tensor(X)
        out = self(X)
        self.train()
        out_mean = np.mean([out[0].detach().numpy(), out[1].detach().numpy()],axis=0)
        return out_mean


class MSE_inore_NAN(torch.nn.Module):
    """Factual MSE loss."""
    def __init__(self, ):
        super().__init__()

    def forward(self, y_pred: Tensor, y: Tensor) -> Tensor:
        mse = nn.MSELoss()
        mask = ~torch.isnan(y)
        loss = mse(y_pred[mask], y[mask])
        return loss


class CFMCDA_loss(torch.nn.Module):
    """Counterfactual Discrepancy loss."""
    def __init__(self):
        super().__init__()

    def forward(self, out1: Tensor, out2: Tensor, y: Tensor):
        loss_fkt = nn.L1Loss()
        loss = 0
        for i in range(out1.shape[1]):
            mask = torch.isnan(y[:,i])
            sample_weight = torch.sum(mask) / mask.shape[0]
            loss += sample_weight*loss_fkt(out1[mask, i], out2[mask, i])
        return loss


def fit_ADBCR(config):

    load_data = torch.load(config["data_path"])

    net = ADBCR(in_features=load_data[0].shape[1], num_nodes_shared=config["shared_layer"],
                num_nodes_indiv=config["individual_layer"], out_features=1, batch_norm=config["batch_norm"],
                dropout=config["dropout"], n_treatments=config["number_treatments"], random_seed=config["seed"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device)
    best_net = copy.deepcopy(net.state_dict())

    optimizer_all = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    optimizer_shared = torch.optim.Adam(net.shared_net.parameters(), lr=config["lr"],
                                        weight_decay=config["weight_decay"])
    optimizer_first_rep_net = torch.optim.Adam(net.first_rep_net.parameters(), lr=config["lr"],
                                               weight_decay=config["weight_decay"])
    optimizer_second_rep_net = torch.optim.Adam(net.second_rep_net.parameters(), lr=config["lr"],
                                                weight_decay=config["weight_decay"])

    if len(load_data) == 2:
        print("Run ADBCR")
        X_train, Y_train = load_data
        X_train = torch.Tensor(X_train).to(device)
        Y_train = torch.Tensor(Y_train).to(device)
        data = TensorDataset(X_train, Y_train)
        test_abs = int(len(X_train) * (1 - config["val_set_fraction"]))
        train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs], generator=torch.Generator().manual_seed(config["seed"]))
    elif len(load_data) == 3:
        print("Run UADBCR")
        X_train, Y_train, X_test = load_data
        X_train = torch.Tensor(X_train).to(device)
        Y_train = torch.Tensor(Y_train).to(device)
        X_test = torch.Tensor(X_test).to(device)
        data = TensorDataset(X_train, Y_train)
        test_abs = int(len(X_train) * (1 - config["val_set_fraction"]))
        train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs],
                                                generator=torch.Generator().manual_seed(config["seed"]))
        train_subset = ConcatDataset([train_subset, TensorDataset(X_test, torch.full([X_test.shape[0], Y_train.shape[1]], torch.nan).to(device))])
    else:
        print("Dataset is not saved/loaded correctly.")
        return

    trainloader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    valloader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=True)

    best_val_loss = np.Inf
    loss_fkt = MSE_inore_NAN()
    unlabelled_loss = CFMCDA_loss()
    early_stopping_count = 0
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        epoch_steps = 0
        train_loss = 0.0
        loss_adv=0.0
        for i, data in enumerate(trainloader, 0):
            x, y = data

            # Step A
            optimizer_all.zero_grad()
            y_pred = net(x)
            lossA = (loss_fkt(y_pred[0], y) + loss_fkt(y_pred[1], y))
            lossA.backward()
            optimizer_all.step()

            # Step B
            optimizer_first_rep_net.zero_grad()
            optimizer_second_rep_net.zero_grad()
            y_pred = net(x)
            lossB = loss_fkt(y_pred[0], y) + loss_fkt(y_pred[1], y) - unlabelled_loss(y_pred[0], y_pred[1],y)
            lossB.backward()
            optimizer_first_rep_net.step()
            optimizer_second_rep_net.step()

            for i in range(config["number_adversarial_steps"]):
                # Step C
                optimizer_shared.zero_grad()
                y_pred = net(x)
                lossC = unlabelled_loss(y_pred[0], y_pred[1],y)
                lossC.backward()
                optimizer_shared.step()
                loss_adv = lossB.item() + lossC.item()

            # Step A
            optimizer_all.zero_grad()
            y_pred = net(x)
            lossA = (loss_fkt(y_pred[0], y) + loss_fkt(y_pred[1], y))
            lossA.backward()
            optimizer_all.step()

            # print statistics
            train_loss = lossA.item()+loss_adv
            epoch_steps += 1

        # Validation loss
        val_loss = 0.0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                x_val, y_val = data
                y_pred_val = net.predict(x_val)
                val_loss_temp = loss_fkt(y_pred_val[0], y_val) + loss_fkt(y_pred_val[1], y_val) + unlabelled_loss(y_pred_val[0], y_pred_val[1], y_val)
                val_loss += val_loss_temp.cpu().numpy()

        early_stopping_count = early_stopping_count + 1

        if val_loss < best_val_loss:
            early_stopping_count = 0
            best_val_loss = val_loss
            best_net = copy.deepcopy(net.state_dict())

        tune.report(loss=(train_loss), val_loss=val_loss, best_val_loss=best_val_loss)
        if early_stopping_count > config["grace_period"]:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(best_net, path)
            return

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(best_net, path)
    return






