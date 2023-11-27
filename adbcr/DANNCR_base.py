import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from ray import tune
import os
import copy
from adbcr.Network_base import network_block
from torch.autograd import Function

class ReverseGradient(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANNCR(nn.Module):
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_nodes_dc, out_features,
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

        self.domain_classifier = network_block(
                num_nodes_shared[-1], num_nodes_dc, 2,
                batch_norm, dropout, output_bias=True, activation=nn.ELU)
        self.softmax=torch.nn.Softmax(dim=1)

        torch.manual_seed(random_seed)

    def forward(self, input: Tensor, alpha):
        if torch.cuda.is_available():
            device = input.get_device()
            y = torch.zeros((input.size()[0], len(self.first_rep_net)), device=device)
        else:
            y = torch.zeros((input.size()[0], len(self.first_rep_net)))
        x_latent = self.shared_net(input)

        for i, net in enumerate(self.first_rep_net):
            out_head = net(x_latent)
            y[:, i] = out_head.flatten()

        reversed_x_latent = ReverseGradient.apply(x_latent, alpha)
        domain_label = self.domain_classifier(reversed_x_latent)

        return y, self.softmax(domain_label)

    def predict(self, X):
        self.eval()
        pred = self(X, alpha=0)
        self.train()
        return pred[0], pred[1]

    def predict_numpy(self, X):
        self.eval()
        X = torch.Tensor(X)
        out = self(X, alpha=0)
        self.train()
        return out[0].detach().numpy()



class MSE_inore_NAN(torch.nn.Module):
    """Factual MSE loss."""
    def __init__(self, ):
        super().__init__()

    def forward(self, y_pred: Tensor, y: Tensor) -> Tensor:
        mse = nn.MSELoss()
        mask = ~torch.isnan(y)
        loss = mse(y_pred[mask], y[mask])
        return loss




def fit_DANNCR(config):

    X_train, Y_train = torch.load(config["data_path"])

    net = DANNCR(in_features=X_train.shape[1], num_nodes_shared=config["shared_layer"],
                num_nodes_indiv=config["individual_layer"],num_nodes_dc=config["dc_layer"], out_features=1, batch_norm=config["batch_norm"],
                dropout=config["dropout"], n_treatments=config["number_treatments"], random_seed=config["seed"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
    net.to(device)
    best_net = copy.deepcopy(net.state_dict())
    X_train = torch.Tensor(X_train).to(device)
    Y_train = torch.Tensor(Y_train).to(device)

    optimizer_all = torch.optim.SGD(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


    domain_label = torch.isnan(Y_train[:,1])*1
    data = TensorDataset(X_train, Y_train, domain_label)
    test_abs = int(len(X_train) * (1 - config["val_set_fraction"]))
    train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs], generator=torch.Generator().manual_seed(config["seed"]))

    trainloader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True, drop_last=True)
    valloader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=True)

    best_val_loss = np.Inf
    loss_fkt = MSE_inore_NAN().to(device)
    loss_fkt_dc = torch.nn.NLLLoss().to(device)
    early_stopping_count = 0
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        epoch_steps = 0
        train_loss = 0.0

        start_steps = (epoch+1) * len(trainloader)
        total_steps = start_steps

        for i, data in enumerate(trainloader, 0):
            x, y, domain_label = data

            p = float(i + start_steps) / total_steps
            #alpha = config["number_adversarial_steps"]
            alpha = (2. / (1. + np.exp(-10 * p)) - 1)

            optimizer_all.zero_grad()
            y_pred, d_label_pred = net(x, alpha)
            pred_loss = loss_fkt(y_pred, y)
            dc_loss = loss_fkt_dc(d_label_pred, domain_label)
            
            total_loss = pred_loss + dc_loss
            total_loss.backward()
            optimizer_all.step()


            # print statistics
            train_loss = total_loss.item()
            epoch_steps += 1

        # Validation loss
        val_loss = 0.0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                x_val, y_val, domain_label_val = data
                y_pred_val, d_label_pred_val = net.predict(x_val)
                val_loss_temp = loss_fkt(y_pred_val, y_val)
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






