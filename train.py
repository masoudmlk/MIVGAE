from __future__ import division
from __future__ import print_function

import argparse
import time
import pandas as pd
import os
from pathlib import Path



import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from gae.model import GCNModelVAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='citeseer', help='type of dataset.')

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    lst_result=[]
    for i in range(10):
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        hidden_emb = None
        max_roc_ap =0
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy()

            roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            roc_ap = roc_curr + ap_curr
            if max_roc_ap < roc_ap:
                max_roc_ap = roc_ap
                h_emb_best_model = hidden_emb

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "val_ap=",
                  "{:.5f}".format(ap_curr), "time=", "{:.5f}".format(time.time() - t))
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            print('Test ROC score: ' + str(roc_score))
            print('Test AP score: ' + str(ap_score))
            print("---------------------------------------")
        print("Optimization Finished!: ", i)
        roc_score, ap_score = get_roc_score(h_emb_best_model, adj_orig, test_edges, test_edges_false)

        # roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
        lst_result.append([i, roc_score, ap_score])

        print('Test ROC score: ' + str(roc_score))
        print('Test AP score: ' + str(ap_score))
    lst_result = np.array(lst_result)
    csv_info = np.append(lst_result, [["mean", np.mean(lst_result[:, 1]), np.mean(lst_result[:, 2])]], axis=0)
    csv_info = np.append(csv_info, [["std", np.std(lst_result[:, 1]), np.std(lst_result[:, 2])]], axis=0)


    t = int(time.time())
    folder = Path(os.path.join(os.getcwd(), "csv"))
    csv_name = "{}_{}_{}_{}_{}.csv".format(args.dataset_str, args.epochs, args.hidden1, args.hidden2, t)

    df = pd.DataFrame(csv_info, columns=['run', 'ROC', "AP"])
    df.to_csv(os.path.join(folder, csv_name))


if __name__ == '__main__':
    gae_for(args)
