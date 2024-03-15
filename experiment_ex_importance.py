from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model, ModelGIN
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
from torch import optim
import time
import os
import pandas as pd 
from utils.utils import create_folders_if_not_exist
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
import random
import torch_geometric
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from random import seed
from utils.data_utils import get_list_graphs
from data_augm import get_augmented_batch
import pickle 

import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from functools import partial
# import seaborn as sns

# # Set Seaborn style
# sns.set()

from LabelFreeXAI.src.lfxai.utils.influence import hessian_vector_product, stack_torch_tensors
from torch.nn import MSELoss
from LabelFreeXAI.src.lfxai.models.pretext import Identity
from LabelFreeXAI.src.lfxai.explanations.examples import SimplEx

def evolution_example_importance(attr_method, device, L_graphs, L_graphs_idx, train_subset, iter_MC, nbr_nodes_to_remove_max, **kwargs):

    res = [0] * len(L_graphs)
    for idx_g in tqdm(range(len(L_graphs))):
        idx_dataset, graph = L_graphs_idx[idx_g], L_graphs[idx_g]
        dic = {}
        for nbr_nodes_to_remove in tqdm(range(1, min(nbr_nodes_to_remove_max, graph.num_nodes)), leave = False):
            graph_processed = get_augmented_batch(graph.cpu(), nbr_nodes_to_remove = nbr_nodes_to_remove, iter_MC = iter_MC)
            test_subset = [graph_processed.get_example(i) for i in range(graph_processed.num_graphs)]
            example_importance, errors = attr_method.attribute_loader(device, [graph] + train_subset, test_subset, batch_size = iter_MC, label=False, min_delta = 1e-5, verbose = False)
            
            dic[nbr_nodes_to_remove] = (example_importance[:,0].tolist(), np.mean(errors[500:], axis = 0))
        res[idx_g] = dic
    return res

def get_train_subset(train_dataset, L_graphs_idx, n_subset = 100):
    seed(42)
    n_tot, n_subset = train_dataset.__len__(), 100
    idxs = random.sample([k for k in range(0, n_tot) if k not in L_graphs_idx], k = n_subset + 10)
    idx = -1 
    l_train = []
    while len(l_train) < n_subset:
        idx += 1 
        i = idxs[idx]
        batch = train_dataset.get(i)
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        if graph_batch.edge_index.shape[0] != 2:
            continue
        l_train.append(graph_batch) 
    return l_train 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nbr_nodes_to_remove_max', required=True)
    parser.add_argument('--iter_MC', default = 10)
    parser.add_argument('--name_exp', required = True)
    parser.add_argument('--n_nodes', default = 22)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelGIN(model_name, num_node_features=300, dim_h=600, dim_encode=768)
    model.to(device)
    name_exp = "SciBert_Loss_GIN"
    checkpoint = torch.load('./checkpoints/' + name_exp + '/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    graph_model = model.get_graph_encoder() ### get the model needed for the graph
    graph_model.eval()
    graph_model.encoder = graph_model
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)

    attr_method = SimplEx(graph_model, loss_f=MSELoss())
    L_graphs, L_graphs_idx = get_list_graphs(train_dataset, n_nodes = int(args.n_nodes))
    for g in L_graphs:
        if g.edge_index.shape[0] != 2:
            print(g, 'DANGER')
    train_subset = get_train_subset(train_dataset, L_graphs_idx, n_subset = 100)
    res = evolution_example_importance(attr_method, device, L_graphs, L_graphs_idx, train_subset, int(args.iter_MC), int(args.nbr_nodes_to_remove_max))
    ### save the results 
    with open(f'./results/ex_importance/{args.name_exp}.pkl', 'wb') as f:
        pickle.dump(res, f)
    with open(f'./results/ex_importance/description_{args.name_exp}.txt', 'w') as W:
        W.write(f'iter_MC:{args.iter_MC};nbr_nodes_to_remove_max:{args.nbr_nodes_to_remove_max};n_nodes:{args.n_nodes}')
    













