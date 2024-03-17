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
# from torch.utils.tensorboard import SummaryWriter
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

import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from functools import partial

def get_list_graphs(train_dataset, n_nodes, lim = 1e7):

    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)
    n_nodes = 22
    L_graphs, L_graphs_idx = [], []
    idx = -1

    for batch in tqdm(train_loader):
        idx += 1
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
    
        for i in range(graph_batch.num_graphs):
            graph = graph_batch.get_example(i)
            # l_nodes[graph.num_nodes] += 1
            if graph_batch.get_example(i).num_nodes == n_nodes:
                L_graphs.append(graph_batch.get_example(i))
                L_graphs_idx.append(idx * train_loader.batch_size + i)
        if len(L_graphs) >= lim:
            return L_graphs[:lim], L_graphs_idx[:lim]
    return L_graphs, L_graphs_idx

def load_embeddings(embedding_folder, lim = 1e7):
    embeddings = []
    idx = -1
    for file in sorted(os.listdir(embedding_folder), key=lambda x: int(x.split('.')[0])):
        idx += 1 
        embedding = np.load(os.path.join(embedding_folder, file))
        embeddings.append(embedding)
        if idx == lim:
            break
    return np.array(embeddings)

def compute_vector_direction(text_embds):
    # Calculate the sum of all embeddings
    total_sum = np.sum(text_embds, axis=0)

    # Initialize an array to store the direction vectors
    vectors = [None] * text_embds.shape[0]

    # Compute the direction vector for each embedding
    for idx in range(text_embds.shape[0]):
        vectors[idx] = text_embds[idx] - (total_sum - text_embds[idx])

    return vectors

def get_augmented_batch(graph_batch, nbr_nodes_to_remove = 1, iter_MC = 100):

    '''
    Input: a graph
    Return: Remove nodes randomnly. First sample the number of nodes to remove (binomial), then removes uniformaly
    ''' 
    processed_graphs = []
    for _ in range(iter_MC):
        # data = graph_batch.get_example(0)
        data = graph_batch.clone()
    
        all_node_indices = list(range(data.num_nodes))
        nodes_to_remove = set(random.sample(all_node_indices, nbr_nodes_to_remove)) ### sample according to the transformation distribution

        # Remove the selected nodes and their edges
        mask = torch.tensor([i not in nodes_to_remove for i in range(data.num_nodes)])
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        data.edge_index = data.edge_index[:, edge_mask]

        # Adjust node features and other node-related attributes if necessary
        if data.x is not None:
            data.x = data.x[mask]

        # Adjust edge indices to reflect the removed nodes
        for i in range(2):
            edge_indices = data.edge_index[i]
            for removed_node in sorted(nodes_to_remove, reverse=True):
                edge_indices[edge_indices > removed_node] -= 1
        processed_graphs.append(data)

    new_batch = Batch.from_data_list(processed_graphs)
    return new_batch



def compute_vector_direction_mean(text_embds):
    # Calculate the sum of all embeddings
    N = text_embds.shape[0]
    total_sum = np.sum(text_embds, axis=0)

    # Initialize an array to store the direction vectors
    vectors = [None] * text_embds.shape[0]

    # Compute the direction vector for each embedding
    for idx in range(N):
        vectors[idx] = text_embds[idx] - (1/(N-1)) * (total_sum - text_embds[idx])

    return vectors



