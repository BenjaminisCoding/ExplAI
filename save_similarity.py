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
from utils import create_folders_if_not_exist
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

import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance

def load_embeddings(embedding_folder):
    embeddings = []
    for file in sorted(os.listdir(embedding_folder), key=lambda x: int(x.split('.')[0])):
        embedding = np.load(os.path.join(embedding_folder, file))
        embeddings.append(embedding)
    return np.array(embeddings)

def compute_similarity_matrix(embeddings):
    # Compute pairwise cosine distances and convert to similarity
    cosine_distances = pdist(embeddings, metric='cosine')
    cosine_similarity = 1 - squareform(cosine_distances)
    return cosine_similarity

def save_top_similarities(similarity_matrix, name_exp, top_k=200):
    top_values = np.zeros((similarity_matrix.shape[0], top_k))
    top_indices = np.zeros((similarity_matrix.shape[0], top_k), dtype=int)

    path = f'./similarity/{name_exp}/'
    if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
    for i in range(similarity_matrix.shape[0]):
        top_k_indices = np.argsort(similarity_matrix[i])[-top_k:]
        top_k_values = similarity_matrix[i][top_k_indices]

        top_values[i] = top_k_values
        top_indices[i] = top_k_indices
    # path = f'./similarity/graph/{name_exp}/'
    # path = f'/Data/CTey/similarity/{name_exp}/'

    # create_folders_if_not_exist(path)
    np.save(path + 'top_values.npy', top_values)
    np.save(path + 'top_indices.npy', top_indices)


if __name__ == '__main__':

    #train_dataset, graph_model, name_exp, load the model, the weights, and it should be fine 

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', required=True)
    args = parser.parse_args()
    embedding_folder = f'./embeddings/graph/{args.name_exp}/'
    embeddings = load_embeddings(embedding_folder)
    print('Embeddings loaded!')
    similarity_matrix = compute_similarity_matrix(embeddings)
    print('Saving the matrix...')
    save_top_similarities(similarity_matrix, args.name_exp)