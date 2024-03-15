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

def save_embeddings(train_dataset, batch_size, graph_model, text_model, name_exp, text = False):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # create_folders_if_not_exist(f'./embeddings/graph/{name_exp}/_')
    if not text:
        path = f'/Data/CTey/embeddings/graph/{name_exp}/'
        print('Saving embeddings for the graph at path:', path)
    if text:
        path = f'/Data/CTey/embeddings/text/{name_exp}/'
        print('Saving embeddings for the text at path:', path)
    # create_folders_if_not_exist(path +'_')
    with torch.no_grad():

        # for idx, batch in tqdm(enumerate(train_loader)):
        idx = -1
        for batch in tqdm(train_loader, total=train_loader.__len__(), desc='Training', leave=True):
            idx += 1
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            if not text:
                latents = graph_model(batch.to(device)) 
            if text:
                latents = text_model(input_ids.to(device), attention_mask=attention_mask.to(device))
                
            for i, latent in enumerate(latents):
                if not text:
                    np.save(f'/Data/CTey/embeddings/graph/{name_exp}/{idx * batch_size + i}.npy', latent.cpu())
                if text:
                    np.save(f'/Data/CTey/embeddings/text/{name_exp}/{idx * batch_size + i}.npy', latent.cpu())
         

if __name__ == '__main__':

    #train_dataset, graph_model, name_exp, load the model, the weights, and it should be fine 

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', required=True)
    parser.add_argument('--model_name', default="allenai/scibert_scivocab_uncased")
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--text', default=0)
    args = parser.parse_args()

    # model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelGIN(args.model_name, num_node_features=300, dim_h=600, dim_encode=768)
    model.to(device)
    # name_exp = "SciBert_Loss_GIN"
    checkpoint = torch.load('./checkpoints/' + args.name_exp + '/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    graph_model = model.get_graph_encoder() ### get the model needed for the graph
    text_model = model.get_text_encoder()
    graph_model.eval()
    text_model.eval()

    save_embeddings(train_dataset, int(args.batch_size), graph_model, text_model, args.name_exp, text = bool(int(args.text)))
