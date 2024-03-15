from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
# from Model import Model, ModelGIN
from Model2 import Model, ModelGIN
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
from torch_geometric.data import Data



CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2, temp = 1):

    v1_norm = F.normalize(v1, p=2, dim=1)
    v2_norm = F.normalize(v2, p=2, dim=1)
    logits = torch.matmul(v1_norm,torch.transpose(v2_norm, 0, 1)) / temp
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def contrastive_loss_baseline(v1, v2):

    logits = torch.matmul(v1,torch.transpose(v2, 0, 1)) 
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def train(args):

    save_path = f'./checkpoints/{args.name_exp}'
    log_dir_path = save_path + '/logs'
    create_folders_if_not_exist(log_dir_path, all = True)
    write_description(args) #write description of the experiments
    writer = SummaryWriter(log_dir=log_dir_path)

    # model_name = 'distilbert-base-uncased'
    # model_name = "allenai/scibert_scivocab_uncased"
    model_name = args.model_name 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_epochs = int(args.nb_epochs)
    max_epochs = int(args.max_epochs)
    if max_epochs > 0:
        nb_epochs = 100000
    batch_size = int(args.batch_size)
    learning_rate = float(args.lr)
 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
    # model = ModelGIN(model_name, num_node_features=300, dim_h=300, dim_encode=768)
    model = ModelGIN(model_name, num_node_features=300, dim_h=600, dim_encode=768)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                        betas=(0.9, 0.999),
                        weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7, 
                            verbose=True, min_lr=5e-7)
    
    if args.freeze is not None:
        model.freeze_bert_encoder()
    if args.load is not None:
        checkpoint = torch.load(save_path + '/model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Next epoch after the saved one
        best_validation_loss = checkpoint['validation_accuracy']
        best_validation_score = checkpoint['validation_score']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
        print('Model loaded!')
    else:
        best_validation_loss, best_validation_score = 1000000, 0
        start_epoch = 0

    no_improvement = 0
    loss, loss_ep = 0, 0
    losses = []
    count_iter = 0
    time1 = time.time()
    printEvery = 50
    # best_validation_loss, best_validation_score = 1000000, 0 #### to erase 
    best_validation_loss = 1000

    # if args.loss is None:
    #     loss_function = lambda emb1, emb2 : model.contrastive_loss_baseline(emb1, emb2)
    # else:
    #     loss_function = lambda emb1, emb2 : model.contrastive_loss(emb1, emb2)


    for i in range(start_epoch, nb_epochs): 
        print('-----EPOCH{}-----'.format(i+1))
        if args.freeze is not None and i == 1: #defreeze the scibert after one epoch, so the graph network has time to learn
            model.defreeze_bert_encoder()
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):    
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            batch_processsed = model.preprocess_graph(graph_batch)
            # process_graph_batch(graph_batch)
            
            # x_graph, x_text = model(graph_batch.to(device), 
            #                         input_ids.to(device), 
            #                         attention_mask.to(device))
            if args.middle is None:
                x_graph, x_text = model(batch_processsed.to(device), 
                                        input_ids.to(device), 
                                        attention_mask.to(device))
                # current_loss = contrastive_loss(x_graph, x_text, temp)   
                # current_loss = model.augmented_loss(x_graph, x_text)
                current_loss = model.augmented_loss_graph(x_graph, x_text)
                # current_loss = loss_function(x_graph, x_text)
                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                loss += current_loss.item()
                count_iter += 1

            else: #args.middle is not None 
                x_graph, x_text, x_graph_middle, x_text_middle = model(batch_processsed.to(device), 
                                        input_ids.to(device), 
                                        attention_mask.to(device))
                # current_loss = contrastive_loss(x_graph, x_text, temp)   
                # current_loss = model.augmented_loss(x_graph, x_text)
                current_loss = model.augmented_loss(x_graph, x_text, x_graph_middle, x_text_middle)
                # current_loss_middle = model.augmented_loss(x_graph_middle, x_text_middle, middle = True)
                # loss_tot = 0.7 * current_loss + 0.3 * current_loss_middle
                # current_loss = loss_function(x_graph, x_text)
                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                loss += current_loss.item()
                count_iter += 1


            if count_iter % printEvery == 0:
                time2 = time.time()
                # print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                #                                                             time2 - time1, loss/printEvery))
                losses.append(loss)
                writer.add_scalar('train_loss_step', loss / printEvery, count_iter)
                writer.add_scalar('temp_loss_step', model.temp, count_iter)
                loss_ep += loss
                loss = 0 


        writer.add_scalar("train_loss_ep", printEvery * (loss_ep / len(train_loader)), i)
        loss_ep = 0 

        model.eval()    

        graph_embeddings = []
        text_embeddings = []

        #args.middle not none
        graph_embeddings_middle = []
        text_embeddings_middle = []    
        
        val_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader)):
                input_ids = batch.input_ids
                batch.pop('input_ids')
                attention_mask = batch.attention_mask
                batch.pop('attention_mask')
                graph_batch = batch
                if args.middle is None:
                    x_graph, x_text = model(graph_batch.to(device), 
                                            input_ids.to(device), 
                                            attention_mask.to(device))

                    for graph_emb in x_graph:
                        graph_embeddings.append(graph_emb.tolist())
                    for graph_text in x_text:
                        text_embeddings.append(graph_text.tolist())
                else:
                    x_graph, x_text, x_graph_middle, x_text_middle = model(graph_batch.to(device), 
                                                input_ids.to(device), 
                                                attention_mask.to(device))
                    
                    for graph_emb in x_graph:
                        graph_embeddings.append(graph_emb.tolist())
                    for graph_text in x_text:
                        text_embeddings.append(graph_text.tolist()) 

                    for graph_emb_mid in x_graph_middle:
                        graph_embeddings_middle.append(graph_emb_mid.tolist())
                    for graph_text_mid in x_text_middle:
                        text_embeddings_middle.append(graph_text_mid.tolist())


                # current_loss = contrastive_loss(x_graph, x_text, temp)   
                current_loss = model.contrastive_loss(x_graph, x_text)
                # current_loss = loss_function(x_graph, x_text)

                val_loss += current_loss.item()

                # for graph_emb in x_graph:
                #     graph_embeddings.append(graph_emb.tolist())
                # for graph_text in x_text:
                #     text_embeddings.append(graph_text.tolist())
            if args.middle is not None:
                similarity = 0.8 * cosine_similarity(text_embeddings, graph_embeddings) + 0.2 * cosine_similarity(text_embeddings_middle, graph_embeddings_middle)
            else:
                similarity = cosine_similarity(text_embeddings, graph_embeddings)
            solution = pd.DataFrame(similarity)
            solution['ID'] = solution.index
            solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]

            df_predicted = solution
            cid_to_id = {v: k for k, v in val_dataset.idx_to_cid.items()}
            y_true = []
            for cid in val_dataset.cids:
                y_true.append(cid_to_id[cid])          
            df_true = np.zeros((len(y_true), df_predicted.shape[1] - 1))
            for y, pos in enumerate(y_true):
                df_true[y, pos] = 1    
            score_val = label_ranking_average_precision_score(df_true, df_predicted.to_numpy()[:,1:])
            writer.add_scalar('val_score', score_val, i)
            scheduler.step(score_val) 
            model.ep += 1 ####
        # return label_ranking_average_precision_score(df_true, df_predicted.to_numpy()[:,1:])
            # del solution
            # del df_true
            # del similarity

            writer.add_scalar('val_loss', val_loss/len(val_loader), i)
            writer.add_scalar('best_val_loss', min(best_validation_loss,val_loss)/len(val_loader), i)
            print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
            print('-----EPOCH'+str(i+1)+'----- done.  Validation score: ', str(score_val) )

            
            # if min(best_validation_loss, val_loss)==val_loss:
            if max(best_validation_score, score_val) == score_val:
                print('validation score improved saving checkpoint...')
                # save_path_weights = save_path + '/model'+ str(i) + '.pt'
                if args.load is not None:
                    save_path_weights = save_path + f'/model_{args.load}' + '.pt'
                else:
                    save_path_weights = save_path + '/model' + '.pt'
                create_folders_if_not_exist(save_path_weights)
                torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'validation_accuracy': val_loss,
                'validation_score': score_val,
                'loss': loss,
                }, save_path_weights)
                print('name file to save', save_path_weights)
                print('checkpoint saved to: {}'.format(save_path_weights))
            if max_epochs > 0: #if we want to stop the training when the val loss no longer increases 
                if best_validation_score > score_val + float(args.delta):
                    no_improvement += 1 
                else:
                    no_improvement = 0 #restart the waiting 
                best_validation_loss = min(best_validation_loss, val_loss)
                best_validation_score = max(best_validation_score, score_val)
                if i == max_epochs - 1 or no_improvement == int(args.C_epochs):
                    # break
                    writer.close()
                    return
            else:
                best_validation_loss = min(best_validation_loss, val_loss)
                best_validation_score = max(best_validation_score, score_val)
    writer.close()
        

def write_description(args):

    with open(f'./checkpoints/{args.name_exp}/description.txt', 'w') as W:
       for arg_name, arg_value in vars(args).items():
            W.write(f'Argument Name: {arg_name}, Argument Value: {arg_value}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epochs', default=5)
    parser.add_argument('--max_epochs', default=0) ### if max_epochs is set to a value above 0, it means we want to train as long as the validation loss decreases 
    parser.add_argument('--C_epochs', default=15) ### if max_epochs > 0, correspond to the number of epoch to continue training when the vall loss does not improve
    parser.add_argument('--delta', default=0) ### if max_epochs > 0, it means that the best val loss needs to be lower than the previous by delta to trigger a new start of the loop
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--name_exp', required=True)
    parser.add_argument('--freeze', default=None)
    parser.add_argument('--temp', default=1)
    parser.add_argument('--load', default=None)
    parser.add_argument('--model_name', default='distilbert-base-uncased')
    parser.add_argument('--loss', default=None) #if None, base loss
    parser.add_argument('--middle', default=None)
    args = parser.parse_args()
    train(args)


    #command line:

    #python train.py --max_epochs 15 --model_name "allenai/scibert_scivocab_uncased" --name_exp "SciBert_bs128" --freeze 1
    #Exp 3 : python train.py --max_epochs 50 --batch_size 64 --lr 3e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "SB_bs64_learned_loss_logtemp" --freeze 1 
    #python train.py --nb_epochs 2 --name_exp "test" 
    # python train.py --max_epochs 50 --batch_size 64 --lr 3e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "SB_bs64_learned_loss_logtemp" --load 1

    #to reproduce the baseline
    #python train.py --nb_epochs 5 --name_exp baseline

    #exp 4: we delete the processed data so it uses the tokenizer of scibert 
    #python train.py --max_epochs 75 --batch_size 64 --lr 3e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "SB_bs64_learned_loss_logtemp_rightprocs" --freeze 1
    #python train.py --max_epochs 50 --batch_size 64 --lr 2e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "SciBert_Loss_GIN" --freeze 1
    #python train.py --max_epochs 75 --batch_size 16 --lr 2e-5 --model_name "facebook/galactica-1.3b" --name_exp "FbG_loss_GCN" --freeze 1 
    #python train.py --max_epochs 200 --batch_size 64 --lr 2e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "Loss_updated" --freeze 1 
    #python train.py --max_epochs 200 --batch_size 64 --lr 2e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "Loss_updated_graph" --freeze 1 
    #python train.py --max_epochs 200 --batch_size 64 --lr 2e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "Loss_updated_graph" --freeze 1 
    #python train.py --max_epochs 200 --batch_size 64 --lr 2e-5 --model_name "allenai/scibert_scivocab_uncased" --name_exp "new_arch_2" --freeze 1 --middle 1
