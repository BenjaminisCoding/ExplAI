from torch import nn
import torch.nn.functional as F
import torch 

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
from torch_geometric.data import Data, Batch
from data_augm import process_graph_batch

CE = torch.nn.CrossEntropyLoss()


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

class GINEncoder(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_features, dim_h, dim_encode):
        super(GINEncoder, self).__init__()
        self.conv1 = GINConv(   nn.Sequential(nn.Linear(dim_features, dim_h),
                                nn.BatchNorm1d(dim_h), nn.ReLU(),
                                nn.Linear(dim_h, dim_h), nn.ReLU()))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(dim_h, dim_h),
                                nn.BatchNorm1d(dim_h), nn.ReLU(),
                                nn.Linear(dim_h, dim_h), nn.ReLU()))
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h), nn.BatchNorm1d(dim_h), nn.ReLU(),
                       nn.Linear(dim_h, dim_h), nn.ReLU()))
        self.lin1 = nn.Linear(dim_h*3, dim_h*3)
        self.lin2 = nn.Linear(dim_h*3, dim_encode)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin2(h)
        
        return h
    
### Himbert
    
# class GINEncoder(torch.nn.Module):
#     """GIN"""
#     def __init__(self, dim_features, dim_h, dim_encode):
#         super(GINEncoder, self).__init__()
#         self.l11 = nn.Linear(dim_features, dim_h)
#         self.l12 = nn.Linear(dim_h, dim_h)
#         self.l21 = nn.Linear(dim_h, dim_h)
#         self.l22 = nn.Linear(dim_h, dim_h)
#         self.l31 = nn.Linear(dim_h, dim_h)
#         self.l32 = nn.Linear(dim_h, dim_h)
#         self.l41 = nn.Linear(dim_h, dim_h)
#         self.l42 = nn.Linear(dim_h, dim_h)
        
#         self.conv1 = GINConv(   nn.Sequential(self.l11,
#                                 nn.BatchNorm1d(dim_h), nn.ReLU(),
#                                 self.l12, nn.ReLU()), train_eps=True)
#         self.conv2 = GINConv(nn.Sequential(self.l21,
#                                 nn.BatchNorm1d(dim_h), nn.ReLU(),
#                                 self.l22, nn.ReLU()), train_eps=True)
#         self.conv3 = GINConv(nn.Sequential(self.l31,
#                                 nn.BatchNorm1d(dim_h), nn.ReLU(),
#                                 self.l32, nn.ReLU()), train_eps=True)
#         self.conv4 = GINConv(nn.Sequential(self.l41,
#                                 nn.BatchNorm1d(dim_h), nn.ReLU(),
#                                 self.l42, nn.ReLU()), train_eps=True)
#         self.lin1 = nn.Linear(dim_h*4, dim_h*3)
#         self.lin2 = nn.Linear(dim_h*3, dim_encode)
        
#         print(self.conv1)
#         nn.init.kaiming_normal_(self.l11.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.l12.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.l21.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.l22.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.l31.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.l32.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.l41.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.l42.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.lin1.weight, nonlinearity='relu')
#         nn.init.xavier_normal_(self.lin2.weight)
        

#     def forward(self, graph_batch):
#         x = graph_batch.x
#         edge_index = graph_batch.edge_index
#         batch = graph_batch.batch
        
#         # Node embeddings 
#         h1 = self.conv1(x, edge_index)
#         h2 = self.conv2(h1, edge_index)
#         h3 = self.conv3(h2, edge_index)
#         h4 = self.conv4(h3, edge_index)
#         # Graph-level readout
#         h1 = global_add_pool(h1, batch)
#         h2 = global_add_pool(h2, batch)
#         h3 = global_add_pool(h3, batch)
#         h4 = global_add_pool(h4, batch)
#         # Concatenate graph embeddings
#         h = torch.cat((h1, h2, h3, h4), dim=1)
#         # Classifier
#         h = self.lin1(h)
#         h = h.relu()
#         h = F.dropout(h, p=0.2, training=self.training)
#         h = self.lin2(h)
        
#         return h
    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.L = nn.Linear(768,768)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return self.L(encoded_text.last_hidden_state[:,0,:])
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        self.temp = nn.Parameter(torch.tensor(0.1).log())
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder

    def freeze_bert_encoder(self):
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = False

    def defreeze_bert_encoder(self):
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = True   
    
    def contrastive_loss(self, v1, v2):

        v1_norm = F.normalize(v1, p=2, dim=1)
        v2_norm = F.normalize(v2, p=2, dim=1)
        logits = torch.matmul(v1_norm,torch.transpose(v2_norm, 0, 1)) / self.temp.exp()
        # logits = torch.matmul(v1_norm,torch.transpose(v2_norm, 0, 1)) / self.temp
        labels = torch.arange(logits.shape[0], device=v1.device)
        return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
    
    def contrastive_loss_baseline(self, v1, v2):

        logits = torch.matmul(v1,torch.transpose(v2, 0, 1)) / self.temp
        labels = torch.arange(logits.shape[0], device=v1.device)
        return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)    
    
    def get_graph_encoder(self):
        return self.graph_encoder

class ModelGIN(nn.Module):
    def __init__(self, model_name, num_node_features, dim_h, dim_encode):
        super(ModelGIN, self).__init__()
        self.graph_encoder = GINEncoder(num_node_features, dim_h, dim_encode)     
        self.text_encoder = TextEncoder(model_name)
        self.temp = nn.Parameter(torch.tensor(0.1).log())
        # self.temp_graph = nn.Parameter(torch.tensor(0.1).log()) #temp parameter for the graph constraing learning
        # self.weight = nn.Parameter(torch.tensor(0.25).log()) #if the loss for graph constrating learning is used, parameter to learn the proportion of the loss
        self.ep = 1
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder

    def freeze_bert_encoder(self):
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = False

    def defreeze_bert_encoder(self):
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = True   
    
    def preprocess_graph(self, graph_batch, n=2):
        '''
        Return n * |graph_batch| augmented graphs, n per graph.
        '''
        # Create augmented batches
        augmented_batches = [process_graph_batch(graph_batch) for _ in range(n)]

        # Interleave graphs from the augmented batches
        interleaved_graphs = []
        # for i in range(graph_batch.num_graphs):
        #     for batch in augmented_batches:
        #         # Extracting the i-th graph from the batch
        #         interleaved_graphs.append(batch.get_example(i))
        for batch in augmented_batches:
            for i in range(graph_batch.num_graphs):
                interleaved_graphs.append(batch.get_example(i))

        # Reconstruct the batch from the interleaved list of processed graphs
        new_batch = Batch.from_data_list(interleaved_graphs)
        return new_batch

    
    def contrastive_loss(self, v1, v2):

        v1_norm = F.normalize(v1, p=2, dim=1)
        v2_norm = F.normalize(v2, p=2, dim=1)
        logits = torch.matmul(v1_norm,torch.transpose(v2_norm, 0, 1)) / self.temp.exp()
        # logits = torch.matmul(v1_norm,torch.transpose(v2_norm, 0, 1)) / self.temp
        labels = torch.arange(logits.shape[0], device=v1.device)
        return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
    
    def augmented_loss(self, v1, v2):

        # Loss to use when upsampling the graph batch by data augmentation
        # v1 corresponds to the graph embeddings
        bs = v2.shape[0] #batch_size 
        v1_norm = F.normalize(v1, p=2, dim=1)
        v2_norm = F.normalize(v2, p=2, dim=1)
        logits = torch.matmul(v1_norm, torch.transpose(v2_norm, 0, 1)) / self.temp.exp()

        labels = torch.arange(logits.shape[1], device=v1.device).repeat_interleave(2)

        # Calculate probabilities_text
        probabilities_text = torch.softmax(torch.transpose(logits, 0, 1), dim=1)

        # compute the cross entropy, taking in consideration that a text is associated to two labels, so we need to modify it 
        # mask = F.one_hot(2*torch.arange(bs), num_classes = 2*bs) + F.one_hot(2*torch.arange(bs) +1, num_classes = 2*bs)
        mask = F.one_hot(torch.arange(bs), num_classes = 2*bs) + F.one_hot(torch.arange(bs) +bs, num_classes = 2*bs)
        mask = mask.to(v1.device)
        loss_text = torch.mean((-torch.log((probabilities_text * mask).sum(dim = 1))))

        # Compute the Cross-Entropy loss for the augmented labels and add a constant 4 (for example)
        return CE(logits, labels) + loss_text

    def augmented_loss_graph(self, v1, v2):
        '''
        In this loss, we also code a loss to perform constrative learning on the graph architecture 
        '''
        bs = v2.shape[0] #batch_size 
        v1_norm = F.normalize(v1, p=2, dim=1)
        v2_norm = F.normalize(v2, p=2, dim=1)
        logits = torch.matmul(v1_norm, torch.transpose(v2_norm, 0, 1)) / self.temp.exp() 

        labels = torch.arange(logits.shape[1], device=v1.device).repeat_interleave(2)   
        probabilities_text = torch.softmax(torch.transpose(logits, 0, 1), dim=1)

        # compute the cross entropy, taking in consideration that a text is associated to two labels, so we need to modify it 
        # mask = F.one_hot(2*torch.arange(bs), num_classes = 2*bs) + F.one_hot(2*torch.arange(bs) +1, num_classes = 2*bs)
        mask = F.one_hot(torch.arange(bs), num_classes = 2*bs) + F.one_hot(torch.arange(bs) +bs, num_classes = 2*bs)

        mask = mask.to(v1.device)
        loss_text = torch.mean((-torch.log((probabilities_text * mask).sum(dim = 1))))

        #graph constrative learning 
        logits_graph = torch.matmul(v1_norm[:bs], torch.transpose(v1_norm[bs:], 0, 1)) / self.temp_graph.exp() 
        labels_graph = torch.arange(logits.shape[1], device=v1.device)
        # mask_graph = -1e6 * torch.ones(2*bs,2*bs).to(v1.device)
        # mask_graph[::2, 1::2] = 1
        # mask_graph[1::2, ::2] = 1

        # probabilities_graph = torch.softmax(logits_graph * mask_graph, dim = 1)
        # loss_graph = 0 
        # for idx in range(2*bs):
        #     loss_graph += -torch.log(probabilities_graph[idx, 2 * (idx // 2) + (1 - idx % 2)])
        # loss_graph = loss_graph / (2 * bs)

        # return (1 - self.weight.exp()) * (CE(logits, labels) + loss_text) + self.weight.exp() * CE(logits_graph, labels_graph)
        return (1 - max((0.5/(torch.sqrt(torch.tensor(self.ep)) + 1)), 0.1)) * (CE(logits, labels) + loss_text) + max((0.5/(torch.sqrt(torch.tensor(self.ep)) + 1)), 0.1) * CE(logits_graph, labels_graph)

    def contrastive_loss_baseline(self, v1, v2):

        logits = torch.matmul(v1,torch.transpose(v2, 0, 1)) / self.temp
        labels = torch.arange(logits.shape[0], device=v1.device)
        return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)    
    
    def get_graph_encoder(self):
        return self.graph_encoder