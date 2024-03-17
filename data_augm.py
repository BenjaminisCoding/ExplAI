import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
import random

def remove_random_edge(data):
    # Randomly remove one edge
    edge_index = data.edge_index.t().tolist()
    if len(edge_index) > 0:
        edge_index.pop(random.randint(0, len(edge_index) - 1))
    data.edge_index = torch.tensor(edge_index).t().contiguous()
    return data

def remove_nodes(data, mode = 'uniform'):

    assert mode in ['uniform', 'low_degree', 'high_degree'], 'mode must be uniform, low_degree, high_degree'
    # Compute node degrees
    node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    # Calculate probabilities (higher for nodes with lower degree)
    if mode =='uniform':
        probabilities = torch.ones(node_degrees.shape[0])
    # probabilities = 1.0 / (node_degrees + 1e-6)  
    probabilities /= probabilities.sum()

    #Select the number of nodes to remove using a binomial, with probability .05 
    nbr_nodes_to_remove = min(torch.distributions.Binomial(total_count=node_degrees.shape[0] - 1, probs=.05).sample(), torch.tensor([node_degrees.shape[0] // 2]))

    if nbr_nodes_to_remove == 0:
        return data 
    
    # Randomly select a node for removal
    if mode =='uniform':
        # Sample `nbr_nodes_to_remove` nodes for removal
        all_node_indices = list(range(data.num_nodes))
        nodes_to_remove = set(random.sample(all_node_indices, int(nbr_nodes_to_remove.item())))

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

    # Update the number of nodes
    # data.num_nodes -= len(nodes_to_remove)

    return data


def process_graph_batch(graph_batch):

    processed_graphs = []
    # cumsum = 0
    for i in range(graph_batch.num_graphs):
        data = graph_batch.get_example(i)
        data = remove_nodes(data, mode = 'uniform')        
        processed_graphs.append(data)

    # Reconstruct the batch from the list of processed graphs
    new_batch = Batch.from_data_list(processed_graphs)
    return new_batch