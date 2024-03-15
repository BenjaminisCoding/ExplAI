from functools import partial
import torch.nn.functional as F
import torch
from tqdm import tqdm 
from utils.data_utils import get_augmented_batch


class Scores:
    def __init__(self):
        self.score = None

    def compute_score(self, latent_original, batch):
        if self.score is not None:
            similarities = self.score(latent_original, batch)
            return similarities
        else:
            raise NotImplementedError("Score function not implemented")

class Cosine(Scores):
    def __init__(self):
        super().__init__()  # Correctly call the parent class' __init__
        self.score = partial(F.cosine_similarity, dim=1)
        self.str = 'cosine'
        
class Dot(Scores):
    def __init__(self):
        super().__init__()  # Correctly call the parent class' __init__
        self.score = partial(F.cosine_similarity, dim=1)
        self.score = self.dot
        self.str = 'dot'

    def dot(self, original_latent, latents):
        return torch.sum(original_latent * latents, dim=1)

class Score_Direction_dot(Scores):
    def __init__(self, concept_vectors):
        super().__init__()  # Correctly call the parent class' __init__
        self.concept_vectors = concept_vectors
        self.score = self.direction_dot
        self.id_cv = None 
        self.str = 'score_direction_dot'
        
    def direction_dot(self, original_latent, latents):
        delta = latents - original_latent
        scores = torch.sum(delta * self.concept_vectors[self.id_cv].unsqueeze(0), dim=1)
        return scores 
    
class Score_Direction_cos(Scores):
    def __init__(self, concept_vectors):
        super().__init__()  # Correctly call the parent class' __init__
        self.concept_vectors = concept_vectors
        self.score = self.direction_dot
        self.id_cv = None 
        self.str = 'score_direction_cos'

    def direction_dot(self, original_latent, latents):
        delta = latents - original_latent
        scores = F.cosine_similarity(delta, self.concept_vectors[self.id_cv], dim=1)
        return scores 


def compute_evolution_2(L_graphs, L_graphs_idx, n_nodes, score, graph_model, iter_MC, vectors_directions, use_directions = False):

    #text_model not used for now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores_sample = {} #### define a dictionnary, that will take as keys the number of nodes from 1 to n_nodes, and we will append the sample obtained 
    for n_node in range(1, n_nodes):
        scores_sample[n_node] = []
    
    for idx_batch in tqdm(range(len(L_graphs))):

        graph_batch = L_graphs[idx_batch].clone()
        # input_ids = batch.input_ids
        # batch.pop('input_ids')
        # attention_mask = batch.attention_mask
        # batch.pop('attention_mask')
        # graph_batch = batch
        with torch.no_grad():
            latent_original = graph_model(graph_batch.to(device))

            for iter_node in range(1, n_nodes):

                if not use_directions:
                    batch_processed = get_augmented_batch(graph_batch.cpu(), nbr_nodes_to_remove = iter_node, iter_MC = iter_MC)
                    latent_processed = graph_model(batch_processed.to(device))
                    scores = score.compute_score(latent_original, latent_processed)
                    scores_sample[iter_node] += scores.tolist()
                else:
                    batch_processed = get_augmented_batch(graph_batch.cpu(), nbr_nodes_to_remove = iter_node, iter_MC = iter_MC)
                    latent_processed = graph_model(batch_processed.to(device))
                    #score_with_fixed_direction = partial(score, vector_direction=vectors_directions)
                    #scores = compute_score(latent_original, latent_processed, score_with_fixed_direction)
                    score.id_cv = L_graphs_idx[idx_batch]
                    scores = score.compute_score(latent_original, latent_processed)
                    scores_sample[iter_node] += scores.tolist()
                ### append the scores obtained to the dictionnary at the good n_node key. 

    return scores_sample ###return the dictionnary