import torch
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

def compute_vector_direction(weights_matrix: np.ndarray) -> List[np.ndarray]:
    '''
    Computes the concept vectors for a classifier using its last linear layer.

    Arguments:
    - weights_matrix: A 2D NumPy array representing the weights of the last linear layer.

    Returns:
    - A list of NumPy arrays, each representing a concept vector.
    '''
    mean_vector = torch.sum(weights_matrix, dim=0)
    n_labels = weights_matrix.shape[0]
    vectors = [weights_matrix[label] - (1 / (n_labels - 1)) * (mean_vector - weights_matrix[label]) for label in range(n_labels)]
    return vectors

def find_border(image: np.ndarray) -> Tuple[int, int, int, int]:
    '''
    Computes the extent to which an image can be translated in all four directions without encountering non-zero pixels.
    '''
    img_array = np.array(image)
    top, bottom = 0, img_array.shape[0]
    left, right = 0, img_array.shape[1]

    top = next((i for i in range(img_array.shape[0]) if not np.all(img_array[i, :] == 0)), top)
    bottom = next((i for i in range(img_array.shape[0] - 1, -1, -1) if not np.all(img_array[i, :] == 0)), bottom)
    left = next((j for j in range(img_array.shape[1]) if not np.all(img_array[:, j] == 0)), left)
    right = next((j for j in range(img_array.shape[1] - 1, -1, -1) if not np.all(img_array[:, j] == 0)), right)

    return top, bottom, left, right

def shift_image_horizontally(image: np.ndarray, h: int) -> np.ndarray:
    """
    Shifts an image horizontally.
    """
    img_array = np.array(image)
    shifted_image = np.zeros_like(img_array)

    if h > 0:
        shifted_image[:, h:] = img_array[:, :-h]
    elif h < 0:
        shifted_image[:, :h] = img_array[:, -h:]

    return shifted_image

def evolution_symmetry(data_loader: torch.utils.data.DataLoader, L_scores: List[Any], model: torch.nn.Module, device: torch.device) -> Dict[str, List[Any]]:
    '''
    Computes the evolution of a score with respect to horizontal image transformations.
    '''
    idx = -1 
    n_test = len(data_loader)
    res = {score.str: [None] * n_test for score in L_scores}
    res['losses'] = [None] * n_test

    with torch.no_grad():
        for batch in tqdm(data_loader):
            idx += 1 
            input, label = batch
            latent = model.forward_latent(input.to(device))
            pred = model.lin_output(latent)
            _, _, _, right_border = find_border(input.squeeze(0).squeeze(0))
            H = 27 - right_border
            batch_translated = torch.zeros((H, 1, 28, 28))
            for h in range(1, H + 1):
                batch_translated[h - 1] = torch.Tensor(shift_image_horizontally(input.squeeze(0).squeeze(0), h))
            latent_translated = model.forward_latent(batch_translated.to(device))
            predictions_translated = model.lin_output(latent_translated)
            for score in L_scores:
                if score.str == 'score_direction_dot':
                    score.id_cv = label
                res[score.str][idx] = score.compute_score(latent, latent_translated)    
            res['losses'][idx] = [model.loss_f(predictions_translated[k].unsqueeze(0), label.to(device)) for k in range(H)]

    return res

def plot_res(res: Dict[str, List[Any]], score_str: str, H: int = 5, H_max: int = 7) -> None:
    '''
    Plots the evolution of scores with respect to horizontal image transformations.
    '''
    values = {k: [] for k in range(H)}
    for sc in res[score_str]:
        if len(sc) >= H_max:
            for idx in range(min(len(sc), H)):
                values[idx].append(sc[idx].cpu())
    
    means = [np.mean(values[idx]) for idx in range(H)]
    stds = [np.std(values[idx]) for idx in range(H)]

    plt.plot(range(1, H + 1), means)
    plt.fill_between(range(1, H + 1), np.array(means) - np.array(stds), np.array(means) + np.array(stds), color='blue', alpha=0.2)
    plt.plot(range(1, H + 1), means, '-o', color='blue')
    plt.xlabel('Pixel translated')
    plt.ylabel(score_str)
    plt.title(f'Evolution of the {score_str} with horizontal translation')
    plt.grid(True)
    plt.show()
