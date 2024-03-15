import numpy as np
import os

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