# ExplAI

The experiments can be run using the notebooks experiments_graphs.ipynb and experiments_MNIST.ipynb

## Package lfxai

I modified the src/lfxai/explanations/examples.py and src/lfxai/explanations/features.py files. 
For the optimization process of the Simplex method, I introduced an early stopping criterion, since I found that the n_epochs parameter indicated of 5000 was in most cases not enough. 
For the features method, I introduced the new score function to compute the feature importance, changed the code so easily leverage it in feature importance calculation. 

## Graphs experiments 

### Experiment setup

#### Downloading the data 

Downloaded from: [Text2Mol Repository](https://github.com/cnedwards/text2mol.git)

#### Training the models

1. **train.py**
   - *Description*: Script to run and train a new experiment.
   - *Usage*: Refer to the argument list in the file. Automatically creates a checkpoint folder and a subfolder for the experiment, where weights and logs are stored.
   - *Example*: `python train.py --max_epochs 15 --model_name "allenai/scibert_scivocab_uncased" --name_exp "exp" --freeze 1`

2. **Model.py**
   - *Description*: Contains the main training model.

#### Evaluation and Testing Files

3. **eval.py**
   - *Description*: Script for generating a submission CSV for the validation dataset.

4. **score.py**
   - *Description*: Script for calculating the score from a CSV on the validation dataset.

#### Utility and Data Handling Files

5. **utils.py**
    - *Description*: Various utility functions.

6. **dataloader.py**
    - *Description*: Data loading script.

7. **data_augm.py**
    - *Description*: Data augmentation utility during training.

The model will be saved in the folder './checkpoints/{name_exp}/model.pt', with name_exp the name of the experiment considered. 

#### Saving the embeddings and the similarity matrix 

Those files are necessary to compute the plots. 

1. **save_embeddings.py**
   - *Description*: Saves embeddings of graph and text data from the training set. This preprocessing step speeds up further computations.
   - *Usage*: Set `text` to 0 to save only graph embeddings; any other number saves both.
   - *Example*: `python save_embeddings.py --name_exp "name_exp" --text 1`

2. **save_similarity.py**
   - *Description*: Computes the 200 closest graphs in the latent space using cosine similarity for every graph. This is essential for comparing performance decreases in perturbed graphs.
   - *Example*: `python save_similarity.py --name_exp "name_exp"`

### Running the experiments

**utils/data_utils.py**
   - *Description*: Contains functions useful for experiments.

**experiment_ex_importance.py**
   - *Description*: Saves embeddings and performs analysis. Results are saved in `./results/ex_importance`.
   - *Usage*: Specify the max number of nodes to remove (`nbr_nodes_to_remove_max`), the number of samples for a given perturbation (`iter_MC`), experiment name (`name_exp`), and the subset of graphs with `n_nodes` in the dataset.
   - *Example*: `python experiment_ex_importance.py --name_exp "name_exp" --iter_MC 100 --nbr_nodes_to_remove_max 5 --n_nodes 22`

**experiments_graphs.ipynb**
   - *Description*: Jupyter notebook for running various graph experiments.

## MNIST experiments 

**experiments_MNIST.py**
   - *Description*: Contains functions useful for MNIST experiments.

**experiments_MNIST.ipynb**
   - *Description*: Notebook to execute MNIST experiments.
