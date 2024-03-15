# ExplAI


## Running the experiments 

The experiments can be run using the notebooks experiments_graphs.ipynb and experiments_MNIST.ipynb

### Graphs experiments 

#### Training the models

1. **train.py**
   - *Description*: File to run and train a new experiment.
   - *Usage*: See the list of arguments in the file to run it. When running an experiment, it automatically creates a checkpoints folder, a folder with the name of the experiment and it stored inside the weights, and the logs during the training.
   - *Example*: `python train.py --max_epochs 15 --model_name "allenai/scibert_scivocab_uncased" --name_exp "exp" --freeze 1`

2. **Model.py**
   - *Description*: Contains the main model to be trained.

#### Evaluation and Testing Files

3. **eval.py**
   - *Description*: File for creating a submission CSV for the validation dataset.

4. **score.py**
   - *Description*: File for computing the score from a CSV on the validation dataset.

#### Utility and Data Handling Files

5. **utils.py**
    - *Description*: Contains several utility functions.

6. **dataloader.py**
    - *Description*: File to load the data.

7. **data_augm.py**
    - *Description*: File for performing data augmentation during the training.

The model will be saved in the folder './checkpoints/{name_exp}/model.pt', with name_exp the name of the experiment considered. 

#### Saving the embeddings and the similary matrix 

Those files are necessary to compute the plots. 

1. **save_embeddings.py**
   - *Description*: File for computing the score from a CSV on the validation dataset.



The different files used:

- Model.py: to load the architecture of the constrative learning model
- dataloader.py: to load the data 
- train.py: to train the model. It 
- 