# Hello!

Below you can find an outline of how to reproduce our solution for the CHAMPS competition.
 

# ARCHIVE CONTENTS  
    Prepare.ipynb             : code to prepare the input data for the model
    Train.ipynb               : code to rebuild models from scratch
    Predict.ipynb             : code to generate predictions from a trained model
    aux_data                  : contains additional data used by the model (Mulliken charges from QM9)
    modles                    : contains the set of serialized model parameters used in the competition
    SETTINGS.json             : contains the paths to the data used in the prepare, train, and predict ipynb files


# HARDWARE: (The following specs were used to create the original solution)  
* Ubuntu 16.04 LTS 
* 16 vCPUs, 128 GB memory
* 1 x NVIDIA 2080 Ti

# SOFTWARE:  
Dockerfile provided in this folder.
Original docker image is available on docker hub: slartibartfast/champs.
Docker image expects that code and data is mounted in `/workspace` path and that port 8888 is exposed. 

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
below are the shell commands used in each step, as run from the top level directory

    mkdir -p data/raw/
    cd data/raw/
    kaggle competitions download -c champs-scalar-coupling

Running `Predict.ipynb` notebook will generate many intermediary files, plus `train_gdata.torch` and `test_gdata.torch` which are final input files for train and test respectively.

# DATA PROCESSING  
Predictions are made by executing the code in the Predict.ipynb notebook. 
Prior to running the notebook, several paths must be set in the SETTINGS.json file:

1) Path to models to be used when predicting the scalar coupling, for each of the coupling types.
Multiple models may be listed for each coupling type, in which case the final prediction will be the 
average of the values predicted by each of the models. This has already been set to point the to the
serialized models we provide, that have been used to generate the competition predictions.

2) Path to input data, as prepared by running Prepare.ipynb. This has been set to point to the preprocessed competition test data,
which we provided.

3) Path to the file with the Mulliken charges from the QM9 dataset, if the model relies on that data.

4) Output location, where we write the out files. These include:
    - A set of files named <COUPLING TYPE>_X.feather. These are the intermediate predictions made by individual models
    - A file named predictions.csv. the set of final, averaged predictions for all of the coupling types

# MODEL BUILD: 
Predictions are made by executing the code in the Train.ipynb notebook. 
Prior to running the notebook, several paths must be set in the SETTINGS.json file:

1) Path to input data, as prepared by running Prepare.ipynb. This has been set to point to the preprocessed competition test data,
which we provided.

2) Path to the file with the Mulliken charges from the QM9 dataset, if the model relies on that data.

3) Path to an output folder where Tensorboard logs will be written during the training.

4) Path to an output folder where serialized model checkpoints will be written during the training.

5) Path to a model checkpoint which should be used as the starting point during the training, if the training is to start from a pre-trained model.

6) Model name, to be used when saving the Tensorboard logs and model checkpoints.

Training the model also requires several parameters to be set in first cell of the Train.ipynb notebook:

- SEED - The seed for the random number generator used to split the data into training and validation sets. In the competition we use two splits with the seeds 42 and 43.
- SCALED - Whether to use standard deviation scaling of the label variable. Some of the components of the model we used in the competition use the scaling, while others don't (details below).
- TRUE_MLKN - Whether to use values of the Mulliken charge from the QM9 dataset, or rely on the predicted values.
- TYPES_TO_PROCESS -  Selection of the scalar coupling types to process. Some of the components of the model we used in the competition are jointly trained on all of the types, while others are fine tuned for specific coupling types (details below).

## Replicating the competition model

Replication of the competition model requires a number of model components to be built.
#### Pre-training
The pre-training phase involves training 4 models for 100 epochs, which are later used as starting points for fine tuning.
    1. SCALED42, with parameters:
        - SEED = 42
        - SCALED = True
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [0, 1, 2, 3, 4, 5, 6, 7]  

  
    2. SCALED43, with parameters:
        - SEED = 43
        - SCALED = True
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [0, 1, 2, 3, 4, 5, 6, 7]  

  
    3. UNSCALED42, with parameters:
        - SEED = 42
        - SCALED = False
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [0, 1, 2, 3, 4, 5, 6, 7]  
 
  
    4. UNSCALED43, with parameters:
        - SEED = 43
        - SCALED = False
        - TRUE_MLKN = True
       - TYPES_TO_PROCESS = [0, 1, 2, 3, 4, 5, 6, 7]  

#### Fine tuning 
Starting from the 4 pre-trained models we train the following set of components of the final model (each of the components is trained by passing the appropriate pre-trained model via the SETTINGS.json file):

    1) Two models for 1JHC
        - starting from UNSCALED42/UNSCALED43
        - SEED = 42/43
        - SCALED = False
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [0]

    2) Two models for 2JHC
        - starting from UNSCALED42/UNSCALED43
        - SEED = 42/43
        - SCALED = False
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [2]

    3) Two models for 3JHC
        - starting from SCALED42/SCALED43
        - SEED = 42/43
        - SCALED = True
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [5]

    4) Two models for 1JHN, 2JHN, and 3JHN
        - starting from SCALED42/SCALED43
        - SEED = 42/43
        - SCALED = True
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [1, 4, 7]

    5) Two models for 2JHH
        - starting from SCALED42/SCALED43
        - SEED = 42/43
        - SCALED = True
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [3]  
        
    6) Two models for 2JHH
        - starting from SCALED42/SCALED43
        - SEED = 42/43
        - SCALED = True
        - TRUE_MLKN = True
        - TYPES_TO_PROCESS = [6]


