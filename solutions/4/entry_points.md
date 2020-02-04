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

- SEED - The seed for the random number generator used to split the data into training and validation sets.
- SCALED - Whether to use standard deviation scaling of the label variable.
- TRUE_MLKN - Whether to use values of the Mulliken charge from the QM9 dataset, or rely on the predicted values.
- TYPES_TO_PROCESS -  Selection of the scalar coupling types to process. 