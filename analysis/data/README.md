# data
All unique final submissions from the top 400 teams can be downloaded by running the [download_submissions.sh](../download_submissions.sh) script.
These are needed for running the [preprocess.py](../preprocess.py) script.
The output of this script (which is needed for running the analysis and plot scripts) can be downloaded by running the [download_preprocessed_data.sh](../download_preprocessed_data.sh) script.

[kaggle_competitions.txt](./kaggle_competitions.txt) contain data related to how the number of participants and prize pool of a competition is related.

[public_kernels.csv](./public_kernels.csv) contain data about how public kernel scores changed over time.

[scores](./scores) contain scores from teams at various values of k (see paper)

[ensemble_scores](./ensemble_scores) contain scores from ensembles trained at various values of k.

[n_contrib](./n_contrib) contain the number of models with a score greater than 0.01 in the ensembles trained at various values of k.
