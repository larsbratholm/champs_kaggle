# analysis
Scripts used to generate the analysis and figures for the paper and the SI.

[data](./data/) contains all original or intermediate data files needed for the analysis. These files can either be downloaded by running the download scripts or by running the python scripts.

[download_submissions.sh](./download_submissions.sh) downloads all unique final submissions from the top 400 teams.
These are needed to be able to run the [preprocess.py](./preprocess.py) script.

[preprocess.py](./preprocess.py) parses the submissions downloaded by [download_submissions.sh](./download_submissions.sh) and creates a pickle with the processed data.
This pickle can be downloaded with the [download_preprocessed_data.sh](./download_preprocessed_data.sh) script.

[download_preprocessed_data.sh](./download_preprocessed_data.sh) downloads the output from the [preprocess.py](-/preprocess.py) script.

[output](./output/) will contain the output from the [blend.py](./blend.py) and [make_plots.py](./make_plots.py) scripts.

[blend.py](./blend.py) fits the weights of two different ensembles, writes auxiliary data files to the [data](./data/) folder and writes the scores and weights to the [output](./output/) folder.

[make_plots.py](./make_plots.py) creates and writes the plots shown in the paper and supporting information to the [output](./output/) folder.

[download_duplicate_submissions.sh](./download_duplicate_submissions.sh) downloads submissions that were a duplicate of another submission and was therefor removed from analysis.

