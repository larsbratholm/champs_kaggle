# analysis
Scripts used to generate the analysis and figures for the paper and the SI.

[download_submissions.sh](./download_submissions.sh) downloads all unique final submissions from the top 400 teams.
These are needed to be able to run the [preprocess.py](./preprocess.py) script.

[preprocess.py](./preprocess.py) parses the submissions and creates a pickle with the processed data.
This pickle can be downloaded with the [download_preprocessed_data.sh](./download_preprocessed_data.sh) script.

[download_preprocessed_data.sh](./download_preprocessed_data.sh) downloads the output from the [preprocess.py](-/preprocess.py) script.

[download_duplicate_submissions.sh](./download_duplicate_submissions.sh) downloads submissions that were a duplicate of another submission and was therefor removed from analysis.
