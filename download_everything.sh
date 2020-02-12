#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Download everything for dataset generation
$DIR/dataset_generation/download_scripts/download_all_data_generation.sh
# Download everything for solutions
$DIR/solutions/download_kaggle_dataset.sh
$DIR/solutions/3/download_checkpoints.sh
$DIR/solutions/3/download_submissions.sh
$DIR/solutions/3/download_qm9.sh
$DIR/solutions/1/download_checkpoints.sh
$DIR/solutions/1/download_submissions.sh
$DIR/solutions/4/download_checkpoints.sh
$DIR/solutions/4/download_preprocessed_data.sh
$DIR/solutions/2/download_checkpoints.sh
# Download everything for analysis
$DIR/analysis/download_duplicate_submissions.sh
$DIR/analysis/download_submissions.sh
$DIR/analysis/download_preprocessed_data.sh

