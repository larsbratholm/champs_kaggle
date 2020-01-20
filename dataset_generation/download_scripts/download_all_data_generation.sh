#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run qm9_exyz.tar.gz download script
$DIR/download_qm9_exyz.sh
# Run xyz_files.tar.gz download script
$DIR/download_xyz_files.sh
# Run gaussian_input_files.tar.gz download script
$DIR/download_gaussian_input_files.sh
# Run gaussian_output_files.tar.gz download script
$DIR/download_gaussian_output_files.sh
# Run kaggle_dataset.tar.gz download script
$DIR/download_kaggle_dataset.sh
