#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Download everything for dataset generation
$DIR/dataset_generation/download_scripts/download_all_data_generation.sh
