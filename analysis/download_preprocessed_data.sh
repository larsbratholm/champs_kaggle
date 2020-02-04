#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/data

# Download preprocessed_data_part_{1,2}
wget https://osf.io/d8n7s/download
wget https://osf.io/3xgbp/download

# Join
cat preprocessed_data_part_{1,2} > data.pkl

# Deleted unneeded files
rm preprocessed_data_part_{1,2}
