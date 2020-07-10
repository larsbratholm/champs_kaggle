#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/data

# Download preprocessed_data_part_{1,2}
wget -O preprocessed_data_part_1 https://osf.io/sw8yh/download
wget -O preprocessed_data_part_2 https://osf.io/32kcw/download

# Join
cat preprocessed_data_part_{1,2} > data.pkl

# Deleted unneeded files
rm preprocessed_data_part_{1,2}
