#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/../data

# Download gaussian_input_files.tar.gz
wget -O gaussian_input_files.tar.gz https://osf.io/v8awm/download
