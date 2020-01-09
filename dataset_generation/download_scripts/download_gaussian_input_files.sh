#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/../data

# Download xyz_files.tar.gz
wget https://osf.io/v8awm/download
