#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/data

# Download team_4_preprocessed_data.tar.gz
wget -O team_4_preprocessed_data.tar.gz https://osf.io/ntdhf/download

# Unpack
tar -zxf team_4_preprocessed_data.tar.gz

# Deleted unneeded files
rm team_4_preprocessed_data.tar.gz
