#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/../data

# Download qm9_exyz.tar.gz
wget -O qm9_exyz.tar.gz https://osf.io/m2gp8/download
