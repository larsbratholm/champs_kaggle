#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to models directory
cd $DIR/models

# Download team_2_checkpoints.tar
wget https://osf.io/3r4zv/download

# Unpack
tar xf team_2_checkpoints.tar

# Delete archive
rm team_2_checkpoints.tar
