#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to models folder
cd $DIR/models

# Download team_4_checkpoints_part_{1,2}.tar.gz
wget -O team_4_checkpoints_part_1.tar.gz https://osf.io/r958y/download
wget -O team_4_checkpoints_part_2.tar.gz https://osf.io/7ze2c/download

# Unpack
tar -zxf team_4_checkpoints_part_1.tar.gz
tar -zxf team_4_checkpoints_part_2.tar.gz

# Deleted unneeded files
rm team_4_checkpoints_part_{1,2}.tar.gz
