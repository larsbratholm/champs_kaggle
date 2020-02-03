#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to input folder
cd $DIR/output

# Download team_3_checkpoints_part_{1,2,3}.tar.gz
wget https://osf.io/jdsc6/download
wget https://osf.io/t5gu8/download
wget https://osf.io/egv3a/download

# Unpack
tar -zxf team_3_checkpoints_part_1.tar.gz
tar -zxf team_3_checkpoints_part_2.tar.gz
tar -zxf team_3_checkpoints_part_3.tar.gz

# Deleted unneeded files
rm team_3_checkpoints_part_{1,2,3}.tar.gz
