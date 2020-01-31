#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to input folder
cd $DIR/output

# Download team_3_submissions.tar.gz
wget https://osf.io/tny2v/download

# Unpack
tar -zxf team_3_submissions.tar.gz

# Deleted unneeded files
rm team_3_submissions.tar.gz
