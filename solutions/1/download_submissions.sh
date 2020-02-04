#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to submissions directory
cd $DIR/submissions

# Download team_1_submissions.tar.gz
wget https://osf.io/hecpw/download

# Unpack
tar zxf team_1_submissions.tar.gz

# Delete archive
rm team_1_submissions.tar.gz
