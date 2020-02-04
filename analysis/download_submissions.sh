#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/data

# Download submissions_part_{1..4}.tar.gz
wget https://osf.io/9tqs2/download
wget https://osf.io/tb5ev/download
wget https://osf.io/4q2tg/download
wget https://osf.io/e5tbu/download

# Unpack
tar zxf submissions_part_1.tar.gz
tar zxf submissions_part_2.tar.gz
tar zxf submissions_part_3.tar.gz
tar zxf submissions_part_4.tar.gz

# Deleted unneeded files
rm submissions_part_{1..4}.tar.gz
