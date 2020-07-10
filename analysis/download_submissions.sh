#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/data

# Download submissions_part_{1..4}.tar.gz
wget -O submissions_part_1 https://osf.io/zk2dv/download
wget -O submissions_part_2 https://osf.io/uwn5x/download
wget -O submissions_part_3 https://osf.io/vbn7s/download

# Unpack
cat submissions_part_1 submissions_part_2 submissions_part_3 > submissions.tgz
tar zxf submissions.tgz

# Deleted unneeded files
rm submissions_part_{1..3}
rm submissions.tgz
