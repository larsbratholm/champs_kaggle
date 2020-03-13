#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Download duplicates.tar.gz
wget -O duplicates.tar.gz https://osf.io/9tqs2/download

# Unpack
tar zxf duplicates.tar.gz

# Deleted unneeded files
rm duplicates.tar.gz
