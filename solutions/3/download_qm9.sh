#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to input folder
cd $DIR/input

# Download dsgdb9nsd.xyz.tar.bz2
wget https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/3195389/dsgdb9nsd.xyz.tar.bz2

# Make output dir
mkdir quantum-machine-9-aka-qm9

# Unpack to quantum-machine-9-aka-qm9
tar -jxf dsgdb9nsd.xyz.tar.bz2 -C quantum-machine-9-aka-qm9

# Deleted unneeded files
rm dsgdb9nsd.xyz.tar.bz2
