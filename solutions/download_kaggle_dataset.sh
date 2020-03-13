#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/data

# Download kaggle_dataset.tar.gz
wget -O kaggle_dataset.tar.gz https://osf.io/46dpj/download
# Download xyz_files.tar.gz
wget -O xyz_files.tar.gz https://osf.io/xp3a5/download

# Unpack
tar zxf kaggle_dataset.tar.gz

# Deleted unneeded files
rm kaggle_dataset.tar.gz
rm potential* magnetic_shielding_tensors_test.csv mulliken_charges_test.csv
rm dipole* data_test.csv scalar_coupling_contributions_test.csv 
rm structures_train.csv structures_test.csv

# Rename to fit competition naming
mv magnetic_shielding_tensors_train.csv magnetic_shielding_tensors.csv
mv mulliken_charges_train.csv mulliken_charges.csv
mv data_train.csv train.csv
mv data_test_masked.csv test.csv
mv scalar_coupling_contributions_train.csv scalar_coupling_contributions.csv

# Create directory with all xyz files
mkdir xyz
cd xyz
mv ../xyz_files.tar.gz .
tar zxf xyz_files.tar.gz
rm xyz_files.tar.gz
