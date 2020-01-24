#!/usr/bin/env bash

# Get location of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to data folder
cd $DIR/data

# Download kaggle_dataset.tar.gz
wget https://osf.io/46dpj/download

# Unpack
tar zxf kaggle_dataset.tar.gz

# Deleted unneeded files
rm potential* magnetic_shielding_tensors_test.csv mulliken_charges_test.csv
rm dipole* data_test.csv scalar_coupling_contributions_test.csv 
rm structures_train.csv structures_test.csv

# Rename to fit competition naming
mv magnetic_shielding_tensors_train.csv magnetic_shielding_tensors.csv
mv mulliken_charges_train.csv mulliken_charges.csv
mv data_train.csv train.csv
mv data_test_masked.csv test.csv
mv scalar_coupling_contributions_train.csv scalar_coupling_contributions.csv
