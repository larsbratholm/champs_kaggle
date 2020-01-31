echo 'unzip input/quantum-machine-9-aka-qm9.zip'
unzip -qq -o input/quantum-machine-9-aka-qm9.zip -d input

echo 'Generating train_cp.csv and test_cp.csv...'
python code/generate_mute_cp.py

echo 'Generating structures_mu.csv...'
python code/generate_structes_df.py
