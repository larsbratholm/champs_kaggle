# data
Data files needed to run the scripts from the parent folder, as well as output from these scripts.
The files in the folder can be downloaded from the accompanying download scripts.
[download_all_data_generation.sh](../download_scripts/download_all_data_generation.sh) will download all of the below files.
[download_everything.sh](../../download_everything.sh) will download everything related to this project, including the below files.

`qm9_exyz.tar.gz` can be downloaded by executing [download_qm9_exyz.sh](../download_scripts/download_qm9_exyz.sh) and contains all the 130,831 extended xyz files of QM9 of molecules that passed the geometry consistency check of the original QM9 paper.

`xyz_files.tar.gz` can be downloaded by executing [download_xyz_files.sh](../download_scripts/download_xyz_files.sh) and contains an archive of all the xyz files generated by running the [convert_exyz.py](../convert_exyz.py) script.

`gaussian_input_files.tar.gz` can be downloaded by executing [download_gaussian_input_files.sh](../download_scripts/download_gaussian_input_files.sh) and contains an archive of all the Gaussian input files for NMR computations generated by running the [convert_exyz.py](../convert_exyz.py) script.