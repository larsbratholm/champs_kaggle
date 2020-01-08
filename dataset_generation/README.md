# dataset_generation
Information and scripts on generating the dataset from the QM9 structures.

[molecules_without_hydrogens.txt](./molecules_without_hydrogens.txt) contain the list of molecules that were excluded from the QM9 superset as they contained no hydrogens.

[data](./data) Contains data required to run the scripts as well as outputs from these scripts.
The folder is empty unless the files have been downloaded using the download scripts in [download_scripts](./download_scripts).

download scripts
convert script
xyz
gaussian input files

[molecules_with_outliers.txt](./molecules_with_outliers.txt) contain the list of molecules that were excluded from the QM9 superset as the scalar coupling constants contained one or more outliers.

The [outlier_structures](./outlier_structures) folder contains the structures listed in [molecules_with_outliers.txt](./molecules_with_outliers.txt) in xyz format.
