Below you can find a outline of how to reproduce our solution for the Predicting Molecular Properties competition.

## Archive contents
```
3rd_solution/
├── input/
│   └── champs-scalar-coupling/
├── model/
│   └── BERT-based/
├── code/
│   └── BERT-based/
├── prepare_data.sh
└── predict.sh 
```
- `model/BERT-based` : contains the pretrained models
- `code` : code to train models from scratch
- `predict.sh` : code to generate predictions from model binaries


### Requirements 
- Ubuntu 16.04 LTS
- Python 3.7.3
- pytorch 1.0.1

- You can use the `pip install -r requirements.txt` to install the necessary packages.

#### Hardware
- 4 x NVIDIA Tesla V100

# External data
Run the [download_qm9.sh](./download_qm9.sh) script to download and extract the QM9 dataset to the [input](./input) folder.
The outputs used to generate the final ensemble for the competition can be downloaded to the [output](./output) folder by running the [download_submissions.sh](./download_submissions.sh) script.
Pre-trained models can be downloaded to the [model](./model) directory by running the [download_checkpoints.sh](./download_checkpoints.sh) script.


## Quick Reproduce
We will generate a 'final_submission.csv' file submitted in this competition using a pre-trained models. 

First, we prepare data to use for inference using the following script.

```
.../3$ python prepare_data.py
```

Second, we generate final_submission.csv file using the following script.
```
.../3$ python predict.py
```

# Slow Reproduce

- We made 15 models using v25_2 (14) and v25_3 (1) codes.
- At first, we pre-trained 15 models using the "pre-training.sh" scripts in the code/BERT-based/transformer_v25_2/scripts/ or code/BERT-based/transformer_v25_3/scripts/ with various seeds. And, we did fine-tuned with dropout(0.0) for all models.
- We differed n_layers and epochs to obtain the diversity of models.
- All pre-training and fine-tunning commends are listed below..
- You can run the scripts or run command listed below in the diretory, code/BERT-based/transformer_v25_2/ or code/BERT-based/transformer_v25_3/

## V25_2 code (pre training)
- python train.py --batch_size 512 --seed 1234 --nepochs 90 
- python train.py --batch_size 512 --seed 12345 --nepochs 90
- python train.py --batch_size 1024 --seed 7 --nepochs 80
- python train.py --batch_size 1024 --seed 77 --nepochs 80
- python train.py --batch_size 1024 --seed 777 --nepochs 90
- python train.py --batch_size 1024 --seed 7777 --nepochs 90
- python train.py --batch_size 1024 --seed 77777 --nepochs 100
- python train.py --batch_size 328 --seed 2019 --nepochs 100 --nlayers 6
- python train.py --batch_size 328 --seed 2020 --nepochs 100 --nlayers 6
- python train.py --batch_size 768 --seed 42 --nepochs 90 --nlayers 6
- python train.py --batch_size 736 --seed 3 --nepochs 90 --nlayers 6
- python train.py --batch_size 384 --seed 8899 --nepochs 100 --nlayers 6
- python train.py --batch_size 384 --seed 8894 --nepochs 75
- python train.py --batch_size 512 --seed 8895 --nepochs 100

## V25_2 code (fine_tunning)

- python train.py --batch_size 512 --seed 1234 --nepochs 120  --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b512_l8_mh8_h832_d0.0_ep89_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s1234.pt
- python train.py --batch_size 512 --seed 12345 --nepochs 130 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b512_l8_mh8_h832_d0.0_ep89_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s12345.pt
- python train.py --batch_size 1024 --seed 7 --nepochs 130 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b1024_l8_mh8_h832_d0.0_ep79_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s7.pt
- python train.py --batch_size 1024 --seed 77 --nepochs 120 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b1024_l8_mh8_h832_d0.0_ep79_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s77.pt
- python train.py --batch_size 1024 --seed 777 --nepochs 120 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b1024_l8_mh8_h832_d0.0_ep89_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s777.pt
- python train.py --batch_size 1024 --seed 7777 --nepochs 120 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b1024_l8_mh8_h832_d0.0_ep89_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s7777.pt
- python train.py --batch_size 1024 --seed 77777 --nepochs 130 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b1024_l8_mh8_h832_d0.0_ep99_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s77777.pt
- python train.py --batch_size 328 --seed 2019 --nepochs 130 --nlayers 6 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b328_l6_mh8_h832_d0.0_ep99_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt
- python train.py --batch_size 328 --seed 2020 --nepochs 130 --nlayers 6 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b328_l6_mh8_h832_d0.0_ep99_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2020.pt
- python train.py --batch_size 768 --seed 42 --nepochs 120 --nlayers 6 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b768_l6_mh8_h832_d0.0_ep89_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s42.pt
- python train.py --batch_size 736 --seed 3 --nepochs 120 --nlayers 6 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b736_l6_mh8_h832_d0.0_ep89_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s3.pt
- python train.py --batch_size 384 --seed 8899 --nepochs 120 --nlayers 6 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b384_l8_mh8_h832_d0.0_ep89_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s8899.pt
- python train.py --batch_size 384 --seed 8894 --nepochs 100 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b384_l8_mh8_h832_d0.0_ep74_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s8894.pt
- python train.py --batch_size 512 --seed 8895 --nepochs 130 --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b512_l8_mh8_h832_d0.0_ep99_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s8895.pt

## V25_3 code
- python train.py --batch_size 768 --seed 2019 --nepochs 80

## V25_2 code (fine_tunning)
- python train.py --batch_size 768 --seed 2019 --nepochs 120  --dropout 0 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_3/b768_l8_mh8_h832_d0.0_ep79_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt

## Pseudo labeling (pre-training)
- With 15 predictions, we made 8 pseudo label files by combining 6~8 submissions with simple average. The 8 combinations are different each other.
- We chose 8 baseline models and did pre-training and fine-tunning with train + test(pseudo labeled).
- All the commends for pseudo-pre-training are listed below.

- python train.py --batch_size 1024 --dropout 0.0 --nepochs 140 --lr 4e-05  --wsteps 700 --seed 1234 --pseudo --pseudo_path pseudo_file_1.csv  --model ../../../models/BERT-based/v25_2/b512_l8_mh8_h832_d0.0_ep119_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s1234.pt
- python train.py --batch_size 1024 --dropout 0.0 --nepochs 150 --lr 4e-05  --wsteps 700 --seed 12345 --pseudo --pseudo_path pseudo_file_2.csv  --model ../../../models/BERT-based/v25_2/b512_l8_mh8_h832_d0.0_ep129_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s12345.pt
- python train.py --batch_size 1024 --dropout 0.0 --nepochs 150 --lr 4e-05  --wsteps 700 --seed 77777 --pseudo --pseudo_path pseudo_file_3.csv  --model ../../../models/BERT-based/v25_2/b1024_l8_mh8_h832_d0.0_ep129_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s77777.pt
- python train.py --batch_size 368 --dropout 0.0 --nepochs 150 --lr 4e-05  --wsteps 700 --seed 1020 --pseudo --pseudo_path pseudo_file_4.csv  --model ../../../models/BERT-based/v25_2/b328_l6_mh8_h832_d0.0_ep129_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt
- python train.py --batch_size 368 --dropout 0.0 --nepochs 150 --lr 4e-05  --wsteps 700 --seed 1021 --pseudo --pseudo_path pseudo_file_5.csv  --model ../../../models/BERT-based/v25_2/b328_l6_mh8_h832_d0.0_ep129_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2020.pt
- python train.py --batch_size 512 --dropout 0.0 --nepochs 140 --lr 4e-05  --wsteps 700 --seed 8895 --pseudo --pseudo_path pseudo_file_6.csv  --model ../../../models/BERT-based/v25_2/b768_l8_mh8_h832_d0.0_ep119_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt
- python train.py --batch_size 512 --dropout 0.0 --nepochs 140 --lr 4e-05  --wsteps 700 --seed 4242 --pseudo --pseudo_path pseudo_file_7.csv  --model ../../../models/BERT-based/v25_2/b768_l8_mh8_h832_d0.0_ep119_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt
- python train.py --batch_size 512 --dropout 0.0 --nepochs 140 --lr 4e-05  --wsteps 700 --seed 2017 --pseudo --pseudo_path pseudo_file_8.csv  --model ../../../models/BERT-based/v25_2/b768_l8_mh8_h832_d0.0_ep119_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt

## Pseudo labeling (fine-tunning)
- python train.py --batch_size 1024 --SEED 1234 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b512_l8_mh8_h832_d0.0_ep139_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s1234.pt
- python train.py --batch_size 1024 --SEED 12345 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b512_l8_mh8_h832_d0.0_ep149_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s12345.pt
- python train.py --batch_size 1024 --SEED 77777 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b1024_l8_mh8_h832_d0.0_ep149_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s77777.pt
- python train.py --batch_size 368 --SEED 1020 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b328_l6_mh8_h832_d0.0_ep149_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt
- python train.py --batch_size 368 --SEED 1021 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b328_l6_mh8_h832_d0.0_ep149_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2020.pt
- python train.py --batch_size 512 --SEED 8895 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b768_l8_mh8_h832_d0.0_ep139_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt
- python train.py --batch_size 512 --SEED 4242 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b768_l8_mh8_h832_d0.0_ep139_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt
- python train.py --batch_size 512 --SEED 2017 --dropout 0 --nepochs 180 --wsteps 700 --lr 4e-05 --model ../../../models/BERT-based/v25_2/b768_l8_mh8_h832_d0.0_ep139_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2019.pt

## Make our score
- Based on the CV scores, we ensemble by types. Unfortunately, CV scores of 5 models were over-fitted because of pseudo-labeling with different seeds. So, we used the CV scores of each baseline and the CV scores of other 3 models.
- The codes is in /code/type_ensemble.py. You can see the csv files which have all CV scores.
- With them, we got the our best score, -3.19498 (Final LB).
