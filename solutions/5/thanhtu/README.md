This folder contains materials to reproduce the submissions files that contributed to the final solution

1. Hardware 
- CPU: 12 cores
- Memory: 32 GB
- GPU: 1 GTX 2080 Ti

2. Software:
- CUDA 10.0 CuDNN 7.4
- Anaconda Python 3.7 2019.03
- pytorch==1.1.0
- pytorch_geometric
- openbabel, rdkit

3. Training: 

+> Generate data graph for every molecule: python data/data.py --data_dir=../../data/ --split_dir=./split/ --graph_dir=./graph/

Note: Need to hardcode these paths in data/dataset.py

+> training: 
first model: python train.py --out_dir=kaggle/output --model=model1 --optim=adam
second model: python train.py --out_dir=kaggle/output --model=model2 --optim=adam
third model: python train.py --out_dir=kaggle/output --model=model1 --optim=ranger

3. Make prediction:
python submit.py --out_dir=kaggle/output/submit --model=model1 --checkpoint=kaggle/output/checkpoints/checkpoint_01.pth
