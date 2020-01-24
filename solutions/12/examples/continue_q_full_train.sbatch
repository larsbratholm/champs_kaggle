#!/bin/sh

#SBATCH --job-name=j1_cormorant
#SBATCH --output=output/j1_cormorant_%A_%a.out
#SBATCH --error=output/j1_cormorant_%A_%a.err
#SBATCH --partition=gm4
##SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=pi-risi
#SBATCH --array=48

module load Anaconda3/2018.12
source activate pt

nch=${SLURM_ARRAY_TASK_ID}
ncg=5
maxl=3
target=${1}


lri=3e-4
lrf=3e-6
echo "----------------------------"
echo "ARGUMENTS"
echo "target" ${target}
echo "num channels" ${nch}
echo "num layers" ${ncg}
echo "maxl" ${maxl}
echo 'initial lr' ${lri}
echo 'final lr' ${lrf}
echo "----------------------------"

cp -r cormorant_${target}_continue_q_zero_ncg_${ncg}_top_mlp_maxl_${maxl}_nc_${nch}/ cormorant_${target}_continue_full_train_focus_3_q_zero_ncg_${ncg}_top_mlp_maxl_${maxl}_lr_${lri}_${lrf}_nc_${nch}/
cd cormorant_${target}_continue_full_train_focus_3_q_zero_ncg_${ncg}_top_mlp_maxl_${maxl}_lr_${lri}_${lrf}_nc_${nch}

python ../continue_edge_cormorant_zero_full_train.py --target=${target} --batch-size=16 --num-cg-levels=${ncg} --top mlp  --datadir=../data/ --num-epoch 200 --maxl=${maxl} --max-sh=${maxl} --num-channels ${nch} --lr-init=${lri} --lr-final=${lrf} --test=False --load=True

cd ..
