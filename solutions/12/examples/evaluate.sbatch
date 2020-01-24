#!/bin/sh

#SBATCH --job-name=eval_cormorant
#SBATCH --output=output/eval_cormorant_%A.out
#SBATCH --error=output/eval_cormorant_%A.err
#SBATCH --partition=gm4
##SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=pi-risi

module load Anaconda3/2018.12
source activate pt

nch=48
ncg=5
maxl=3
target=${1}
identifier=${2}
# lri=1e-3
# lrf=1e-5
lri=${3}
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

cd cormorant_${target}_continue_${identifier}_ncg_${ncg}_top_mlp_maxl_${maxl}_lr_${lri}_${lrf}_nc_${nch}
cp -r model model_copy_final

python ../evaluate_edge_cormorant_qs.py --target=${target} --batch-size=16 --num-cg-levels=${ncg} --top mlp --datadir=../data/ --maxl=${maxl} --max-sh=${maxl} --num-channels ${nch} --load True --lr-init=3e-6 --lr-final=3e-6 --predictdir predict_out_final --logdir log_out_final --modeldir model_copy_final

cd ..
