show_usage() {
	echo -e "Usage: $0 [BATCH_SIZE] [SEED] [MODEL_PATH]"
	echo ""
}

if [ $# -lt 3 ]
then
	show_usage
	exit 1
fi

 python train.py \
 --batch_size $1 \
 --seed $2 \
 --dropout 0 \
 --nepochs 100 \
 --wsteps 700 \
 --lr 4e-05 \
 --model $3
