show_usage() {
	echo -e "Usage: $0 [BATCH_SIZE] [SEED] [PSEUDO_FILE_NAME] [MODEL_WEIGHT]"
	echo ""
}

if [ $# -lt 3 ]
then
	show_usage
	exit 1
fi


python train.py --batch_size $1 --dropout 0.0 --nepochs 120 --lr 4e-05  --wsteps 700 --seed $2  --pseudo --pseudo_path $3  --model $4
