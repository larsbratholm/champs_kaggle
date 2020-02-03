show_usage() {
	echo -e "Usage: $0 [BATCH_SIZE] [SEED]"
	echo ""
}

if [ $# -lt 2 ]
then
	show_usage
	exit 1
fi

python train.py --batch_size $1 --seed $2 --nepochs 80
