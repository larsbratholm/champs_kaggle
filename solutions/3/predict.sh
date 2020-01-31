echo 'Reproducing [1234, 12345, 77777, 2017, 4242, 1020, 1021, other 4242] submission files ...'

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 1234 \
	--model models/BERT-based/v25_2_final_score/b1024_l8_mh8_h832_d0.0_ep159_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s1234.pt

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 12345 \
	--model models/BERT-based/v25_2_final_score/b1024_l8_mh8_h832_d0.0_ep169_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s12345.pt

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 77777 \
	--model models/BERT-based/v25_2_final_score/b1024_l8_mh8_h832_d0.0_ep179_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s77777.pt

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 2017 --nlayers 6 \
	--model models/BERT-based/v25_2_final_score/b736_l6_mh8_h832_d0.0_ep176_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s2017.pt

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 4242 --nlayers 6 \
	--model models/BERT-based/v25_2_final_score/b768_l6_mh8_h832_d0.0_ep173_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s4242.pt

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 1020 \
	--model models/BERT-based/v25_2_final_score/b512_l8_mh8_h832_d0.0_ep160_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s1020.pt

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 1021 \
	--model models/BERT-based/v25_2_final_score/b512_l8_mh8_h832_d0.0_ep156_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s1021.pt

CUDA_VISIBLE_DEVICES=0 python code/BERT-based/transformer_v25_2/train.py --eval --seed 4242 --nlayers 6 \
	--model models/BERT-based/v25_2_final_score/b768_l6_mh8_h832_d0.0_ep173_1JHC_1JHN_2JHC_2JHH_2JHN_3JHC_3JHH_3JHN_s4242.pt

echo 'Generating final_submission.csv ...'
python code/generate_final_submission.py
