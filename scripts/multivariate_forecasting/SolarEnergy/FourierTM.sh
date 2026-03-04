export CUDA_VISIBLE_DEVICES=0
model_name=FourierTM
fourier_args="--use_asb 1 --tokenization fft --attention_mode dual_path --asb_n_bands 4"

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_FourierTM \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 0 \
  --wv "db8" \
  --m 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  $fourier_args

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_FourierTM \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.003 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 0 \
  --wv "db8" \
  --m 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 2 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  $fourier_args

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_FourierTM \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.003 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 0 \
  --wv "db8" \
  --m 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 2 \
  --alpha 0.1 \
  --l1_weight 0.005 \
  $fourier_args

python -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_FourierTM \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.009 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 0 \
  --wv "db8" \
  --m 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  $fourier_args
