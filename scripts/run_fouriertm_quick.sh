#!/bin/bash
# FourierTM Phase 1: Quick validation on ETTh1 + ETTh2
# Single run (itr=1) per config, ~2-3 min each
# Total: 8 runs, ~20 min

export CUDA_VISIBLE_DEVICES=0
model_name=FourierTM

echo "=========================================="
echo "FourierTM Quick Validation - Phase 1"
echo "=========================================="

# --- ETTh1 ---
for pred_len in 96 192 336 720; do
  if [ "$pred_len" == "336" ]; then
    e_layers=4; d_model=64; d_ff=64; lr=0.002; m=3; alpha=0.0; l1=0.0
  elif [ "$pred_len" == "720" ]; then
    e_layers=1; d_model=32; d_ff=32; lr=0.009; m=1; alpha=0.9; l1=0.0005
  elif [ "$pred_len" == "192" ]; then
    e_layers=1; d_model=32; d_ff=32; lr=0.02; m=3; alpha=1.0; l1=5e-05
  else
    e_layers=1; d_model=32; d_ff=32; lr=0.02; m=3; alpha=0.3; l1=0.0005
  fi

  echo "--- ETTh1 pred_len=$pred_len ---"
  python -u run.py \
    --is_training 1 --lradj TST --patience 3 \
    --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --model_id ETTh1 --model $model_name --data ETTh1 \
    --features M --seq_len 96 --pred_len $pred_len \
    --e_layers $e_layers --d_model $d_model --d_ff $d_ff \
    --learning_rate $lr --batch_size 256 --fix_seed 2025 \
    --use_norm 1 --m $m --enc_in 7 --dec_in 7 --c_out 7 \
    --des Exp --itr 1 --alpha $alpha --l1_weight $l1 \
    --asb_n_bands 4
done

# --- ETTh2 ---
for pred_len in 96 192 336 720; do
  if [ "$pred_len" == "336" ]; then
    e_layers=4; d_model=64; d_ff=64; lr=0.002; m=3; alpha=0.0; l1=0.0
  elif [ "$pred_len" == "720" ]; then
    e_layers=1; d_model=32; d_ff=32; lr=0.009; m=1; alpha=0.9; l1=0.0005
  elif [ "$pred_len" == "192" ]; then
    e_layers=1; d_model=32; d_ff=32; lr=0.02; m=3; alpha=1.0; l1=5e-05
  else
    e_layers=1; d_model=32; d_ff=32; lr=0.02; m=3; alpha=0.3; l1=0.0005
  fi

  echo "--- ETTh2 pred_len=$pred_len ---"
  python -u run.py \
    --is_training 1 --lradj TST --patience 3 \
    --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
    --model_id ETTh2 --model $model_name --data ETTh2 \
    --features M --seq_len 96 --pred_len $pred_len \
    --e_layers $e_layers --d_model $d_model --d_ff $d_ff \
    --learning_rate $lr --batch_size 256 --fix_seed 2025 \
    --use_norm 1 --m $m --enc_in 7 --dec_in 7 --c_out 7 \
    --des Exp --itr 1 --alpha $alpha --l1_weight $l1 \
    --asb_n_bands 4
done

echo "=========================================="
echo "Quick validation complete!"
echo "Results saved in result_long_term_forecast.txt"
echo "=========================================="
