#!/bin/bash
# FourierTM Phase 2: Ablation study on ETTh1 + ETTh2
#
# 5 variants x 2 datasets x 4 pred_lens x 3 itr = 120 runs
# SimpleTM baseline uses original code (--model SimpleTM)
#
# Ablation flags cheat sheet:
#   SimpleTM:       --model SimpleTM (original code, no flags needed)
#   +FFT only:      --model FourierTM --use_asb 0 --tokenization fft --attention_mode combined
#   +ASB only:      --model FourierTM --use_asb 1 --tokenization swt --attention_mode combined
#   +DualPath only: --model FourierTM --use_asb 0 --tokenization swt --attention_mode dual_path
#   FourierTM Full: --model FourierTM --use_asb 1 --tokenization fft --attention_mode dual_path

export CUDA_VISIBLE_DEVICES=0

# Common hyperparams per pred_len (from SimpleTM paper)
get_hparams() {
  local pred_len=$1
  case $pred_len in
    96)  echo "1 32 32 0.02 3 0.3 0.0005" ;;
    192) echo "1 32 32 0.02 3 1.0 5e-05" ;;
    336) echo "4 64 64 0.002 3 0.0 0.0" ;;
    720) echo "1 32 32 0.009 1 0.9 0.0005" ;;
  esac
}

run_variant() {
  local variant_name=$1
  local model=$2
  local use_asb=$3
  local tokenization=$4
  local attention_mode=$5
  local dataset=$6
  local data_path=$7

  echo ""
  echo "############################################################"
  echo "# Variant: $variant_name | Dataset: $dataset"
  echo "############################################################"

  for pred_len in 96 192 336 720; do
    read e_layers d_model d_ff lr m alpha l1 <<< $(get_hparams $pred_len)

    echo "--- $variant_name | $dataset | pred_len=$pred_len ---"

    if [ "$model" == "SimpleTM" ]; then
      python -u run.py \
        --is_training 1 --lradj TST --patience 3 \
        --root_path ./dataset/ETT-small/ --data_path $data_path \
        --model_id ${dataset}_${variant_name} --model SimpleTM --data $dataset \
        --features M --seq_len 96 --pred_len $pred_len \
        --e_layers $e_layers --d_model $d_model --d_ff $d_ff \
        --learning_rate $lr --batch_size 256 --fix_seed 2025 \
        --use_norm 1 --wv db1 --m $m --enc_in 7 --dec_in 7 --c_out 7 \
        --des Exp --itr 3 --alpha $alpha --l1_weight $l1
    else
      python -u run.py \
        --is_training 1 --lradj TST --patience 3 \
        --root_path ./dataset/ETT-small/ --data_path $data_path \
        --model_id ${dataset}_${variant_name} --model FourierTM --data $dataset \
        --features M --seq_len 96 --pred_len $pred_len \
        --e_layers $e_layers --d_model $d_model --d_ff $d_ff \
        --learning_rate $lr --batch_size 256 --fix_seed 2025 \
        --use_norm 1 --wv db1 --m $m --enc_in 7 --dec_in 7 --c_out 7 \
        --des Exp --itr 3 --alpha $alpha --l1_weight $l1 \
        --use_asb $use_asb --tokenization $tokenization --attention_mode $attention_mode \
        --asb_n_bands 4
    fi
  done
}

for dataset_info in "ETTh1 ETTh1.csv" "ETTh2 ETTh2.csv"; do
  read dataset data_path <<< $dataset_info

  # 1. SimpleTM baseline
  run_variant "SimpleTM"     SimpleTM  0 fft combined  $dataset $data_path

  # 2. +FFT only
  run_variant "FFT_only"     FourierTM 0 fft combined  $dataset $data_path

  # 3. +ASB only
  run_variant "ASB_only"     FourierTM 1 swt combined  $dataset $data_path

  # 4. +DualPath only
  run_variant "DualPath_only" FourierTM 0 swt dual_path $dataset $data_path

  # 5. FourierTM Full
  run_variant "FourierTM"    FourierTM 1 fft dual_path $dataset $data_path
done

echo ""
echo "=========================================="
echo "Ablation study complete!"
echo "Results in result_long_term_forecast.txt"
echo "=========================================="
