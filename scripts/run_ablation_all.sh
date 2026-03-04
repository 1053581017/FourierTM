#!/bin/bash
# FourierTM Ablation Study: 5 variants x 8 datasets
# Usage: nohup bash scripts/run_ablation_all.sh > ablation_all.log 2>&1 &
#
# Variants:
#   1. SimpleTM (baseline)
#   2. +FFT only:      use_asb=0, tokenization=fft, attention_mode=combined
#   3. +ASB only:      use_asb=1, tokenization=swt, attention_mode=combined
#   4. +DualPath only: use_asb=0, tokenization=swt, attention_mode=dual_path
#   5. FourierTM Full: use_asb=1, tokenization=fft, attention_mode=dual_path

export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Helper: run a single experiment
# ============================================================
run_one() {
  local variant=$1; shift
  local model=$1; shift
  local fourier_flags="$1"; shift
  # remaining args are the per-config settings
  local extra="$@"

  local model_id_suffix=""
  [ "$variant" != "SimpleTM" ] && model_id_suffix="_${variant}"

  if [ "$model" == "SimpleTM" ]; then
    python -u run.py --model SimpleTM $extra
  else
    python -u run.py --model FourierTM $fourier_flags --asb_n_bands 4 $extra
  fi
}

# ============================================================
# Run all 5 variants for a given dataset config
# ============================================================
run_5variants() {
  local dataset_id=$1; shift
  local common="$@"

  echo ""
  echo "############################################################"
  echo "# Dataset: $dataset_id"
  echo "############################################################"

  echo "--- [1/5] SimpleTM baseline ---"
  python -u run.py --model SimpleTM --model_id ${dataset_id}_SimpleTM $common

  echo "--- [2/5] +FFT only ---"
  python -u run.py --model FourierTM --model_id ${dataset_id}_FFT_only \
    --use_asb 0 --tokenization fft --attention_mode combined --asb_n_bands 4 $common

  echo "--- [3/5] +ASB only ---"
  python -u run.py --model FourierTM --model_id ${dataset_id}_ASB_only \
    --use_asb 1 --tokenization swt --attention_mode combined --asb_n_bands 4 $common

  echo "--- [4/5] +DualPath only ---"
  python -u run.py --model FourierTM --model_id ${dataset_id}_DualPath_only \
    --use_asb 0 --tokenization swt --attention_mode dual_path --asb_n_bands 4 $common

  echo "--- [5/5] FourierTM Full ---"
  python -u run.py --model FourierTM --model_id ${dataset_id}_FourierTM \
    --use_asb 1 --tokenization fft --attention_mode dual_path --asb_n_bands 4 $common
}

echo "=========================================="
echo "Ablation Study - All 8 Datasets"
echo "Start: $(date)"
echo "=========================================="

# ============================================================
# ETTh1 (4 pred_lens)
# ============================================================
common_etth1="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --batch_size 256 --fix_seed 2025 --use_norm 1 --wv db1 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 3"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.02  --m 3 --alpha 0.3 --l1_weight 0.0005" ;;
    192) hp="--pred_len 192 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.02  --m 3 --alpha 1.0 --l1_weight 5e-05" ;;
    336) hp="--pred_len 336 --e_layers 4 --d_model 64 --d_ff 64 --learning_rate 0.002 --m 3 --alpha 0.0 --l1_weight 0.0" ;;
    720) hp="--pred_len 720 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.009 --m 1 --alpha 0.9 --l1_weight 0.0005" ;;
  esac
  run_5variants "ETTh1_pl${pl}" "$common_etth1 $hp"
done

# ============================================================
# ETTh2
# ============================================================
common_etth2="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --features M --seq_len 96 --batch_size 256 --fix_seed 2025 --use_norm 1 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 3"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.006 --wv bior3.1 --m 1 --alpha 0.1 --l1_weight 0.0005" ;;
    192) hp="--pred_len 192 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.006 --wv db1     --m 1 --alpha 0.1 --l1_weight 0.005" ;;
    336) hp="--pred_len 336 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.003 --wv db1     --m 1 --alpha 0.9 --l1_weight 0.0" ;;
    720) hp="--pred_len 720 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.003 --wv db1     --m 1 --alpha 1.0 --l1_weight 5e-05" ;;
  esac
  run_5variants "ETTh2_pl${pl}" "$common_etth2 $hp"
done

# ============================================================
# ETTm1
# ============================================================
common_ettm1="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --features M --seq_len 96 --batch_size 256 --fix_seed 2025 --use_norm 1 --wv db1 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 3"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.02 --m 3 --alpha 0.1 --l1_weight 0.005" ;;
    192) hp="--pred_len 192 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.02 --m 3 --alpha 0.1 --l1_weight 0.005" ;;
    336) hp="--pred_len 336 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.02 --m 1 --alpha 0.1 --l1_weight 0.005" ;;
    720) hp="--pred_len 720 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.02 --m 3 --alpha 0.1 --l1_weight 0.005" ;;
  esac
  run_5variants "ETTm1_pl${pl}" "$common_ettm1 $hp"
done

# ============================================================
# ETTm2
# ============================================================
common_ettm2="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --fix_seed 2025 --use_norm 1 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 3"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.006 --batch_size 256 --wv bior3.1 --m 3 --alpha 0.3 --l1_weight 0.0005" ;;
    192) hp="--pred_len 192 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.006 --batch_size 256 --wv bior3.1 --m 1 --alpha 0.0 --l1_weight 0.005" ;;
    336) hp="--pred_len 336 --e_layers 1 --d_model 64 --d_ff 64 --learning_rate 0.006 --batch_size 128 --wv bior3.3 --m 1 --alpha 0.6 --l1_weight 5e-5" ;;
    720) hp="--pred_len 720 --e_layers 1 --d_model 96 --d_ff 96 --learning_rate 0.003 --batch_size 256 --wv db1     --m 3 --alpha 1.0 --l1_weight 0.0" ;;
  esac
  run_5variants "ETTm2_pl${pl}" "$common_ettm2 $hp"
done

# ============================================================
# Weather
# ============================================================
common_weather="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/weather/ --data_path weather.csv --data custom --features M --seq_len 96 --batch_size 256 --fix_seed 2025 --use_norm 1 --wv db4 --enc_in 21 --dec_in 21 --c_out 21 --des Exp --itr 3"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --e_layers 4 --d_model 32 --d_ff 32 --learning_rate 0.01  --m 1 --alpha 0.3 --l1_weight 5e-05" ;;
    192) hp="--pred_len 192 --e_layers 4 --d_model 32 --d_ff 32 --learning_rate 0.009 --m 1 --alpha 0.3 --l1_weight 0.0" ;;
    336) hp="--pred_len 336 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.009 --m 3 --alpha 1.0 --l1_weight 5e-05" ;;
    720) hp="--pred_len 720 --e_layers 1 --d_model 32 --d_ff 32 --learning_rate 0.02  --m 1 --alpha 0.9 --l1_weight 0.005" ;;
  esac
  run_5variants "Weather_pl${pl}" "$common_weather $hp"
done

# ============================================================
# ECL (321 channels)
# ============================================================
common_ecl="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/electricity/ --data_path electricity.csv --data custom --features M --seq_len 96 --batch_size 256 --fix_seed 2025 --use_norm 1 --wv db1 --m 3 --enc_in 321 --dec_in 321 --c_out 321 --des Exp --itr 3 --d_model 256 --d_ff 1024"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --learning_rate 0.01  --alpha 0.0 --l1_weight 0.0" ;;
    192) hp="--pred_len 192 --learning_rate 0.006 --alpha 0.0 --l1_weight 0.0" ;;
    336) hp="--pred_len 336 --learning_rate 0.006 --alpha 0.0 --l1_weight 5e-5" ;;
    720) hp="--pred_len 720 --learning_rate 0.006 --alpha 0.0 --l1_weight 5e-5" ;;
  esac
  run_5variants "ECL_pl${pl}" "$common_ecl $hp"
done

# ============================================================
# Traffic (862 channels) - itr=1 due to cost
# ============================================================
common_traffic="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/traffic/ --data_path traffic.csv --data custom --features M --seq_len 96 --fix_seed 2025 --use_norm 1 --wv db1 --enc_in 862 --dec_in 862 --c_out 862 --des Exp --itr 1"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --e_layers 2 --d_model 512  --d_ff 1024 --learning_rate 0.003  --batch_size 24 --m 3 --alpha 0.1 --l1_weight 0.0" ;;
    192) hp="--pred_len 192 --e_layers 1 --d_model 1024 --d_ff 2048 --learning_rate 0.0005 --batch_size 32 --m 1 --alpha 0.1 --l1_weight 0.0" ;;
    336) hp="--pred_len 336 --e_layers 1 --d_model 1024 --d_ff 2048 --learning_rate 0.0005 --batch_size 32 --m 1 --alpha 0.1 --l1_weight 0.0" ;;
    720) hp="--pred_len 720 --e_layers 1 --d_model 1024 --d_ff 2048 --learning_rate 0.0005 --batch_size 32 --m 1 --alpha 0.1 --l1_weight 0.0" ;;
  esac
  run_5variants "Traffic_pl${pl}" "$common_traffic $hp"
done

# ============================================================
# Solar-Energy (137 channels)
# ============================================================
common_solar="--is_training 1 --lradj TST --patience 3 --root_path ./dataset/solar/ --data_path solar_AL.txt --data Solar --features M --seq_len 96 --batch_size 256 --fix_seed 2025 --use_norm 0 --wv db8 --enc_in 137 --dec_in 137 --c_out 137 --des Exp --d_model 128 --d_ff 256"

for pl in 96 192 336 720; do
  case $pl in
    96)  hp="--pred_len 96  --learning_rate 0.01  --m 3 --alpha 0.0 --l1_weight 0.005 --itr 3" ;;
    192) hp="--pred_len 192 --learning_rate 0.003 --m 1 --alpha 0.0 --l1_weight 0.005 --itr 2" ;;
    336) hp="--pred_len 336 --learning_rate 0.003 --m 1 --alpha 0.1 --l1_weight 0.005 --itr 2" ;;
    720) hp="--pred_len 720 --learning_rate 0.009 --m 1 --alpha 0.0 --l1_weight 0.005 --itr 3" ;;
  esac
  run_5variants "Solar_pl${pl}" "$common_solar $hp"
done

echo ""
echo "=========================================="
echo "Ablation Study Complete!"
echo "End: $(date)"
echo "Results: result_long_term_forecast.txt"
echo "=========================================="
