#!/bin/bash
# FourierTM Full: run on all 8 datasets
# Usage: nohup bash scripts/run_fouriertm_all.sh > fouriertm_all.log 2>&1 &
#
# FourierTM = ASB + FFT tokenization + DualPath attention (all 3 innovations)
# Hyperparams identical to SimpleTM paper for fair comparison.

echo "=========================================="
echo "FourierTM Full - All 8 Datasets"
echo "Start: $(date)"
echo "=========================================="

# ETT (small datasets, fast)
echo ""
echo ">>> ETTh1"
bash scripts/multivariate_forecasting/ETT/FourierTM_h1.sh

echo ""
echo ">>> ETTh2"
bash scripts/multivariate_forecasting/ETT/FourierTM_h2.sh

echo ""
echo ">>> ETTm1"
bash scripts/multivariate_forecasting/ETT/FourierTM_m1.sh

echo ""
echo ">>> ETTm2"
bash scripts/multivariate_forecasting/ETT/FourierTM_m2.sh

# Large datasets
echo ""
echo ">>> Weather"
bash scripts/multivariate_forecasting/Weather/FourierTM.sh

echo ""
echo ">>> ECL (321 channels)"
bash scripts/multivariate_forecasting/ECL/FourierTM.sh

echo ""
echo ">>> Traffic (862 channels)"
bash scripts/multivariate_forecasting/Traffic/FourierTM.sh

echo ""
echo ">>> Solar-Energy (137 channels)"
bash scripts/multivariate_forecasting/SolarEnergy/FourierTM.sh

echo ""
echo "=========================================="
echo "FourierTM Full - All Done!"
echo "End: $(date)"
echo "Results: result_long_term_forecast.txt"
echo "=========================================="
