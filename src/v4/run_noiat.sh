#!/bin/bash
# No-IAT sensitivity runs: RF (CPU, 3 at a time) + xgb_best (GPU0: 46n, GPU7: 38n)
PY=/dev/shm/saksham/ids_env/bin/python
S=/home/kishor_kumar/saksham/capstone-cloud/src/v4/train_v4.py
L=/dev/shm/saksham/v4/logs
( for f in 46n 38n; do for t in 2 8 34; do
    OMP_NUM_THREADS=16 $PY $S rf $t $f > $L/rf_${t}c_${f}f.log 2>&1 &
  done; wait; done ) &
for pair in "46n 0" "38n 7"; do set -- $pair
  ( export CUDA_VISIBLE_DEVICES=$2
    for t in 2 8 34; do $PY $S xgb_best $t $1 0 > $L/xgb_best_${t}c_${1}f.log 2>&1; done ) &
done
wait
echo NOIAT_DONE
