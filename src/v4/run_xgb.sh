#!/bin/bash
# XGB queues pinned to idle GPUs (profile sets CUDA_VISIBLE_DEVICES=5, so override per queue)
PY=/dev/shm/saksham/ids_env/bin/python
S=/home/kishor_kumar/saksham/capstone-cloud/src/v4/train_v4.py
L=/dev/shm/saksham/v4/logs
for pair in "46 0" "38 7"; do set -- $pair
  ( export CUDA_VISIBLE_DEVICES=$2
    for t in 2 8 34; do for m in xgb_best xgb_unw; do
      $PY $S $m $t $1 0 > $L/${m}_${t}c_${1}f.log 2>&1
    done; done ) &
done
wait
echo XGB_DONE
