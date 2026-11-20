#!/bin/bash
# v4 launcher: 6 RF on CPU in parallel, XGB queues on GPU0 (46f) and GPU7 (38f)
PY=/dev/shm/saksham/ids_env/bin/python
S=/home/kishor_kumar/saksham/capstone-cloud/src/v4/train_v4.py
L=/dev/shm/saksham/v4/logs
( for f in 46 38; do for t in 2 8 34; do
    OMP_NUM_THREADS=16 $PY $S rf $t $f > $L/rf_${t}c_${f}f.log 2>&1 &
  done; wait; done ) &   # 3 RFs at a time (48 cores) - shared machine
for pair in "46 0" "38 7"; do set -- $pair
  ( for t in 2 8 34; do for m in xgb_best xgb_unw; do
      $PY $S $m $t $1 $2 > $L/${m}_${t}c_${1}f.log 2>&1
    done; done ) &
done
wait
echo ALL_DONE
