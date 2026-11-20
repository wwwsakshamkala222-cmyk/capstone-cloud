#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
P="/dev/shm/saksham/ids_env/bin/python -I /dev/shm/saksham/ids/v3/src/phase_b1_catboost.py"
for t in 2_class 34_class; do for c in c1 c2; do for w in bal none; do $P $t $c $w 42; done; done; done
echo ALLDONE
