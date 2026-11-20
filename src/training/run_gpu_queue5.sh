#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
P=/dev/shm/saksham/ids_env/bin/python; S=/dev/shm/saksham/ids/v3/src
for s in 1 2; do $P -I $S/phase_c1_hier.py $s; done
for s in 1 2; do $P -I $S/phase_b1_catboost.py 34_class c1 none $s; $P -I $S/phase_b1_catboost.py 2_class c1 none $s; done
echo QUEUE5DONE
