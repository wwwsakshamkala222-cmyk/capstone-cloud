#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
P=/dev/shm/saksham/ids_env/bin/python; S=/dev/shm/saksham/ids/v3/src
until grep -q QUEUE2DONE /dev/shm/saksham/ids/v3/logs/gpu_queue2.log; do sleep 30; done
$P -I $S/phase_b1_catboost.py 34_class c1 none 42
$P -I $S/phase_b1_catboost.py 34_class c2 bal 42
echo QUEUE3DONE
