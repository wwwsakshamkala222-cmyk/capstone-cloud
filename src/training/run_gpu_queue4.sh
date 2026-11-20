#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
P=/dev/shm/saksham/ids_env/bin/python; S=/dev/shm/saksham/ids/v3/src
until grep -q QUEUE3DONE /dev/shm/saksham/ids/v3/logs/gpu_queue3.log; do sleep 30; done
CB_BORDERS=1024 $P -I $S/phase_b1_catboost.py 34_class c2 none 42
echo QUEUE4DONE
