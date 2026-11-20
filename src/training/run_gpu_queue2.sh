#!/bin/bash
# waits for B1 chain, then: B1 reruns (2-class fixed + 34 c1_bal fixed), C1 hier s42, B2 focal s42
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
P=/dev/shm/saksham/ids_env/bin/python; S=/dev/shm/saksham/ids/v3/src
until grep -q ALLDONE /dev/shm/saksham/ids/v3/logs/b1_catboost.log; do sleep 30; done
for c in c1 c2; do for w in bal none; do $P -I $S/phase_b1_catboost.py 2_class $c $w 42; done; done
$P -I $S/phase_b1_catboost.py 34_class c1 bal 42
$P -I $S/phase_c1_hier.py 42
$P -I $S/phase_b2_focal.py 42
echo QUEUE2DONE
