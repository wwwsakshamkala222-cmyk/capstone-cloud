#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
PY="/dev/shm/saksham/ids_env/bin/python -I /dev/shm/saksham/ids/v3/src/cache_probs.py"
$PY 34_class xgb_34_class_balanced_p050.json val,test
$PY 34_class lgbm_34_class_cfg3_weighted.txt val,test
$PY 34_class xgb_34_class_cfg3_weighted.json val,test
$PY 34_class lgbm_34_class_cfg2.txt val,test
$PY 34_class catboost_34_class_balanced.cbm val,test
$PY 8_class xgb_8_class_cfg3_weighted.json val,test
$PY 2_class lgbm_2_class_cfg2.txt val,test
$PY 2_class xgb_2_class_cfg2.json val,test
$PY 2_class lgbm_2_class_cfg1.txt val,test
echo ALLDONE
