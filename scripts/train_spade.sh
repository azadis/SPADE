#!/bin/bash
python train.py --name ade20k --dataset_mode ade20k --dataroot datasets/ADE15c --rgb --gpu_ids 2,3,4,5,6,7,8 --batchSize 7 --tf_log
