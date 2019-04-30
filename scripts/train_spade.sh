#!/bin/bash
python train.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c --gpu_ids 2,3,4,5,6,7,8 --batchSize 7 --tf_log  --rgb --continue_train --which_epoch 50 --niter 200
