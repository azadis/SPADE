#!/bin/bash

#ade
# python train.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c --gpu_ids 2,3,4,5,6,7,8 --batchSize 7 --tf_log  --rgb --continue_train --which_epoch 50 --niter 200


#cityscpaes
# python train.py --name cityscapes_noinstance_64x128 --dataset_mode cityscapes --dataroot ../dynamo/datasets/cityscapes --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 24 --tf_log  --niter 100 --niter_decay 100 --no_instance \
# --load_size 128 --crop_size 128 --display_winsize 128
# --continue_train


#ade_indoor
python train.py --name ade_indoor --dataset_mode ade_indoor --dataroot ../dynamo/datasets/ADE_indoor --gpu_ids 2,3 --batchSize 4 --tf_log  --niter 150 --niter_decay 150 --no_instance \
