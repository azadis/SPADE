#!/bin/bash

#ade
# python train.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c --gpu_ids 2,3,4,5,6,7,8 --batchSize 7 --tf_log  --rgb --continue_train --which_epoch 50 --niter 200


#cityscpaes
	#res:256
# python train.py --name ft_cityscapes_noinstance --dataset_mode cityscapes --dataroot /mnt/disks/sazadi/segGAN/dynamo/datasets/cityscapes --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 24 --tf_log  --niter 100 --niter_decay 100 --no_instance \
# --load_size 512 --crop_size 512 --display_winsize 128 --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints # --continue_train --which_epoch 100

	#res:128
# python train.py --name cityscapes_noinstance_128 --dataset_mode cityscapes --dataroot /mnt/disks/sazadi/segGAN/dynamo/datasets/cityscapes --gpu_ids 0,1 --batchSize 24 --tf_log  --niter 100 --niter_decay 100 --no_instance \
# --load_size 256 --crop_size 256 --display_winsize 128 --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --no_html # --continue_train --which_epoch 100


#ade_indoor
	#res:256
# python train.py --name ade_indoor --dataset_mode ade_indoor --dataroot /mnt/disks/sazadi/segGAN/dynamo/datasets/ADE_indoor --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 48 --tf_log  --niter 150 --niter_decay 150 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints

	# res:128
# python train.py --name ade_indoor_128 --dataset_mode ade_indoor --dataroot /mnt/disks/sazadi/segGAN/dynamo/datasets/ADE_indoor --gpu_ids 2,3 --batchSize 48 --tf_log  --niter 150 --niter_decay 150 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --load_size 128 --crop_size 128 --no_html



#ade_bedroom
	#res:256
# python train.py --name ade_bedroom --dataset_mode ade_bedroom --dataroot /mnt/disks/sazadi/segGAN/dynamo/datasets/ADE_bedroom --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 48 --tf_log  --niter 300 --niter_decay 300 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints

	# res:128
# python train.py --name ade_bedroom_128 --dataset_mode ade_bedroom --dataroot /mnt/disks/sazadi/segGAN/dynamo/datasets/ADE_bedroom --gpu_ids 4,5 --batchSize 48 --tf_log  --niter 300 --niter_decay 300 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --load_size 128 --crop_size 128 --no_html


#cityscapes_full
	#res:256
# python train.py --name cityscapes_full --dataset_mode cityscapes_full_weighted --dataroot /home/sazadi/projects/segGAN/dynamo/datasets --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 32 --tf_log  --niter 30 --niter_decay 30 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --not_sort


	# res:128
python train.py --name cityscapes_full_128 --dataset_mode cityscapes_full_weighted --dataroot /home/sazadi/projects/segGAN/dynamo/datasets --gpu_ids 6,7 --batchSize 32 --tf_log  --niter 15 --niter_decay 15 --no_instance \
 --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --not_sort  --load_size 256 --crop_size 256 --no_html
