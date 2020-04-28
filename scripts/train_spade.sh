#!/bin/bash

#ade
# python train.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c --gpu_ids 2,3,4,5,6,7,8 --batchSize 7 --tf_log  --rgb --continue_train --which_epoch 50 --niter 200


#cityscpaes
	#res:256
# python train.py --name ft_cityscapes_noinstance --dataset_mode cityscapes --dataroot /mnt/disks/sazadi/segGAN/SBGAN/datasets/cityscapes --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 24 --tf_log  --niter 100 --niter_decay 100 --no_instance \
# --load_size 512 --crop_size 512 --display_winsize 128 --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints # --continue_train --which_epoch 100

	#res:128
# python train.py --name cityscapes_noinstance_128 --dataset_mode cityscapes --dataroot /mnt/disks/sazadi/segGAN/SBGAN/datasets/cityscapes --gpu_ids 0,1 --batchSize 24 --tf_log  --niter 100 --niter_decay 100 --no_instance \
# --load_size 256 --crop_size 256 --display_winsize 128 --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --no_html # --continue_train --which_epoch 100


#ade_indoor
	#res:256
# python train.py --name ade_indoor_2 --dataset_mode ade_indoor --dataroot /mnt/disks/sazadi/segGAN/SBGAN/datasets/ADE_indoor --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 48 --tf_log  --niter 150 --niter_decay 150 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --no_html

	# res:128
# python train.py --name ade_indoor_128 --dataset_mode ade_indoor --dataroot /mnt/disks/sazadi/segGAN/SBGAN/datasets/ADE_indoor --gpu_ids 2,3 --batchSize 48 --tf_log  --niter 150 --niter_decay 150 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --load_size 128 --crop_size 128 --no_html



#ade_bedroom
	#res:256
# python train.py --name ade_bedroom --dataset_mode ade_bedroom --dataroot /mnt/disks/sazadi/segGAN/SBGAN/datasets/ADE_bedroom --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 48 --tf_log  --niter 300 --niter_decay 300 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints

	# res:128
# python train.py --name ade_bedroom_128 --dataset_mode ade_bedroom --dataroot /mnt/disks/sazadi/segGAN/SBGAN/datasets/ADE_bedroom --gpu_ids 4,5 --batchSize 48 --tf_log  --niter 300 --niter_decay 300 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --load_size 128 --crop_size 128 --no_html


#cityscapes_full
	#res:256
# python train.py --name cityscapes_full --dataset_mode cityscapes_full_weighted --dataroot /home/sazadi/projects/segGAN/SBGAN/datasets --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 32 --tf_log  --niter 30 --niter_decay 30 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --not_sort


#cityscapes_10k
	#res:256
# python train.py --name cityscapes_10k --dataset_mode cityscapes_10k_weighted --dataroot /home/sazadi/projects/segGAN/SBGAN/datasets --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 32 --tf_log  --niter 80 --niter_decay 80 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --not_sort

#cityscapes_15k
	#res:256
# python train.py --name cityscapes_15k --dataset_mode cityscapes_15k_weighted --dataroot /home/sazadi/projects/segGAN/SBGAN/datasets --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 32 --tf_log  --niter 40 --niter_decay 40 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --not_sort --no_html


#cityscapes_15k
	#res:128
# python train.py --name cityscapes_15k_128 --dataset_mode cityscapes_15k_weighted --dataroot /home/sazadi/projects/segGAN/SBGAN/datasets --gpu_ids 8,9 --batchSize 28 --tf_log  --niter 20 --niter_decay 20 --no_instance \
#  --checkpoints_dir /shared/sazadi/data1/segGAN/SPADE/checkpoints --not_sort --no_html --load_size 256 --crop_size 256

#cityscapes_10k
	#res:128
python train.py --name cityscapes_10k_128 --dataset_mode cityscapes_10k_weighted --dataroot /home/sazadi/projects/segGAN/SBGAN/datasets --gpu_ids 1,2 --batchSize 28 --tf_log  --niter 30 --niter_decay 30 --no_instance \
 --checkpoints_dir /shared/sazadi/data1/segGAN/SPADE/checkpoints --not_sort --no_html --load_size 256 --crop_size 256

	# res:128
# python train.py --name cityscapes_full_128 --dataset_mode cityscapes_full_weighted --dataroot /home/sazadi/projects/segGAN/SBGAN/datasets --gpu_ids 6,7 --batchSize 32 --tf_log  --niter 15 --niter_decay 15 --no_instance \
#  --checkpoints_dir /mnt/disks/sazadi/segGAN/SPADE/checkpoints --not_sort  --load_size 256 --crop_size 256 --no_html
