#!/bin/bash

#ade20k
# python test.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c_synth_127500 --rgb
# python test.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c --rgb
# --rgb

#cityscapes
# python test.py --name cityscapes_noinstance --dataset_mode cityscapes --dataroot ../SBGAN/datasets/cityscapes --no_instance


#cityscapes_full
python test.py --name cityscapes_full --dataset_mode cityscapes_full_weighted --dataroot ../SBGAN/datasets --no_instance

#ade_bedroom
# python test.py --name ade_bedroom --dataset_mode ade_bedroom --dataroot ../SBGAN/datasets/ade_bedroom --no_instance
