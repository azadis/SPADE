#!/bin/bash

#ade20k
# python test.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c_synth_127500 --rgb
# python test.py --name ade15c --dataset_mode ade20k --dataroot datasets/ADE15c --rgb
# --rgb

#cityscapes
python test.py --name cityscapes_noinstance --dataset_mode cityscapes --dataroot ../dynamo/datasets/cityscapes --no_instance
