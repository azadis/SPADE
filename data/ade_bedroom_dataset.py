"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os

class ADEBedroomDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=66)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        # with open('%s/ADE_indoor_lbl_info_%s.txt'%(root, phase),'r') as f:
        #     label_paths_all = f.readlines()

        label_dir = os.path.join(root, 'ADE_bedroom_%s_lbl'%phase)
        label_paths_all = make_dataset(label_dir, recursive=True)
        label_paths = [p for p in label_paths_all if p.endswith('.png')]
        # label_paths = [p.split(' ')[0].strip() for p in label_paths_all if p.split(' ')[0].endswith('.png')]

        # with open('%s/ADE_indoor_im_info_%s.txt'%(root, phase),'r') as f:
        #     image_paths_all = f.readlines()
        image_dir = os.path.join(root, 'ADE_bedroom_%s_im'%phase)
        image_paths_all = make_dataset(image_dir, recursive=True)
        image_paths = [im for im in image_paths_all if im.endswith('.jpg')]
        # image_paths = [im.split(' ')[0].strip() for im in image_paths_all if im.split(' ')[0].endswith('.jpg')]


        instance_paths = []  # don't use instance map for ade20k

        return label_paths, image_paths, instance_paths

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    # we ignored this for the indoor portion!
    def postprocess(self, input_dict):
        label = input_dict['label']