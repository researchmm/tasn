"""
Adapted from https://github.com/phunterlau/iNaturalist.git

by Heliang Zheng
03/28/2019

"""
import random
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import fit, evaluate
import common.cub_data as data
import mxnet as mx
import numpy as np
import os, urllib
import model


if __name__ == "__main__":

    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "2"
    random.seed(0)
    np.random.seed(0)
    mx.random.seed(0)
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 1)
    parser.set_defaults(image_shape = '3,512,512', num_epochs=300,
                        lr=0.1, lr_step_epochs='100,200', wd=0, mom=0)

    args = parser.parse_args()
    batch_size_per_gpu = np.int(args.batch_size/len(args.gpus.replace(',','')))
    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    prefix = 'model/resnet-50'
    epoch = 0
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)


    (new_sym, new_args, new_auxs) = model.tasn(
        sym, arg_params, aux_params, args.num_classes, batch_size_per_gpu)     
    
    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_custom_iter,
            arg_params  = new_args,
            aux_params  = new_auxs,
            label_names = (['att_net_label', 'part_net_label', 'master_net_label', 'part_net_aux_label', 'master_net_aux_label']),
            eval_metric = evaluate.Multi_Accuracy(num=6))
