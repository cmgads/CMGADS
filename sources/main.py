import torch

import os
import sys
import argparse
import random
import numpy as np
import logging
import time
from prettytable import PrettyTable

from args import add_args
from train import train
from pre_train import pre_train
import enums

logger = logging.getLogger(__name__)
def main(args):
    model = None
    tokenizer = None
    if args.do_pre_train: 
        model, tokenizer = pre_train(args=args)
    if args.do_fine_tune:
        train(args=args,
            trained_model=model,
            tokenizer=tokenizer)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.register()
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])
    add_args(parser)
    main_args = parser.parse_args()
    #define and make dirs
    main_args.output_root = os.path.join('..','outputs')
    # Root for outputs during pre-training
    main_args.pre_train_output_root = os.path.join(main_args.output_root, 'pre_train')
    # Root for saving checkpoints
    main_args.checkpoint_root = os.path.join(main_args.output_root, 'checkpoints')
    # Root for saving models
    main_args.model_root = os.path.join(main_args.output_root, 'models')
    # Root for saving vocabs
    main_args.vocab_root = os.path.join(main_args.output_root, 'vocabs')
    # Root for tensorboard
    main_args.tensor_board_root = os.path.join(main_args.output_root, 'runs')
    for d in [main_args.checkpoint_root, main_args.model_root, main_args.vocab_root, main_args.tensor_board_root,
              main_args.dataset_save_dir, main_args.vocab_save_dir, main_args.pre_train_output_root]:
        if not os.path.exists(d):
            os.makedirs(d)

    # cuda and parallel      
    if main_args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = main_args.cuda_visible_devices
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main_args.use_cuda = torch.cuda.is_available()
    main_args.parallel = torch.cuda.device_count() > 1

    # set random seed
    if main_args.random_seed > 0:
        random.seed(main_args.random_seed)
        np.random.seed(main_args.random_seed)
        torch.manual_seed(main_args.random_seed)
        torch.cuda.manual_seed_all(main_args.random_seed)

    # logging, log to both console and file, debug level only to file
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    #handler,console
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)


    #handler,log file
    file = logging.FileHandler(os.path.join(main_args.output_root, 'run.log'))
    file.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
    file.setFormatter(formatter)
    console.setFormatter(formatter)
    
    #add logger into handler
    logger.addHandler(console)
    logger.addHandler(file)
    
    main(main_args)



