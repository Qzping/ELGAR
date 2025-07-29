# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on audios.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_spd_data import get_dataset_loader

from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import logging
import time
from torch.utils.tensorboard import SummaryWriter
from diffusion import logger


def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    dist_util.setup_dist(args.device)

    curr_t = time.localtime()
    run_id = f'{curr_t.tm_year}' + f'{curr_t.tm_mon:02d}' +\
        f'{curr_t.tm_mday:02d}' + f'{curr_t.tm_hour:02d}' +\
        f'{curr_t.tm_min:02d}'
    run_id = int(run_id)
    run_dir = os.path.join(args.save_dir, str(run_id))  # create new path
    os.makedirs(run_dir, exist_ok=True)

    args_path = os.path.join(run_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # initialize logger
    logger.get_current(run_dir)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(run_dir)
    train_platform.report_args(args, name='Args')

    # print("creating data loader...")
    logger.log("creating data loader...")
    data = get_dataset_loader(batch_size=args.batch_size, datapath=args.datapath, filename=args.filename, cond_mode=args.cond_mode, train_mode=args.train_mode)
    logger.log("data loader created!")
    # print("data loader created!")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    # model.rot2xyz.smpl_model.eval()

    logger.log('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))

    logger.log("Training...")
    TrainLoop(args, train_platform, model, diffusion, data, run_dir).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
