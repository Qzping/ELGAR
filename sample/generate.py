# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_spd_data import get_dataset_loader
from icecream import ic
from diffusion import logger
import time


def main():
    args = generate_args()
    fixseed(args.seed)

    mname = args.model_path
    mname = mname.split("/")[-1]
    prefix = "model"
    suffix = ".pt"
    ckpt_it = mname[len(prefix):-len(suffix)]

    ckpt_it = int(ckpt_it)
    if ckpt_it >= 1000:
        divided = ckpt_it / 1000
        if divided.is_integer():
            ckpt_it = f"{int(divided)}k"
        else:
            ckpt_it = f"{divided:.1f}k"
    else:
        ckpt_it = str(ckpt_it)

    filename = args.filename

    if 'full' in filename:
        filename = filename['full']
    else:
        filename = filename[args.cond_mode]

    datapath = args.datapath
    fullpath = os.path.join(datapath, filename)

    if args.num_samples == 0:
        a = np.load(fullpath)
        args.num_samples = np.load(fullpath).shape[0]

    filename = os.path.basename(filename)
    filename = filename.split('.')[0]

    out_path = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), 'sample')
    curr_t = time.localtime()
    run_id = f'{curr_t.tm_year}' + f'{curr_t.tm_mon:02d}' + \
             f'{curr_t.tm_mday:02d}' + f'{curr_t.tm_hour:02d}' + \
             f'{curr_t.tm_min:02d}'
    run_id = int(run_id)
    out_path = os.path.join(out_path, str(run_id))  # create new path
    out_path = f"{out_path}_{filename}"
    os.makedirs(out_path, exist_ok=True)

    logger.get_current(out_path)

    max_frames = 150
    fps = 30
    n_frames = min(max_frames, int(args.motion_length * fps))
    is_using_data = True
    dist_util.setup_dist(args.device)

    if args.num_samples > args.batch_size:
        args.num_samples = args.batch_size
    else:
        args.batch_size = args.num_samples

    logger.log('Loading dataset...')

    data = get_dataset_loader(batch_size=args.batch_size, datapath=args.datapath, split='test', filename=args.filename,
                              shuffle=False, cond_mode=args.cond_mode)
    logger.log(f"Sample numeber: {args.num_samples}")
    total_num_samples = args.num_samples * args.num_repetitions

    logger.log("Creating model and diffusion...")
    args.timestep_respacing = 'ddim50'
    model, diffusion = create_model_and_diffusion(args, data)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    gparam = args.guidance_param
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler

    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        try:
            gt, model_kwargs = next(iterator)
        except StopIteration:
            raise Exception('Sample num is set larger than data length.')
    else:
        collate_args = [{'lh_pose': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples

    # print(model_kwargs)
    model_kwargs = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs.items()}
    unify_device(model_kwargs, device=dist_util.dev())

    if gt is not None:
        gt_translations = gt[:, -1, :3]
        gt_rotations = gt[:, :-1]

    for rep_i in range(args.num_repetitions):
        all_motions = []
        all_lengths = []
        all_instruments = []
        all_gt_rotations = []
        all_gt_trans = []

        all_lh_rot = []
        all_rh_rot = []
        all_body_rot = []

        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch, denoted by 's' from the sampling step in the paper
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        # sample_fn = diffusion.p_sample_loop
        sample_fn = diffusion.ddim_sample_loop

        sample = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            long=True
        )

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' or gt is None else model_kwargs['y']['mask'].reshape(
            args.batch_size, n_frames).bool()
        all_body_rot.append(sample[:, :21].cpu().numpy())
        all_lh_rot.append(sample[:, 21:36].cpu().numpy())
        all_rh_rot.append(sample[:, 36:51].cpu().numpy())
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        all_motions.append(sample.cpu().numpy())
        if gt is not None:
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
            # all_instruments.append(model_kwargs['y']['instrument'].cpu().numpy())
            all_gt_rotations.append(gt_rotations.permute(0, 3, 1, 2))
            all_gt_trans.append(gt_translations.permute(0, 2, 1))

        print(f"created {len(all_motions) * args.batch_size} samples")

        all_motions = np.concatenate(all_motions, axis=0)[:total_num_samples]  # [bs, njoints, 3, seqlen] xyz repr
        all_body_rot = np.concatenate(all_body_rot, axis=0)[:total_num_samples]
        all_lh_rot = np.concatenate(all_lh_rot, axis=0)[:total_num_samples]
        all_rh_rot = np.concatenate(all_rh_rot, axis=0)[:total_num_samples]
        print('all_motions:', all_motions.shape)
        # all_text = all_text[:total_num_samples]
        if gt is not None:
            all_gt_rotations = np.concatenate(all_gt_rotations, axis=0)[:total_num_samples]
            all_gt_trans = np.concatenate(all_gt_trans, axis=0)[:total_num_samples]
            all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
            # all_instruments = np.concatenate(all_instruments, axis=0)[:total_num_samples]

        # if os.path.exists(out_path):
        #     shutil.rmtree(out_path)
        # os.makedirs(out_path)

        fname = f"{filename}_{ckpt_it}_g{gparam}_rep{rep_i}.npy"

        npy_path = os.path.join(out_path, fname)
        logger.log(f"saving results file to [{npy_path}]")

        np.save(npy_path,
                {'motion': all_motions, 'lh_rot': all_lh_rot, 'rh_rot': all_rh_rot, 'body_rot': all_body_rot,
                 'lengths': all_lengths, 'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})

        # with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        #     fw.write('\n'.join([str(l) for l in all_lengths]))

        abs_path = os.path.abspath(out_path)
        print(f'[Done] Results are at [{abs_path}]')


def unify_device(nested_dict, device):
    """
    Unify device of elements in nested dictionary by recursion.
    """
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            unify_device(val, device)
        else:
            nested_dict[key] = val.to(device) if torch.is_tensor(val) else val


if __name__ == "__main__":
    main()
