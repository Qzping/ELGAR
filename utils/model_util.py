from model.pdm import PDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from icecream import ic

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = PDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):
    data_rep = 'rot6d'
    njoints = 21 + 15 + 15 + 1   # body + lh_joints + rh_joints + bow
    nfeats = 6

    
    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'translation': True, 
            'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True, 'latent_dim': args.latent_dim, 
            'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4, 'dropout': 0.1, 
            'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'audio_type': args.audio_type, 'cond_mask_prob': args.cond_mask_prob, 
            'arch': args.arch, 'emb_trans_dec': args.emb_trans_dec, 'dataset': args.dataset,
            'norm_first': args.norm_first, 'use_rotary': args.use_rotary}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = args.timestep_respacing
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X  # predict noise or signal itself
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_vel_rcxyz=args.lambda_vel_rcxyz,
        lambda_accel_rcxyz=args.lambda_accel_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_hicl = args.lambda_hicl,
        lambda_bicl = args.lambda_bicl
    )
