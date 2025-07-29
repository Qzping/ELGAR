import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

import json
import pickle
import numpy as np
from icecream import ic
import smplx
import torch
import glob
from data_process.postprocess import get_long_rot
from utils.rotation_conversions import quaternion_to_axis_angle, axis_angle_to_matrix, matrix_to_axis_angle
from torchmin import minimize
from data_process.data_manipulation import DataManipulatorCello as DMC


def elbow_torch_obj_func(elbow_rot, smplx_model, pose_body, betas, trans, root_orient, wrist_target):

    pose_body[:, -12:-6] = elbow_rot
    output = smplx_model(body_pose=pose_body, betas=betas, transl=trans, global_orient=root_orient)
    J_shaped = output.joints[:, :22]  # body
    wrist_result = J_shaped[:, -2:]

    dist = torch.linalg.norm(wrist_target - wrist_result)
    if dist < 1e-6:
        print("Dist below threshold, stopping early.")
        return 0
    return dist

smooth_rot = DMC.smooth_rot
if __name__ == '__main__':
    
    model_name = 'full'
    timestamp = '2025'
    song_name = 'your_song_name'
    save_npz_dir = f'./npz/generate/{model_name}'

    generated_motion_file = glob.glob(f"../save/{model_name}/{timestamp}/sample/*_{song_name}/*.npy")[0]
    ic(generated_motion_file)

    long_rot_dir = f'./genearted_rot/{model_name}/{song_name}/'

    long_rot_file = get_long_rot(generation_path=generated_motion_file, long_rot_dir=long_rot_dir)

    long_rot = np.load(long_rot_file, allow_pickle=True)

    lh_hand_quats = torch.tensor(long_rot['lh_motion'])
    rh_hand_quats = torch.tensor(long_rot['rh_motion'])
    body_quats = torch.tensor(long_rot['body_motion'])

    lh_hand_rots = quaternion_to_axis_angle(lh_hand_quats)
    rh_hand_rots = quaternion_to_axis_angle(rh_hand_quats)
    body_rots = quaternion_to_axis_angle(body_quats)

    piece_len = body_rots.shape[0]

    left_hand = lh_hand_rots.reshape(-1, 45)
    right_hand = rh_hand_rots.reshape(-1, 45)
    hands = np.concatenate((left_hand, right_hand), axis=1)

    body_mean_path = f'../data_process/ik_joints/results/body_mean.json'
    body_mean = json.load(open(body_mean_path, 'r'))

    body_root_orient = np.ones((piece_len, 3)) * np.array(body_mean['body_root_orient'])
    body_trans = np.ones((piece_len, 3)) * np.array(body_mean['body_trans'])


    body_betas = torch.tensor(body_mean['body_betas'])
    body_scale = torch.ones((piece_len, 1)) * torch.tensor(body_mean['body_scale'])
    body_root_orient = torch.tensor(body_root_orient)
    body_trans = torch.tensor(body_trans)
    assert hands.shape[0] == piece_len

    model_path = '../model/smplx/SMPLX_NEUTRAL.npz'
    model = smplx.create(model_path=model_path, model_type='smplx', batch_size=body_rots.shape[0], use_pca=False, flat_hand_mean=True)
    device = torch.device('cuda')
    model = model.to(device)

    # body_torch = torch.tensor(body_rots).to(device).float()
    # left_hand_torch = torch.tensor(left_hand).to(device).float()
    # right_hand_torch = torch.tensor(right_hand).to(device).float()
    body_torch = body_rots.clone().detach().to(device).float()
    left_hand_torch = left_hand.clone().detach().to(device).float()
    right_hand_torch = right_hand.clone().detach().to(device).float()
    body_root_orient = body_root_orient.clone().to(device).float()
    body_trans_torch = body_trans.clone().to(device).float()
    body_betas_torch = (torch.ones((piece_len, 10)) * torch.tensor(body_mean['body_betas'])).to(device).float()

    output = model(body_pose=body_torch, left_hand_pose=left_hand_torch, right_hand_pose=right_hand_torch,
                   global_orient=body_root_orient, return_full_pose=True, transl=body_trans_torch, betas=body_betas_torch)
    # full_pose = output.full_pose.detach().cpu().numpy()

    body_joints = output.joints[:, :22]  # body
    
    # smooth the body pose
    body_torch_matrix = axis_angle_to_matrix(body_torch.reshape(-1, 21, 3)).reshape(-1, 21, 3, 3).cpu()
    body_torch = matrix_to_axis_angle(torch.tensor(smooth_rot(body_torch_matrix, njoints=21, window_length=31),
                                                             dtype=torch.float)).reshape(-1, 63).to(device)

    # additional IK to correct the wrist position after the whole body smooth
    elbow_rot_vec = body_torch[:, -12:-6].clone()
    obj_func = lambda x: elbow_torch_obj_func(x, model, body_torch, body_betas_torch, body_trans_torch, body_root_orient, body_joints[:, -2:])
    elbow_result = minimize(obj_func, elbow_rot_vec,
                            method='bfgs',
                            options=dict(line_search='strong-wolfe'),
                            max_iter=50,
                            disp=2)
    
    output = model(body_pose=body_torch, left_hand_pose=left_hand_torch, right_hand_pose=right_hand_torch,
                   global_orient=body_root_orient, return_full_pose=True, transl=body_trans_torch, betas=body_betas_torch)
    full_pose = output.full_pose.detach().cpu().numpy()
    pose_body = body_torch.detach().cpu().numpy()


    os.makedirs(save_npz_dir, exist_ok=True)
    npz_file = os.path.join(save_npz_dir, rf'{song_name}.npz')

    ic(npz_file)

    np.savez(npz_file,
            gender='neutral',
            mocap_framerate=np.array(30.),
            poses=full_pose,
            pose_body=pose_body,
            pose_hand=hands,
            num_betas=np.array(10, dtype=int),
            trans=body_trans,
            betas=body_betas,
            )