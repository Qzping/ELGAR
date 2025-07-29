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

from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle


if __name__ == '__main__':

    train_data_dir = '../data_process/ik_joints/results'
    motion_file_dir = '../data_process/train_data/wholebody_normalized'
    save_npz_dir = './npz/train'

    piece_id = '01' # choose the piece

    body_file = os.path.join(train_data_dir, f'body_cello{piece_id}_calc.pkl')

    pkl = pickle.load(open(body_file, 'rb'))
    body_rots = pkl['pose_body'].reshape(-1, 63)
    piece_len = body_rots.shape[0]

    motion_file = os.path.join(motion_file_dir, rf'cello{piece_id}.json')
    motion_data = json.load(open(motion_file, 'r'))

    lh_hand_rots = torch.tensor(motion_data['lh_pose']).reshape(-1, 16, 6)
    lh_hand_rots = matrix_to_axis_angle(rotation_6d_to_matrix(lh_hand_rots)).numpy()
    rh_hand_rots = torch.tensor(motion_data['rh_pose']).reshape(-1, 16, 6)
    rh_hand_rots = matrix_to_axis_angle(rotation_6d_to_matrix(rh_hand_rots)).numpy()

    left_hand = lh_hand_rots[:, 1:].reshape(-1, 45)
    right_hand = rh_hand_rots[:, 1:].reshape(-1, 45)
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
    model = smplx.create(model_path=model_path, model_type='smplx', batch_size=body_rots.shape[0],
                         use_pca=False, flat_hand_mean=True)
    device = torch.device('cuda')
    model = model.to(device)

    body_torch = body_rots.clone().detach().to(device).float()
    left_hand_torch = torch.tensor(left_hand).to(device).float()
    right_hand_torch = torch.tensor(right_hand).to(device).float()
    body_root_orient = body_root_orient.clone().to(device).float()
    body_trans_torch = body_trans.clone().to(device).float()
    body_betas_torch = (torch.ones((piece_len, 10)) * torch.tensor(body_mean['body_betas'])).to(device).float()

    output = model(body_pose=body_torch, left_hand_pose=left_hand_torch, right_hand_pose=right_hand_torch,
                   global_orient=body_root_orient, return_full_pose=True, transl=body_trans_torch, betas=body_betas_torch)
    full_pose = output.full_pose.detach().cpu().numpy()
    joints = output.joints.detach().cpu().numpy()


    os.makedirs(save_npz_dir, exist_ok=True)
    npz_file = os.path.join(save_npz_dir, rf'cello{piece_id}.npz')

    # obj_file = os.path.join(save_npz_dir, rf'cello{piece_id}.obj')
    # with open(obj_file, 'w') as fp:
    #     for v in joints[0]:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    np.savez(npz_file,
             gender='neutral',
             mocap_framerate=np.array(30.),
             poses=full_pose,
             pose_body=body_rots,
             pose_hand=hands,
             num_betas=np.array(10, dtype=int),
             trans=body_trans,
             betas=body_betas,
             )
