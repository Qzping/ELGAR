import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import json
import pickle
import torch
import smplx
import numpy as np
from os import path as osp
from icecream import ic
from utils.rotation_conversions import axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_axis_angle
# from torchmin import minimize
from data_manipulation import DataManipulatorCello as DMC


def ik_get_wrist_pose(pose_body, root_orient, hand_pose, hand_type = 'left'):
    pose_body = pose_body.reshape(-1, 21, 3)
    pose_body_original_shape = pose_body.shape
    pose_body = axis_angle_to_matrix(pose_body)  # (bs, 21, 3, 3)
    R_accumulate = axis_angle_to_matrix(root_orient).clone() # (bs, 3, 3)

    wrist_global = rotation_6d_to_matrix(hand_pose.reshape(pose_body.shape[0], 16, 6)[:,0,:]) # (bs, 3, 3)

    if hand_type == 'left':
        R_index = [2, 5, 8, 12, 15, 17]
    else:
        R_index = [2, 5, 8, 13, 16, 18]

    for id in R_index:
        R_accumulate = torch.matmul(R_accumulate[:],pose_body[:, id])
    
    wrist_local = torch.matmul(torch.linalg.inv(R_accumulate), wrist_global)

    if hand_type == 'left':
        pose_body[:,19] = wrist_local
    else:
        pose_body[:,20] = wrist_local
    
    pose_body = matrix_to_axis_angle(pose_body).reshape(pose_body_original_shape) #(bs, 63)
    return pose_body


def elbow_obj_func(elbow_rot, smplx_model, pose_body, betas, trans, root_orient, wrist_target):

    pose_body_new = pose_body.copy()
    pose_body_new[-12:-6] = elbow_rot
    pose_body_tensor = torch.tensor(pose_body_new).unsqueeze(0)
    output = smplx_model(body_pose=pose_body_tensor, betas=betas, transl=trans, global_orient=root_orient)
    J_shaped = output.joints[0, :22]  # body
    wrist_result = J_shaped[-2:].detach().numpy()

    dist = np.sum((wrist_target - wrist_result)**2)
    if dist < 1e-6:
        print("Dist below threshold, stopping early.")
        return 0
    return dist

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
    parser = argparse.ArgumentParser(prog='ik_joints_training')
    parser.add_argument('--proj_dir', default='cello01', type=str, required=False)
    parser.add_argument('--human_kp_dir', default='trainning_ik', type=str, required=False)
    parser.add_argument('--motion_pkl_save_dir', default='results', type=str, required=False)
    parser.add_argument('--body_mean_dir', default='results', type=str, required=False)
    args = parser.parse_args()

    proj_dir = args.proj_dir
    human_kp_dir = args.human_kp_dir
    motion_pkl_save_dir = args.motion_pkl_save_dir
    body_mean_dir = args.body_mean_dir
    
    trainning_time = 3
    body_smpl_path = f'{motion_pkl_save_dir}/body_{proj_dir}_round_2.pkl'
    
    keypoints_path = osp.join(human_kp_dir, proj_dir+'.json')
    with open(keypoints_path, 'r') as f:
        iked = json.load(f)
    f.close()
    
    lh_pose = torch.tensor(iked['lh_pose'])
    rh_pose = torch.tensor(iked['rh_pose'])

    with open(f'{body_mean_dir}/body_mean.json', 'r') as f:
        body_mean = json.load(f)
    f.close()

    with open(body_smpl_path, 'rb') as f:
        body_dic = pickle.load(f)
    f.close()

    pose_body = body_dic['pose_body']
    betas = torch.ones_like(body_dic['betas'])*torch.tensor(body_mean['body_betas'])
    scaling = torch.ones_like(body_dic['scaling'])*torch.tensor(body_mean['body_scale']) * 1000
    trans = torch.ones_like(body_dic['trans'])*torch.tensor(body_mean['body_trans'])
    root_orient = torch.ones_like(body_dic['root_orient'])*torch.tensor(body_mean['body_root_orient'])
    frame_num = pose_body.shape[0]

    model_path = '../../model'
    model = smplx.create(model_path=model_path, model_type='smplx', use_pca=False, flat_hand_mean=True, batch_size=frame_num)
    output = model(body_pose=pose_body, betas=betas, transl=trans, global_orient=root_orient)
    # body_joints = output.joints[:, :22]  # body
    
    # # smooth the body pose
    # pose_body_matrix = axis_angle_to_matrix(pose_body.reshape(-1, 21, 3)).reshape(-1, 21, 3, 3)
    # pose_body = matrix_to_axis_angle(torch.tensor(smooth_rot(pose_body_matrix, njoints=21, window_length=31),
    #                                                          dtype=torch.float)).reshape(-1, 63)

    # # additional IK to correct the wrist position after the whole body smooth
    # elbow_rot_vec = pose_body[:, -12:-6].clone()
    # obj_func = lambda x: elbow_torch_obj_func(x, model, pose_body, betas, trans, root_orient, body_joints[:, -2:])
    # elbow_result = minimize(obj_func, elbow_rot_vec,
    #                         method='bfgs',
    #                         options=dict(line_search='strong-wolfe'),
    #                         max_iter=50,
    #                         disp=2)

    pose_body = ik_get_wrist_pose(pose_body, root_orient, lh_pose, hand_type='left')
    pose_body = ik_get_wrist_pose(pose_body, root_orient, rh_pose, hand_type='right')


    motion_file = os.path.join(motion_pkl_save_dir, f'body_{proj_dir}_calc.pkl')
    with open(motion_file, 'wb') as f:
        pickle.dump({
            'betas': betas.detach().cpu(),
            'scaling': scaling.detach().cpu(),
            'trans': trans.detach().cpu(),
            'pose_body': pose_body.detach().cpu(),
            'root_orient': root_orient.detach().cpu(),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
