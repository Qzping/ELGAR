import json
import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import pickle
import numpy as np
from icecream import ic
import torch
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R
from data_process.ik_joints.vis.pose_3d_view import visualize_3d_whole_combination
from data_manipulation import DataManipulatorCello as DMC
import smplx
import argparse


smooth_loc = DMC.smooth_loc
smooth_rot = DMC.smooth_rot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='vis_ik_results')
    parser.add_argument('--instrument', default='cello', type=str, required=False)
    parser.add_argument('--proj_first', default=1, type=int, required=True)
    parser.add_argument('--proj_last', default=85, type=int, required=True)
    
    args = parser.parse_args()

    instrument = args.instrument
    proj_first = args.proj_first
    proj_last = args.proj_last

    instrument = 'cello'
    proj_range = range(proj_first, proj_last+1)
    # proj_range = [1]
    
    proj_dirs = [instrument+'0'*(2-len(str(i)))+str(i) for i in proj_range]
    
    ###### for fbx conversion ######
    save_pkl = 0
    ###### for fbx conversion ######

    for proj_dir in proj_dirs:
        # Load *calc.pkl
        directory = '../results'
        body_smpl_path = os.path.join(directory, f'body_{proj_dir}_calc.pkl')

        body_mean_dir = directory
        model_path = '../../../model'
        normalized_file_path = f'../../train_data/wholebody_normalized/{proj_dir}.json'

        if not os.path.exists(normalized_file_path):
            print(f'{proj_dir} NOT exist!')
            continue

        with open(normalized_file_path, 'r') as f:
            normalized = json.load(f)
        f.close()

        lh_hand_rots = torch.tensor(normalized['lh_pose']).reshape(-1, 16, 6)
        lh_hand_rots = matrix_to_axis_angle(rotation_6d_to_matrix(lh_hand_rots))
        rh_hand_rots = torch.tensor(normalized['rh_pose']).reshape(-1, 16, 6)
        rh_hand_rots = matrix_to_axis_angle(rotation_6d_to_matrix(rh_hand_rots))
        
        left_hand = lh_hand_rots[:, 1:].reshape(-1, 45)
        right_hand = rh_hand_rots[:, 1:].reshape(-1, 45)
        
        body_dic = pickle.load(open(body_smpl_path, 'rb'))
        body_poses = body_dic['pose_body']
        body_scale = body_dic['scaling'] / 1000.  # mm to m
        body_trans = body_dic['trans']
        body_root_orient = body_dic['root_orient']
        
        # Smooth filtering
        # body_poses_matrix = axis_angle_to_matrix(body_poses).reshape(-1, 21, 3, 3)
        # smoothed_body_poses = matrix_to_axis_angle(torch.tensor(smooth_rot(body_poses_matrix, njoints=21, window_length=31),
        #                                                          dtype=torch.float))

        smoothed_body_poses = body_poses
        
        body_betas = torch.ones_like(body_dic['betas'])*(body_dic['betas'])
        body_trans = torch.ones_like(body_dic['trans'])*(body_dic['trans'].mean(axis=0))

        body_root_orient = torch.ones_like(body_dic['root_orient'])*(body_dic['root_orient'].mean(axis=0))
        
        # Load the data of "body_mean"
        with open(f'{body_mean_dir}/body_mean.json', 'r') as f:
            body_mean = json.load(f)
        f.close()

        body_betas = torch.ones((body_root_orient.shape[0],10))*torch.tensor(body_mean['body_betas'])
        body_trans = torch.ones((body_root_orient.shape[0],3))*torch.tensor(body_mean['body_trans'])
        body_scale = torch.ones((body_root_orient.shape[0],1))*torch.tensor( body_mean['body_scale'] )
        body_root_orient = torch.ones((body_root_orient.shape[0],3))*torch.tensor(body_mean['body_root_orient'])
        
        ###### for fbx conversion ######
        if save_pkl:
            motion_file = f'{proj_dir}_smooth.pkl'
            with open(motion_file, 'wb') as f:
                pickle.dump({
                    'betas': torch.tensor(body_betas).clone().requires_grad_(False).cpu(),
                    'scaling': torch.tensor(body_scale).clone().requires_grad_(False).cpu(),
                    'trans': torch.tensor(body_trans).clone().requires_grad_(False).cpu(),
                    'pose_body': torch.tensor(smoothed_body_poses).clone().requires_grad_(False).cpu(),
                    'root_orient': torch.tensor(body_root_orient).clone().requires_grad_(False).cpu(),
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        ###### for fbx conversion ######
        
        smoothed_body_poses = smoothed_body_poses.reshape(-1, 63)
        
        # Build a SMPLX layer
        model = smplx.create(model_path=model_path, model_type='smplx', use_pca=False, flat_hand_mean=True, batch_size=smoothed_body_poses.shape[0])
        output = model(body_pose=smoothed_body_poses, global_orient=body_root_orient, betas=body_betas, transl=body_trans, left_hand_pose=left_hand, right_hand_pose=right_hand)
        joints = output.joints

        pdist = torch.nn.PairwiseDistance(p=2)

        hand_joints = joints[:, 25:55]
        more_hand_joints = joints[:, 66:76]

        lh_pose = np.asarray(normalized['lh_pose'])
        lh_trans = np.asarray(normalized['lh_trans'])
        rh_pose = np.asarray(normalized['rh_pose'])
        rh_trans = np.asarray(normalized['rh_trans'])

        coco_wholebody_body = np.asarray(normalized['body'])
        coco_wholebody_joints = np.zeros((coco_wholebody_body.shape[0], 133, 3))
        
        dmc = DMC()

        mano_lh = dmc.rot2position_hand(lh_trans, lh_pose, hand_type='left')
        mano_rh = dmc.rot2position_hand(rh_trans, rh_pose, hand_type='right')

        coco_wholebody_joints[:, 0:5] = smooth_loc(loc_3d=coco_wholebody_body[:, 0:5], sf=30) #face
        coco_wholebody_joints[:, 5:7] = smooth_loc(loc_3d=coco_wholebody_body[:, 5:7], sf=120) #shoulder
        coco_wholebody_joints[:, 7] = coco_wholebody_body[:, 7] # left arm
        coco_wholebody_joints[:, 8] = smooth_loc(loc_3d=coco_wholebody_body[:, 8], sf=15) # right arm
        coco_wholebody_joints[:, 9] = coco_wholebody_body[:, 9] # left wrist
        coco_wholebody_joints[:, 10] = smooth_loc(loc_3d=coco_wholebody_body[:, 10], sf=15) # right wrist
        coco_wholebody_joints[:, 11:15] = smooth_loc(loc_3d=coco_wholebody_body[:, 11:15], sf=200) # body and knees
        coco_wholebody_joints[:, 15:23] = smooth_loc(loc_3d=coco_wholebody_body[:, 15:23], sf=45) # feet

        coco_wholebody_joints[:, 91:112] = mano_lh # left hand
        coco_wholebody_joints[:, 112:133] = mano_rh # right hand
        
        joints = joints.detach().numpy() * body_scale.detach().numpy().mean()
        
        data = {'coco_wholebody':coco_wholebody_joints , 'smpl':joints}
        
        # Visual result in the $proj_path
        visualize_3d_whole_combination(data, proj_path=proj_dir)
    