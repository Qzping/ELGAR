import argparse
import json
import pickle
from typing import Union
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from torch import nn
from human_body_prior.models.ik_engine import unify_body_mappings
from human_body_prior.models.ik_engine import IK_Engine
from os import path as osp
from icecream import ic
from sklearn.metrics import mean_squared_error #root_mean_squared_error
from scipy import signal

from data_manipulation import DataManipulatorCello as DMC
smooth_loc = DMC.smooth_loc
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle


class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 n_joints: int=22,
                 kpts_colors: Union[np.ndarray, None] = None,
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, num_betas=10, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []#self.bm.f
        self.n_joints = n_joints
        self.kpts_colors = np.array([Color('grey').rgb for _ in range(n_joints)]) if kpts_colors == None else kpts_colors

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {'source_kpts':new_body.Jtr[:,:self.n_joints], 'body': new_body, 'verts': new_body.v}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ik_joints_optimizing')
    parser.add_argument('--proj_dir', default='cello01', type=str, required=False)
    parser.add_argument('--instrument', default='cello', type=str, required=False)
    parser.add_argument('--use_vposer', default=True, type=int, required=False)
    parser.add_argument('--only_fit_lh', default=False, type=int, required=False)
    parser.add_argument('--bm_fname', default='data/smpl_models/SMPLX_NEUTRAL.npz', type=str, required=False)
    parser.add_argument('--vposer', default='data/V02_05', type=str, required=False)
    parser.add_argument('--human_kp_dir', default='noaudio_body_iked_v1', type=str, required=False)
    parser.add_argument('--motion_pkl_save_dir', default='results', type=str, required=False)
    parser.add_argument('--cuda_device', default='0', type=str, required=False)
    parser.add_argument('--body_mean_dir', default='results', type=str, required=False)
    args = parser.parse_args()

    proj_dir = args.proj_dir
    instrument = args.instrument
    use_vposer = args.use_vposer
    only_fit_lh = args.only_fit_lh
    bm_fname = args.bm_fname
    vposer_expr_dir = args.vposer
    human_kp_dir = args.human_kp_dir
    motion_pkl_save_dir = args.motion_pkl_save_dir
    body_mean_dir = args.body_mean_dir
    if args.cuda_device == 'cpu':
        cuda_device = args.cuda_device
    else:
        cuda_device = int(args.cuda_device)
    
    if use_vposer and not only_fit_lh:
        optimizing_time = 1
        body_smpl_path = None
        optimizer_args = {'type':'LBFGS', 'max_iter':10000, 'lr':1, 'tolerance_change': 1e-3, 'history_size':200}
    elif not use_vposer and not only_fit_lh:
        optimizing_time = 2
        body_smpl_path = f'{motion_pkl_save_dir}/body_{proj_dir}_round_1.pkl'
        optimizer_args = {'type':'LBFGS', 'max_iter':10000, 'lr':5e-1, 'tolerance_change': 1e-6, 'history_size':200}
    
    if not use_vposer:
        fixed_paramters = True
    else:
        fixed_paramters = False
    
    keypoints_path = osp.join(human_kp_dir, proj_dir+'.json')
    with open(keypoints_path, 'r') as f:
        iked = json.load(f)
    f.close()

    coco_body = np.asarray(iked['body'])
    batch_size = coco_body.shape[0]
    coco_full_data = np.zeros((batch_size,133,3))

    coco_full_data[:, 0:5] = smooth_loc(loc_3d=coco_body[:, 0:5], sf=30) # face
    coco_full_data[:, 5:7] = smooth_loc(loc_3d=coco_body[:, 5:7], sf=120) # shoulder
    coco_full_data[:, 7] = coco_body[:, 7] # left arm
    coco_full_data[:, 8] = smooth_loc(loc_3d=coco_body[:, 8], sf=15) # right arm
    coco_full_data[:, 9] = coco_body[:, 9] # left wrist
    coco_full_data[:, 10] = smooth_loc(loc_3d=coco_body[:, 10], sf=15) # right wrist
    coco_full_data[:, 11:15] = smooth_loc(loc_3d=coco_body[:, 11:15], sf=200) # body and knees
    coco_full_data[:, 15:23] = smooth_loc(loc_3d=coco_body[:, 15:23], sf=45) # feet

    if fixed_paramters:
        lh_pose = np.asarray(iked['lh_pose'])
        lh_trans = np.asarray(iked['lh_trans'])
        rh_pose = np.asarray(iked['rh_pose'])
        rh_trans = np.asarray(iked['rh_trans'])
        used_finger_idx = np.asarray(iked['used_finger_idx'])

        dmc = DMC()
        
        mano_file_appendix = 'SMPLX'
        mano_lh = dmc.rot2position_hand(lh_trans, lh_pose, 'left', mano_file_appendix)
        mano_rh = dmc.rot2position_hand(rh_trans, rh_pose, 'right', mano_file_appendix)

        coco_full_data[:, 91:112] = mano_lh # mano left hand
        coco_full_data[:, 112:133] = mano_rh # mano right hand
        
    ratio_to_mm = 1000
    coco_full_data *= ratio_to_mm

    comp_device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    # SMPLX has 144 joints with contours, elif 76 with extra points, else 55
    n_joints_max_id= 144 #66


    red = Color("red")
    blue = Color("blue")
    kpts_colors = [c.rgb for c in list(red.range_to(blue, n_joints_max_id))]

    # create source and target key points and make sure they are index aligned
    data_loss = torch.nn.MSELoss(reduction='mean')

    stepwise_weights = [
        {'data': 0.5, 'poZ_body': .001, 'betas': 0.01, 'vel': 0.25},
                        ]

    if fixed_paramters:
        ic(proj_dir)
        
        body_dic = pickle.load(open(body_smpl_path, 'rb'))
        pose_body = body_dic['pose_body'].to(comp_device)

        with open(f'{body_mean_dir}/body_mean.json','r') as f:
            body_mean = json.load(f)
        f.close()

        betas = torch.ones_like(body_dic['betas'])*torch.tensor(body_mean['body_betas'])
        betas = betas.to(comp_device)

        scaling = torch.ones_like(body_dic['scaling'])*torch.tensor(body_mean['body_scale'])
        if torch.mean(scaling) < 1000:
            scaling *= 1000
        scaling = scaling.to(comp_device)

        trans = torch.ones_like(body_dic['trans'])*torch.tensor(body_mean['body_trans'])
        trans = trans.to(comp_device)

        root_orient = torch.ones_like(body_dic['root_orient'])*torch.tensor(body_mean['body_root_orient'])
        root_orient = root_orient.to(comp_device)

        lh_pose = lh_pose.reshape(batch_size, 16, 6)[:,1:,:]
        rh_pose = rh_pose.reshape(batch_size, 16, 6)[:,1:,:]

        lh_vec = matrix_to_axis_angle(rotation_6d_to_matrix(torch.as_tensor(lh_pose).float()))
        rh_vec = matrix_to_axis_angle(rotation_6d_to_matrix(torch.as_tensor(rh_pose).float()))

        pose_hand = torch.cat((lh_vec,rh_vec),axis=1).to(comp_device)
        
        target_pts = torch.tensor(coco_full_data[:batch_size, :, :].reshape([batch_size, -1, 3])).to(comp_device)
        
        use_vposer = (torch.ones(1)*(use_vposer and not only_fit_lh)).to(comp_device)

        ik_initial_body_params = {'betas':betas,'trans':trans,'scaling':scaling,'root_orient':root_orient,'pose_body':pose_body,'pose_hand':pose_hand,'used_finger_idx':used_finger_idx,'use_vposer':use_vposer}
    else:
        target_pts = torch.tensor(coco_full_data[:batch_size, 0:23, :].reshape([batch_size, 23, 3])).to(comp_device)
        ik_initial_body_params = {}
        
    
    source_pts = SourceKeyPoints(bm=bm_fname, n_joints=n_joints_max_id, kpts_colors=kpts_colors).to(comp_device) #, pose_body=pose_body

    ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                          verbosity=2,
                          display_rc= (2, 2),
                          data_loss=data_loss,
                          stepwise_weights=stepwise_weights,
                          optimizer_args=optimizer_args).to(comp_device)
    
    ik_res = ik_engine.forward(source_pts, target_pts, initial_body_params=ik_initial_body_params)

    ik_res_detached = {k: v.detach() for k, v in ik_res['free_vars'].items()}

    
    if 'scaling' in ik_res_detached.keys():
        scaling = ik_res_detached['scaling']
    else:
        scaling = ik_res['static_vars']['scaling']

    if 'betas' in ik_res_detached.keys():
        betas = ik_res_detached['betas']
    else:
        betas = ik_res['static_vars']['betas']

    if 'trans' in ik_res_detached.keys():
        nan_mask = torch.isnan(ik_res_detached['trans']).sum(-1) != 0
        if nan_mask.sum() != 0: 
            raise ValueError('Sum results were NaN!')
        trans = ik_res_detached['trans']
    else:
        trans = ik_res['static_vars']['trans']
    
    if 'root_orient' in ik_res_detached.keys():
        root_orient = ik_res_detached['root_orient']
    else:
        root_orient = ik_res['static_vars']['root_orient']

    pose_body = ik_res_detached['pose_body']
    

    result_pts = (source_pts(ik_res_detached)['source_kpts']*scaling[:, np.newaxis])[:, unify_body_mappings(dataset='smplx'), :]
    target_pts = target_pts.detach().cpu().numpy()[:, unify_body_mappings(dataset='coco'), :]
    result_pts = result_pts.detach().cpu().numpy()

    rmse = []
    for i in range(batch_size):
        #rmse.append(root_mean_squared_error(target_pts[i], result_pts[i]))
        rmse.append(np.sqrt(mean_squared_error(target_pts[i], result_pts[i])))
    
    os.makedirs(motion_pkl_save_dir, exist_ok=True)
    motion_file = os.path.join(motion_pkl_save_dir, f'body_{proj_dir}_round_{optimizing_time}.pkl')
    with open(motion_file, 'wb') as f:
        pickle.dump({
            'betas': betas.detach().cpu(),
            'scaling': scaling.detach().cpu(),
            'trans': trans.detach().cpu(),
            'pose_body': pose_body.detach().cpu(),
            'root_orient': root_orient.detach().cpu(),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)