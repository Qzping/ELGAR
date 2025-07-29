# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry

from model.mano import MANO
from model.smpl import SMPL
from icecream import ic

def normalize_vector(v):
        """
        v: (n, 3)
        """
        norms = torch.linalg.norm(v, axis=1, keepdims=True)
        # 防止除以0的情况，保留原值
        norms = torch.add(norms, 1e-8)
        # 对每一行进行归一化
        normalized_arr = v / norms
        return normalized_arr

class Rotation2xyz:
    def __init__(self):
        self.smpl_model = SMPL()
        self.lh_model = MANO('left')
        self.rh_model = MANO('right')

    def __call__(self, x, pose_rep, mask=None,
                 get_rotations_back=False, **kwargs):
                
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        # x: [bs, 52, nfeats, nframes]

        x_body_rot = x[:, :21]
        x_lh_rot = x[:, 21:36]
        x_rh_rot = x[:, 36:51]
        x_bow_vec = x[:, -1, :3]

        x_lh_rot = x_lh_rot.permute(0, 3, 1, 2)
        x_rh_rot = x_rh_rot.permute(0, 3, 1, 2)
        x_body_rot = x_body_rot.permute(0, 3, 1, 2)
        bs, nframes, _, _ = x_lh_rot.shape

        assert x_lh_rot.shape == x_rh_rot.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rot6d":
            lh_rot_mat = geometry.rotation_6d_to_matrix(x_lh_rot[mask])
            rh_rot_mat = geometry.rotation_6d_to_matrix(x_rh_rot[mask])
            body_rot_mat = geometry.rotation_6d_to_matrix(x_body_rot[mask])
        else:
            raise NotImplementedError("Not supported for this rotation representation.")

        # if betas is None:
        #     betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
        #                         dtype=rotations.dtype, device=rotations.device)
        #     betas[:, 1] = beta
        # out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
        
        # x_lh_trans = x_lh_trans.permute(0, 2, 1)  # [bs, nframes, 3]
        # x_lh_trans = x_lh_trans.reshape(-1, 3)  # [bs*nframes, 3]
        # x_rh_trans = x_rh_trans.permute(0, 2, 1)  # [bs, nframes, 3]
        # x_rh_trans = x_rh_trans.reshape(-1, 3)  # [bs*nframes, 3]

        body_joints = self.smpl_model.get_joint_positions_batch(body_rot_mat)
        body_root_orient = self.smpl_model.root_orient.clone().to(device=x.device)
        body_root_orient = body_root_orient.unsqueeze(0).repeat(bs*nframes, 1)
        body_root_orient_rot_mat = geometry.axis_angle_to_matrix(body_root_orient)
        lh_rot_mat_accumulate = body_root_orient_rot_mat.clone()
        rh_rot_mat_accumulate = body_root_orient_rot_mat.clone()

        lh_trans = body_joints[:, 20]
        rh_trans = body_joints[:, 21]

        lh_chain_index = [2, 5, 8, 12, 15, 17, 19]
        rh_chain_index = [2, 5, 8, 13, 16, 18, 20]

        for id in range(7):
            lh_index = lh_chain_index[id]
            lh_rot_mat_accumulate = torch.matmul(lh_rot_mat_accumulate, body_rot_mat[:, lh_index])
            rh_index = rh_chain_index[id]
            rh_rot_mat_accumulate = torch.matmul(rh_rot_mat_accumulate, body_rot_mat[:, rh_index])

        lh_wrist_global = lh_rot_mat_accumulate.contiguous().view(bs*nframes, 1, 3, 3)
        rh_wrist_global = rh_rot_mat_accumulate.contiguous().view(bs*nframes, 1, 3, 3)

        lh_rot_mat = torch.cat((lh_wrist_global, lh_rot_mat), dim=1)
        rh_rot_mat = torch.cat((rh_wrist_global, rh_rot_mat), dim=1)

        lh_joints = self.lh_model.get_joint_positions_batch(lh_trans, lh_rot_mat)  # [bs*nframes, njoints, 3]
        rh_joints = self.rh_model.get_joint_positions_batch(rh_trans, rh_rot_mat)  # [bs*nframes, njoints, 3]
        # x_bow = x_bow.permute(0, 2, 1).reshape(-1, 2, 3)  # [bs*nframes, 2, 3]
        # joints = torch.cat((lh_joints, rh_joints, x_bow), dim=1)  # [bs*nframes, 2njoints+2, 3]
        # x_bow = x_bow.permute(0, 2, 1).reshape(-1, 1, 3)  # [bs*nframes, 1, 3]
        # joints = torch.cat((lh_joints, rh_joints, x_bow), dim=1)  # [bs*nframes, 2njoints+1, 3]
        x_bow_vec = x_bow_vec.permute(0, 2, 1).reshape(-1, 3)  # [bs*nframes, 3]
        x_bow_vec = normalize_vector(x_bow_vec)
        hand_joints_bow = [5, 6, 11, 12, 14, 15]
        bow_start = torch.mean(rh_joints[:, hand_joints_bow], axis=1)
        bow_end = bow_start + x_bow_vec * 0.75  # fix length for bow
        bow_joints = torch.zeros((x_bow_vec.shape[0], 2, 3), device=rh_joints.device)
        bow_joints[:, 0, :] = bow_start
        bow_joints[:, 1, :] = bow_end
        
        # ic(lh_joints.shape)  # (bs*nframes, 21, 3)
        # ic(rh_joints.shape)  # (bs*nframes, 21, 3)
        # ic(bow_joints.shape)  # (bs*nframes, 2, 3)
        # ic(body_joints.shape)  # (bs*nframes, 22, 3)

        joints = torch.cat((lh_joints, rh_joints, bow_joints, body_joints), dim=1)  # [bs*nframes, 2njoints+2, 3]

        x_xyz = torch.empty(bs, nframes, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        if get_rotations_back:
            return x_xyz, x_lh_rot, x_rh_rot
        else:
            return x_xyz
