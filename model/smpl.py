import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import utils.rotation_conversions as geometry
import os


SMPL_PARENTS_INDICES = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def batch_rigid_transform(rot_mats, joints, parents):
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = torch.cat([F.pad(rot_mats.reshape(-1, 3, 3), [0, 0, 0, 1]),
                                F.pad(rel_joints.reshape(-1, 3, 1), [0, 0, 0, 1],
                                      value=1)], dim=2).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms

from icecream import ic
class SMPL:
    def __init__(self, **kwargs):
        self.rest_pose = json.load(open(f'{os.path.dirname(os.path.abspath(__file__))}/smpl_info/rest_pose.json'))
        self.rest_joints = torch.tensor(self.rest_pose['joints'])
        self.root_orient = torch.tensor(self.rest_pose['root_orient'])
        self.parents = torch.tensor(SMPL_PARENTS_INDICES)
    
    def get_joint_positions_batch(self, rotations):
        bs = rotations.shape[0]
        device = rotations.device
        parents = self.parents.to(device)
        rest_joints = self.rest_joints.unsqueeze(0).repeat(bs, 1, 1).to(device)
        root_orient_rot_vec = self.root_orient.unsqueeze(0).repeat(bs, 1).to(device)
        root_orient_rot_mat = geometry.axis_angle_to_matrix(root_orient_rot_vec)
        rotations = torch.cat((root_orient_rot_mat.view(bs, 1, 3, 3),
                               rotations.view(bs, -1, 3, 3),), dim=1)
        joints, _ = batch_rigid_transform(rotations, rest_joints, parents)
        return joints
