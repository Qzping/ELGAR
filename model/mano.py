import numpy as np
import torch.nn as nn
from utils.misc import to_numpy, to_torch
import torch


MANO_PARENTS_INDICES = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]
BONE_LENGTHS = np.array([0.094, 0.0335, 0.024, 0.094, 0.034, 0.03, 0.081, 0.029, 0.019, 0.087,
                   0.035, 0.027, 0.039, 0.034, 0.032, 0.019, 0.022, 0.019, 0.022, 0.023])

def normalize_vector(v):
    """
    v: (nsamples, 3)
    """
    # 计算每一行的模
    norms = torch.linalg.norm(v, axis=1, keepdims=True)
    # 防止除以0的情况
    norms = torch.add(norms, 1e-8)
    # 对每一行进行归一化
    normalized_arr = v / norms
    return normalized_arr


class MANO:
    def __init__(self, hand_type, **kwargs):
        self.hand_type = hand_type
    
    def get_init_pose_batch(self, hand_trans, bs):
        filepath = f"./model/mano_info/J3_{self.hand_type}.txt"
        with open(filepath) as f:
            lines = f.readlines()
        init_pose = torch.zeros((21, 3)).to(hand_trans.device)
        for l in range(len(lines)):
            init_pose[l] = torch.tensor([float(x) for x in lines[l].rstrip().split(' ')])
        init_pose = init_pose - init_pose[0]
        init_pose_batch = init_pose.unsqueeze(0).repeat(bs, 1, 1)  # repeat dim 0, keep dim 1, 2 unchanged 
        # init_pose_batch = torch.tensor([init_pose + trans for trans in hand_trans]).to(hand_trans.device)
        hand_trans = hand_trans.unsqueeze(1)
        init_pose_batch = torch.add(init_pose, hand_trans)
        return init_pose_batch
    
    def get_joint_positions_batch(self, trans, rotations, bone_lengths=BONE_LENGTHS, parent_indices=MANO_PARENTS_INDICES):
        """
        - init_positions: shape (nsamples, 21, 3), the initial joint positions for each frame.
        - rotations:  shape (nsamples, 16, 3, 3), the rotation matrices for the hand joints for each sample.
        - bone_lengths: numpy array of shape (20,)
        """
        # trans = to_numpy(trans)
        # rotations = to_numpy(rotations)
        
        init_positions = self.get_init_pose_batch(hand_trans=trans, bs=rotations.shape[0])
        positions = torch.zeros_like(init_positions).to(rotations.device)  # shape (nsamples, 21, 3)
        positions[:, 0, :] = init_positions[:, 0, :]  # The root joint (wrist) remains the same

        # Loop over each joint (1 to 20), and calculate its position relative to its parent joint
        for i in range(1, 21):
            parent_index = parent_indices[i]

            # We start from the parent and accumulate all rotation matrices down to the current joint
            R = rotations[:, parent_index]  # Get the parent's rotation matrix for the entire batch

            # Accumulate rotation matrices from the root to the parent
            current_parent_index = parent_index
            while current_parent_index != 0:
                current_parent_index = parent_indices[current_parent_index]
                R = torch.einsum('bij,bjk->bik', rotations[:, current_parent_index], R)  # Batch matrix multiplication

            # Bone length for the current joint
            bone_length = bone_lengths[i - 1]

            # Calculate the vector between the parent and the current joint in the initial pose
            original_parent_position = init_positions[:, parent_index, :]  # (nsamples, 3)
            original_self_position = init_positions[:, i, :]  # (nsamples, 3)
            # ATTENTION the way of normalizing! in terms of each row rather than the whole array
            original_vector = torch.sub(original_self_position, original_parent_position)
            original_vector = bone_length * normalize_vector(original_vector)

            # Apply the accumulated rotation to this vector
            relative_position = torch.einsum('bij,bj->bi', R, original_vector)  # Batch matrix-vector multiplication
            # The new global position of the current joint is the parent's new position plus the relative displacement
            positions[:, i, :] = relative_position + positions[:, parent_index, :]

        # positions = to_torch(positions).float()

        return positions
    
    
    
    