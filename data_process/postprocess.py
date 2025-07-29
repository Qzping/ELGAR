import numpy as np
import json
import h5py
import torch

# from data_process.train_data_v3.rotation2xyz import Rotation2xyz
from data_process.data_alignment import *
from data_process.data_manipulation import DataManipulatorCello
from model.smpl import SMPL
from icecream import ic
import pickle
from scipy.spatial.transform import Rotation
from utils.rotation_conversions import matrix_to_quaternion, rotation_6d_to_matrix


SMPL_PARENTS_INDICES = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

def get_original_motion(original_motion_file, original_body_file, instrument_3d_file='../data_process/instrument/instrument.npy', if_smooth=True):
    data_manipulator_cello = DataManipulatorCello()

    # Body File
    body_dict = pickle.load(open(original_body_file, 'rb'))
    body_poses = body_dict['pose_body'].reshape(-1, 21, 3)
    num_frame = body_poses.shape[0]
    data_body_R = Rotation.from_rotvec(body_poses.reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)
    data_body_R = torch.tensor(data_body_R).float()

    smpl_model = SMPL()
    body_joints = smpl_model.get_joint_positions_batch(data_body_R)
    body_joints = np.array(body_joints)
    body_root_orient = smpl_model.root_orient.unsqueeze(0).repeat(num_frame, 1)
    body_root_orient_rot_mat = torch.tensor(Rotation.from_rotvec(body_root_orient).as_matrix()).float()
    lh_rot_mat_accumulate = body_root_orient_rot_mat.clone()
    rh_rot_mat_accumulate = body_root_orient_rot_mat.clone()

    lh_chain_index = [2, 5, 8, 12, 15, 17, 19]
    rh_chain_index = [2, 5, 8, 13, 16, 18, 20]

    for id in range(7):
        lh_index = lh_chain_index[id]
        lh_rot_mat_accumulate = torch.matmul(lh_rot_mat_accumulate, data_body_R[:, lh_index])
        rh_index = rh_chain_index[id]
        rh_rot_mat_accumulate = torch.matmul(rh_rot_mat_accumulate, data_body_R[:, rh_index])

    lh_wrist_global = lh_rot_mat_accumulate.contiguous().view(num_frame, 3, 3)
    rh_wrist_global = rh_rot_mat_accumulate.contiguous().view(num_frame, 3, 3)
    lh_trans_global = body_joints[:, 20]
    rh_trans_global = body_joints[:, 21]

    with open(original_motion_file, 'r') as file:
        original_data = json.load(file)

    lh_wrist_global6d = lh_wrist_global[:, :2, :].reshape(len(lh_wrist_global), 6)
    rh_wrist_global6d = rh_wrist_global[:, :2, :].reshape(len(rh_wrist_global), 6)

    # Hands
    original_lh_rot = np.array(original_data['lh_pose'])
    # ic(original_lh_rot.shape)
    original_lh_rot[:, 0:6] = lh_wrist_global6d

    original_rh_rot = np.array(original_data['rh_pose'])
    original_rh_rot[:, 0:6] = rh_wrist_global6d
    original_num_frames = original_lh_rot.shape[0]
    original_lh_3d = data_manipulator_cello.rot2position_hand(hand_trans=lh_trans_global,
                                                              hand_6d_96=original_lh_rot,
                                                              hand_type='left', smooth=if_smooth)
    original_rh_3d = data_manipulator_cello.rot2position_hand(hand_trans=rh_trans_global,
                                                              hand_6d_96=original_rh_rot,
                                                              hand_type='right', smooth=if_smooth)

    # Instrument
    if 'instrument' in original_data:
        original_instrument_frame0 = np.array(original_data['instrument'])
    else:
        original_instrument_frame0 = np.load(instrument_3d_file, allow_pickle=True)

    original_R_s, original_instrument_frame0_rotated = R_of_orientating(original_instrument_frame0)
    original_instrument_3d = original_instrument_frame0.reshape(-1, 11, 3).repeat(original_num_frames, axis=0)
    # original_instrument_3d = data_manipulator_cello.get_second_third_string(original_instrument_3d)

    # Bow
    bow_vec = np.array(original_data['bow'])
    hand_joints_bow = [5, 6, 11, 12, 14, 15]
    bow_start = np.mean(original_rh_3d[:, hand_joints_bow], axis=1)
    bow_end = bow_start + bow_vec * 0.75
    original_bow_3d = np.zeros((original_num_frames, 2, 3))
    original_bow_3d[:, 0, :] = bow_start
    original_bow_3d[:, 1, :] = bow_end

    # Contact point
    cp_info = np.array(original_data['cp_info'])
    original_cp_3d = data_manipulator_cello.calculate_cp_coordinates(cp_info, original_instrument_3d)
    original_cp_3d = original_cp_3d[:, np.newaxis, :]

    # used finger
    used_finger_idx = np.array(original_data['used_finger_idx'])

    return original_lh_3d, original_rh_3d, body_joints, original_instrument_3d, original_bow_3d, original_cp_3d, cp_info, used_finger_idx, original_R_s


def get_window_array(window_size, fade_size):
    fadein = np.linspace(0, 1, fade_size)
    fadeout = np.linspace(1, 0, fade_size)
    window = np.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def get_final_gen_motion_old(generation_path, instrument_3d_file='../data_process/instrument/instrument.npy'):
    data_manipulator_cello = DataManipulatorCello()

    results = np.load(generation_path, allow_pickle=True)
    results = results[()]  # 0-dim
    results = {key: np.array(value) for key, value in results.items()}
    motion = results['motion']
    # ic(motion.shape)
    # hand_motion = hand_motion.transpose(0, 3, 1, 2)
    lh_motion = motion[:, :21]
    rh_motion = motion[:, 21:42]
    bow_motion = motion[:, 42:44]
    body_motion = motion[:, 44:]

    body_motion[:, -1] = rh_motion[:, 0]
    body_motion[:, -2] = lh_motion[:, 0]

    bs = lh_motion.shape[0]
    seq_len = 150  # chunk length ( 5 x 30 = 150 )
    overlap_num = 5  # num of overlaps
    step = int(seq_len // overlap_num)
    weight_len = 30  # weight length

    length_init = ((bs - 1) * 30 // step) * step + seq_len  # total length after cat

    window = get_window_array(seq_len, weight_len)

    hand_shape = tuple((21, 3, length_init))
    bow_shape = tuple((2, 3, length_init))
    body_shape = tuple((22, 3, length_init))

    lh_compound_result = np.zeros(hand_shape, dtype=np.float32)
    lh_counter = np.zeros(hand_shape, dtype=np.float32)
    rh_compound_result = np.zeros(hand_shape, dtype=np.float32)
    rh_counter = np.zeros(hand_shape, dtype=np.float32)
    bow_compound_result = np.zeros(bow_shape, dtype=np.float32)
    bow_counter = np.zeros(bow_shape, dtype=np.float32)
    body_compound_result = np.zeros(body_shape, dtype=np.float32)
    body_counter = np.zeros(body_shape, dtype=np.float32)

    i = 0
    j = 0
    k = step // 30
    # print(f'scale factor: {k}')

    while i < bs:
        if i == 0:
            window[:weight_len] = 1
        elif (i + k) >= bs:
            window[-weight_len:] = 1

        x_lh = np.array(lh_motion[i])
        x_rh = np.array(rh_motion[i])
        x_bow = np.array(bow_motion[i])
        x_body = np.array(body_motion[i])

        lh_compound_result[..., (j * step):(j * step + seq_len)] += x_lh * window[..., :seq_len]
        lh_counter[..., (j * step):(j * step + seq_len)] += window[..., :seq_len]

        rh_compound_result[..., (j * step):(j * step + seq_len)] += x_rh * window[..., :seq_len]
        rh_counter[..., (j * step):(j * step + seq_len)] += window[..., :seq_len]

        bow_compound_result[..., (j * step):(j * step + seq_len)] += x_bow * window[..., :seq_len]
        bow_counter[..., (j * step):(j * step + seq_len)] += window[..., :seq_len]

        body_compound_result[..., (j * step):(j * step + seq_len)] += x_body * window[..., :seq_len]
        body_counter[..., (j * step):(j * step + seq_len)] += window[..., :seq_len]

        i += k
        j += 1

    lh_result = lh_compound_result / lh_counter
    rh_result = rh_compound_result / rh_counter
    bow_result = bow_compound_result / bow_counter
    body_result = body_compound_result / body_counter
    # print(f'actual shape: {estimated_sources.shape}')

    lh_motion_clip = lh_result.transpose(2, 0, 1)
    rh_motion_clip = rh_result.transpose(2, 0, 1)
    bow_motion_clip = bow_result.transpose(2, 0, 1)
    body_motion_clip = body_result.transpose(2, 0, 1)
    nframes, njoints, nxyz = lh_motion_clip.shape

    if 'instrument' in results:
        instrument_pos = np.array(results['instrument'])
    else:
        instrument_pos = np.load(instrument_3d_file, allow_pickle=True)

    R_s, instrument_pos_rotated = R_of_orientating(instrument_pos)

    instrument_motion_clip = instrument_pos[np.newaxis, :, :].repeat(nframes, axis=0)

    return lh_motion_clip, rh_motion_clip, body_motion_clip, instrument_motion_clip, bow_motion_clip, R_s


def get_final_gen_motion(generation_path, instrument_3d_file='../data_process/instrument/instrument.npy'):
    data_manipulator_cello = DataManipulatorCello()

    results = np.load(generation_path, allow_pickle=True)
    results = results[()]  # 0-dim
    results = {key: np.array(value) for key, value in results.items()}
    motion = results['motion']

    lh_motion = motion[:, :21]
    rh_motion = motion[:, 21:42]
    bow_motion = motion[:, 42:44]
    body_motion = motion[:, 44:]

    bs = lh_motion.shape[0]
    seq_len = 150  # chunk length ( 5 x 30 = 150 )
    overlap_num = 5 / 3  # num of overlaps
    step = 30
    weight_len = seq_len - step
    length_init = ((bs - 1) * 30 // step) * step + seq_len  # total length after cat
    # ic(length_init)
    window = get_window_array(seq_len, weight_len)

    hand_shape = tuple((21, 3, length_init))
    bow_shape = tuple((2, 3, length_init))
    body_shape = tuple((22, 3, length_init))

    lh_compound_result = np.zeros(hand_shape, dtype=np.float32)
    lh_counter = np.zeros(hand_shape, dtype=np.float32)
    rh_compound_result = np.zeros(hand_shape, dtype=np.float32)
    rh_counter = np.zeros(hand_shape, dtype=np.float32)
    bow_compound_result = np.zeros(bow_shape, dtype=np.float32)
    bow_counter = np.zeros(bow_shape, dtype=np.float32)
    body_compound_result = np.zeros(body_shape, dtype=np.float32)
    body_counter = np.zeros(body_shape, dtype=np.float32)

    i = 0
    j = 0
    k = step // 30

    while i < bs - k:
        fade_size = seq_len - step

        pad = np.ones(step)
        fadein = np.linspace(0, 1, fade_size)
        fadeout = np.linspace(1, 0, fade_size)
        out_win = np.concatenate((pad, fadeout))
        in_win = np.concatenate((fadein, pad))

        x_lh_p = lh_motion[i]
        x_rh_p = rh_motion[i]
        x_bow_p = bow_motion[i]
        x_body_p = body_motion[i]

        x_lh_n = lh_motion[i + k]
        x_rh_n = rh_motion[i + k]
        x_bow_n = bow_motion[i + k]
        x_body_n = body_motion[i + k]

        if i == 0:
            lh_compound_result[..., :seq_len] = x_lh_p
            rh_compound_result[..., :seq_len] = x_rh_p
            bow_compound_result[..., :seq_len] = x_bow_p
            body_compound_result[..., :seq_len] = x_body_p

        s1 = j * step
        s2 = (j + 1) * step
        e1 = s1 + seq_len
        e2 = s2 + seq_len
        lh_compound_result[..., s1:e1] *= out_win
        lh_compound_result[..., s2:e2] += x_lh_n * in_win
        rh_compound_result[..., s1:e1] *= out_win
        rh_compound_result[..., s2:e2] += x_rh_n * in_win
        bow_compound_result[..., s1:e1] *= out_win
        bow_compound_result[..., s2:e2] += x_bow_n * in_win
        body_compound_result[..., s1:e1] *= out_win
        body_compound_result[..., s2:e2] += x_body_n * in_win

        i += k
        j += 1

    lh_result = lh_compound_result
    rh_result = rh_compound_result
    bow_result = bow_compound_result
    body_result = body_compound_result

    lh_motion_clip = lh_result.transpose(2, 0, 1)
    rh_motion_clip = rh_result.transpose(2, 0, 1)
    bow_motion_clip = bow_result.transpose(2, 0, 1)
    body_motion_clip = body_result.transpose(2, 0, 1)
    nframes, njoints, nxyz = lh_motion_clip.shape

    if 'instrument' in results:
        instrument_pos = np.array(results['instrument'])
    else:
        instrument_pos = np.load(instrument_3d_file, allow_pickle=True)

    R_s, instrument_pos_rotated = R_of_orientating(instrument_pos)

    instrument_motion_clip = instrument_pos[np.newaxis, :, :].repeat(nframes, axis=0)

    return lh_motion_clip, rh_motion_clip, body_motion_clip, instrument_motion_clip, bow_motion_clip, R_s

def ensure_quaternion_consistency(q: torch.Tensor) -> torch.Tensor:
    """
    params:
        q: (bs, njoints, nframes, 4)
    return:
        q_fixed: same shape as q
    """
    
    bs, njoints, nframes, _ = q.shape

    if nframes <= 1:
        return q

    dot = (q[:, :, 1:] * q[:, :, :-1]).sum(dim=-1)  # (bs, njoints, nframes-1)

    sign_flip = torch.where(
        dot < 0,
        torch.tensor(-1., dtype=q.dtype, device=q.device),
        torch.tensor( 1., dtype=q.dtype, device=q.device)
    )  # (bs, njoints, nframes-1)

    sign_cumprod = sign_flip.cumprod(dim=2)  # cumulative product along time axis

    #    leading_ones: (bs, njoints, 1)
    leading_ones = torch.ones(
        (bs, njoints, 1),
        dtype=q.dtype,
        device=q.device
    )

    # (bs, njoints, nframes)
    sign = torch.cat([leading_ones, sign_cumprod], dim=2)

    # sign.unsqueeze(-1): (bs, njoints, nframes, 1)
    q_fixed = q * sign.unsqueeze(-1)

    return q_fixed

def get_long_rot(generation_path, long_rot_dir):
    """
    Concatenate the generated motion and get the npz with lh, rh and body rotations
    """

    results = np.load(generation_path, allow_pickle=True)

    results = results[()]  # 0-dim
    results = {key: np.array(value) for key, value in results.items()}

    lh_motion = results['lh_rot']
    rh_motion = results['rh_rot']
    body_motion = results['body_rot']

    lh_motion = torch.tensor(lh_motion.transpose(0, 1, 3, 2))
    rh_motion = torch.tensor(rh_motion.transpose(0, 1, 3, 2))
    body_motion = torch.tensor(body_motion.transpose(0, 1, 3, 2))

    lh_motion = matrix_to_quaternion(rotation_6d_to_matrix(lh_motion))
    rh_motion = matrix_to_quaternion(rotation_6d_to_matrix(rh_motion))
    body_motion = matrix_to_quaternion(rotation_6d_to_matrix(body_motion))

    lh_motion = ensure_quaternion_consistency(lh_motion)
    rh_motion = ensure_quaternion_consistency(rh_motion)
    body_motion = ensure_quaternion_consistency(body_motion)

    lh_motion = lh_motion.numpy().transpose(0, 1, 3, 2)
    rh_motion = rh_motion.numpy().transpose(0, 1, 3, 2)
    body_motion = body_motion.numpy().transpose(0, 1, 3, 2)

    bs = lh_motion.shape[0]
    seq_len = 150
    step = 30
    weight_len = seq_len - step  # weight length
    length_init = ((bs - 1) * 30 // step) * step + seq_len  # total length after cat
    window = get_window_array(seq_len, weight_len)

    hand_shape = tuple((15, 4, length_init))
    body_shape = tuple((21, 4, length_init))

    lh_compound_result = np.zeros(hand_shape, dtype=np.float32)
    lh_counter = np.zeros(hand_shape, dtype=np.float32)
    rh_compound_result = np.zeros(hand_shape, dtype=np.float32)
    rh_counter = np.zeros(hand_shape, dtype=np.float32)
    body_compound_result = np.zeros(body_shape, dtype=np.float32)
    body_counter = np.zeros(body_shape, dtype=np.float32)

    i = 0
    j = 0
    k = step // 30

    while i < bs - k:
        fade_size = seq_len - step

        pad = np.ones(step)
        fadein = np.linspace(0, 1, fade_size)
        fadeout = np.linspace(1, 0, fade_size)
        out_win = np.concatenate((pad, fadeout))
        in_win = np.concatenate((fadein, pad))

        x_lh_p = lh_motion[i]
        x_rh_p = rh_motion[i]
        x_body_p = body_motion[i]

        x_lh_n = lh_motion[i+k]
        x_rh_n = rh_motion[i+k]
        x_body_n = body_motion[i+k]

        if i == 0:
            lh_compound_result[..., :seq_len] = x_lh_p
            rh_compound_result[..., :seq_len] = x_rh_p
            body_compound_result[..., :seq_len] = x_body_p

        s1 = j * step
        s2 = (j+1) * step
        e1 = s1 + seq_len
        e2 = s2 + seq_len
        lh_compound_result[..., s1:e1] *= out_win
        lh_compound_result[..., s2:e2] += x_lh_n * in_win
        rh_compound_result[..., s1:e1] *= out_win
        rh_compound_result[..., s2:e2] += x_rh_n * in_win
        body_compound_result[..., s1:e1] *= out_win
        body_compound_result[..., s2:e2] += x_body_n * in_win

        i += k
        j += 1


    lh_result = lh_compound_result
    rh_result = rh_compound_result
    body_result = body_compound_result

    lh_motion_clip = lh_result.transpose(2, 0, 1)

    rh_motion_clip = rh_result.transpose(2, 0, 1)
    body_motion_clip = body_result.transpose(2, 0, 1)

    dirname = os.path.dirname(generation_path)

    os.makedirs(long_rot_dir, exist_ok=True)
    long_rot_file = os.path.join(long_rot_dir, 'lh_rh_body_rot.npz')
    np.savez(long_rot_file,
            lh_motion=lh_motion_clip,
            rh_motion=rh_motion_clip,
            body_motion=body_motion_clip)

    return long_rot_file
