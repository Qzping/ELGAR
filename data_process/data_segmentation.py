import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import numpy as np
import glob
import json
import h5py
from scipy.spatial.transform import Rotation
from icecream import ic
from tqdm import tqdm
from data_process.data_manipulation import DataManipulatorCello
import pickle
import torch
import smplx
from audio_encoder.encode import extract_jukebox_feats


def get_rest_pose(body_mean_dir):
    model_path = '../model'
    mean_file = 'body_mean.json'
    mean_path = os.path.join(body_mean_dir, mean_file)
    mean_data = json.load(open(mean_path, 'r'))

    betas_avg = torch.tensor(mean_data['body_betas']).unsqueeze(0)
    trans_avg = torch.tensor(mean_data['body_trans']).unsqueeze(0)
    scaling_avg = torch.tensor(mean_data['body_scale'])
    root_orient_avg = mean_data['body_root_orient']

    model = smplx.create(model_path=model_path, model_type='smplx', use_pca=False, flat_hand_mean=True)
    output = model(betas=betas_avg, transl=trans_avg)

    J_shaped = output.joints[:, :22]  # body
    joints_rest = J_shaped * scaling_avg
    joints_rest = joints_rest.squeeze()
    rest_pose = {
        'joints': joints_rest.tolist(),
        'root_orient': root_orient_avg,
    }

    save_dir = os.path.join(model_path, 'smpl_info')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rest_pose.json')
    with open(save_path, 'w') as file:
        json.dump(rest_pose, file, indent=4)


SEQ_LEN = 150   # number of frames per sequence
HOP_LEN = 30    # gap size between consecutive sequence
FRAME_RATE = 30

def find_pkl_files(directory):
    pkl_files = sorted(glob.glob(os.path.join(directory, '**', 'body_*_calc.pkl'), recursive=True))
    return pkl_files

if __name__ == "__main__":
    print("In Progress ...")

    instrument_type = 'cello'
    save_dir = "./train_data/wholebody_processed"
    data_dir = "./train_data/wholebody_normalized"
    body_rot_dir = "./ik_joints/results/"
    spd_gen_dir = f"../dataset/SPD-GEN/{instrument_type}/full_data/motion"

    data_manip = DataManipulatorCello()
    body_rot_list = find_pkl_files(body_rot_dir)

    loop_data_path_list = tqdm(body_rot_list, total=len(body_rot_list), leave=True)

    get_rest_pose(body_mean_dir=body_rot_dir)

    bow_sequences = []
    lh_trans_sequences = []
    lh_pose_sequences = []
    rh_trans_sequences = []
    rh_pose_sequences = []
    cp_info_sequences = []
    cp_pos_sequences = []
    body_sequences = []
    used_finger_sequences = []
    sample_num_sequences = []
    instrument = None
    audio_feats_sequences = []
    for data_path in loop_data_path_list:

        body_dict = pickle.load(open(data_path, 'rb'))
        body_poses = body_dict['pose_body'].reshape(-1, 21, 3)

        piece_idx = data_path.split('_')[-2][-2:]
        other_data_file = os.path.join(data_dir, f'{instrument_type}{piece_idx}.json')
        other_data = json.load(open(other_data_file, 'r'))

        data_lh_pose_6d = np.array(other_data['lh_pose']).reshape(-1, 16, 6)
        data_rh_pose_6d = np.array(other_data['rh_pose']).reshape(-1, 16, 6)
        data_lh_pose_6d = data_lh_pose_6d[:, 1:].reshape(-1, 15*6)
        data_rh_pose_6d = data_rh_pose_6d[:, 1:].reshape(-1, 15*6)

        data_bow = np.array(other_data['bow']).reshape(-1, 3)
        data_cp_info = np.array(other_data['cp_info']).reshape(-1, 2)
        instrument_piece = np.array(other_data['instrument'])
        if instrument is None:
            instrument = instrument_piece
        else:
            assert np.array_equal(instrument, instrument_piece)
        piece_len = other_data['length']

        data_body_R = Rotation.from_rotvec(body_poses.reshape(-1, 3)).as_matrix().reshape(-1, 3, 3)
        data_body_6d = data_body_R[:, :2, :].reshape(len(body_poses), 21 * 6)

        assert body_poses.shape[0] == data_lh_pose_6d.shape[0]

        data_used_finger = np.array(other_data['used_finger_idx'])

        data_dict = {
            'lh_pose': data_lh_pose_6d.tolist(),
            'rh_pose': data_rh_pose_6d.tolist(),
            'bow': data_bow.tolist(),
            'cp_info': data_cp_info.tolist(),
            'body': data_body_6d.tolist(),
            'instrument': instrument.tolist(),
            'used_finger': data_used_finger.tolist()
        }

        os.makedirs(spd_gen_dir, exist_ok=True)
        save_path = os.path.join(spd_gen_dir, f'{instrument_type}{piece_idx}.json')
        with open(save_path, 'w') as file:
            json.dump(data_dict, file, indent=4)

        audio_path = f'../dataset/SPD-GEN/{instrument_type}/audio/cello{piece_idx}.wav'
        audio_feats = extract_jukebox_feats(audio_path, (SEQ_LEN / FRAME_RATE), (HOP_LEN / FRAME_RATE), model_path='../audio_encoder/models')

        cur_lh_pose_sequences = []
        cur_rh_pose_sequences = []
        cur_bow_sequences = []
        cur_cp_info_sequences = []
        cur_cp_pos_sequences = []
        cur_body_sequences = []
        cur_used_finger_sequences = []

        for start_frame in range(0, piece_len, HOP_LEN):
            end_frame = min(start_frame + SEQ_LEN, piece_len)
            seq_bow = data_bow[start_frame: end_frame]
            seq_lh_pose = data_lh_pose_6d[start_frame: end_frame]
            seq_rh_pose = data_rh_pose_6d[start_frame: end_frame]
            seq_cp_info = data_cp_info[start_frame: end_frame]
            seq_body = data_body_6d[start_frame: end_frame]
            seq_used_finger = data_used_finger[start_frame: end_frame]

            if len(seq_bow) < SEQ_LEN:
                padding_length = SEQ_LEN - len(seq_bow)
                seq_bow = np.pad(seq_bow, ((0, padding_length), (0, 0)), mode='edge')
                seq_lh_pose = np.pad(seq_lh_pose, ((0, padding_length), (0, 0)), mode='edge')
                seq_rh_pose = np.pad(seq_rh_pose, ((0, padding_length), (0, 0)), mode='edge')
                seq_cp_info = np.pad(seq_cp_info, ((0, padding_length), (0, 0)), mode='edge')
                seq_body = np.pad(seq_body, ((0, padding_length), (0, 0)), mode='edge')
                seq_used_finger = np.pad(seq_used_finger, (0, padding_length), mode='edge')

            seq_instrument = instrument[np.newaxis, :, :].repeat(piece_len, axis=0)
            seq_cp_pos = data_manip.calculate_cp_coordinates(seq_cp_info, seq_instrument)

            cur_lh_pose_sequences.append(seq_lh_pose)
            cur_rh_pose_sequences.append(seq_rh_pose)
            cur_bow_sequences.append(seq_bow)
            cur_cp_info_sequences.append(seq_cp_info)
            cur_cp_pos_sequences.append(seq_cp_pos)
            cur_body_sequences.append(seq_body)
            cur_used_finger_sequences.append(seq_used_finger)

        assert len(audio_feats) == len(cur_lh_pose_sequences), \
            f'audio: {len(audio_feats)} != pose: {len(cur_lh_pose_sequences)}'

        bow_sequences.extend(cur_bow_sequences)
        sample_num_sequences.append(len(cur_bow_sequences))
        lh_pose_sequences.extend(cur_lh_pose_sequences)
        rh_pose_sequences.extend(cur_rh_pose_sequences)
        cp_info_sequences.extend(cur_cp_info_sequences)
        cp_pos_sequences.extend(cur_cp_pos_sequences)
        body_sequences.extend(cur_body_sequences)
        used_finger_sequences.extend(cur_used_finger_sequences)
        audio_feats_sequences.extend(audio_feats)

    bow_sequences = np.array(bow_sequences)
    sample_num_sequences = np.array(sample_num_sequences)
    lh_pose_sequences = np.array(lh_pose_sequences)
    rh_pose_sequences = np.array(rh_pose_sequences)
    cp_info_sequences = np.array(cp_info_sequences)
    cp_pos_sequences = np.array(cp_pos_sequences)
    body_sequences = np.array(body_sequences)
    used_finger_sequences = np.array(used_finger_sequences)
    audio_feats_sequences = np.array(audio_feats_sequences)

    cp_info_sequences[cp_info_sequences == None] = -1
    cp_info_sequences = cp_info_sequences.astype(float)

    os.makedirs(save_dir, exist_ok=True)

    audio_path = os.path.join(save_dir, 'audio.npy')
    np.save(audio_path, audio_feats_sequences)

    motion_path = os.path.join(save_dir, 'motion.hdf5')
    f = h5py.File(motion_path, "w")
    bow_dataset = f.create_dataset('bow', data=bow_sequences)
    lh_pose_dataset = f.create_dataset('lh_pose', data=lh_pose_sequences)
    rh_pose_dataset = f.create_dataset('rh_pose', data=rh_pose_sequences)
    cp_info_dataset = f.create_dataset('cp_info', data=cp_info_sequences)
    cp_pos_sequences = f.create_dataset('cp_pos', data=cp_pos_sequences)
    body_dataset = f.create_dataset('body', data=body_sequences)
    instrument_dataset = f.create_dataset('instrument', data=instrument)
    used_finger_dataset = f.create_dataset('used_finger', data=used_finger_sequences)
    sample_num_sequences = f.create_dataset('sample_num', data=sample_num_sequences)

    # instrument_path = os.path.join(save_dir, 'instrument.npy')
    # np.save(instrument_path, instrument)
