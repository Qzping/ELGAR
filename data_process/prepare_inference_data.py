import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
from icecream import ic
from audio_encoder.encode import extract_jukebox_feats
from data_process.data_normalization import SEQ_LEN, FRAME_RATE, HOP_LEN
import pickle
import glob
from pydub import AudioSegment
import shutil
import argparse


def get_test_wavs(dir_path, suffix):
    wav_files = []
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(suffix): 
                    file_path = os.path.join(dir_path,  entry.name) 
                    wav_files.append(file_path) 
        return wav_files 
    except FileNotFoundError:
        print(f"'{dir_path}' NOT exist!")
        return []
    
def get_test_data(test_data_dir):

    short_cello_idices = ['01', '05', '08', '12', '45', '47']  # short test data that could be directly used
    for idx in short_cello_idices:
        body_file = f"./ik_joints/results/body_cello{idx}_calc.pkl"
        motion_file = f"./train_data/wholebody_normalized/cello{idx}.json"
        
        new_motion_dir = os.path.join(test_data_dir, 'motion')
        os.makedirs(new_motion_dir, exist_ok=True)
        
        shutil.copy(body_file, os.path.join(new_motion_dir, f"body_cello{idx}_calc.pkl"))
        shutil.copy(motion_file, os.path.join(new_motion_dir, f"cello{idx}.json"))

        print(f'body_file and motion_file {idx} copied.')

    split_cello_indices = ['26', '59', '60']  # split long test data that need to be split into sub-sequences

    for idx in split_cello_indices:
        body_file = f"./ik_joints/results/body_cello{idx}_calc.pkl"
        body_dict = pickle.load(open(body_file, 'rb'))

        motion_file = f"./train_data/wholebody_normalized/cello{idx}.json"
        with open(motion_file, 'r') as file:
            original_data = json.load(file)
        
        audio_dir = os.path.join(test_data_dir, 'audio')
        audio_files = sorted(glob.glob(f"{audio_dir}/cello{idx}*"))
        sub_idx = 0
        pointer = 0
        for audio_file in audio_files:
            # ic(audio_file)sp
            sub_idx += 1
            audio = AudioSegment.from_wav(audio_file)
            duration_ms = len(audio)
            duration_seconds = duration_ms / 1000
            frame_count = int(duration_seconds * 30)

            body_poses_i = body_dict['pose_body'][pointer : pointer + frame_count]
            lh_pose_i = original_data['lh_pose'][pointer : pointer + frame_count]
            rh_pose_i = original_data['rh_pose'][pointer : pointer + frame_count]
            bow_i = original_data['bow'][pointer : pointer + frame_count]
            cp_info_i = original_data['cp_info'][pointer: pointer + frame_count]
            used_finger_idx_i = original_data['used_finger_idx'][pointer: pointer + frame_count]
            instrument = original_data['instrument']
            pointer += frame_count
            
            new_motion_dir = os.path.join(test_data_dir, 'motion')
            os.makedirs(new_motion_dir, exist_ok=True)
            new_body_file = os.path.join(new_motion_dir, f"body_cello{idx}_{sub_idx}_calc.pkl")
            new_motion_file = os.path.join(new_motion_dir, f"cello{idx}_{sub_idx}.json")

            body_dict_new = {'pose_body': body_poses_i}
            with open(new_body_file, 'wb') as f:
                pickle.dump(body_dict_new, f)

            motion_dict_new = {'lh_pose': lh_pose_i,
                               'rh_pose': rh_pose_i,
                               'bow': bow_i,
                               'cp_info': cp_info_i,
                               'used_finger_idx': used_finger_idx_i,
                               'instrument': instrument}
            with open(new_motion_file, 'w') as f:
                json.dump(motion_dict_new, f)
            
            print(f'body_file and motion_file {idx}_{sub_idx} splitted.')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='prepare_inference_data')
    parser.add_argument('--data_type', default='test', type=str, required=True)
    parser.add_argument('--filename', default='test', type=str, required=False)
    parser.add_argument('--instrument', default='cello', type=str)
    
    args = parser.parse_args()

    data_type = args.data_type
    filename = args.filename
    instrument = args.instrument

    if data_type == 'test':

        test_data_dir = f'../dataset/SPD-GEN/{instrument}/test_data'
        get_test_data(test_data_dir)
        # extract jukebox features for test split
        audio_dir = os.path.join(test_data_dir, 'audio')
        audio_file_suffix = '.wav'

        audio_files = sorted(get_test_wavs(audio_dir, audio_file_suffix))

        for audio_path in audio_files:
            print(f'Extrating juke features from {audio_path}...')

            filename = os.path.basename(audio_path).split('.')[0]

            audio_feats = extract_jukebox_feats(audio_path, int(SEQ_LEN / FRAME_RATE), int(HOP_LEN / FRAME_RATE), model_path='../audio_encoder/model')

            test_feature_dir = os.path.join(test_data_dir, 'juke_feature')
            os.makedirs(test_feature_dir, exist_ok=True)
            test_feature_path = os.path.join(test_feature_dir, f'{filename}.npy')
            np.save(test_feature_path, audio_feats)

            print(f'{filename} feature extrated.')
        
            
    elif data_type == 'wild':
        # extract jukebox features for in-the-wild data
        wild_data_dir = f'../dataset/wild_data/{instrument}'
        wild_audio_dir = os.path.join(wild_data_dir, 'audio')

        audio_path = os.path.join(wild_audio_dir, filename)

        audio_feats = extract_jukebox_feats(audio_path, int(SEQ_LEN / FRAME_RATE), int(HOP_LEN / FRAME_RATE), model_path='../audio_encoder/model')

        test_feature_dir = os.path.join(wild_data_dir, 'juke_feature')
        os.makedirs(test_feature_dir, exist_ok=True)
        songname = os.path.basename(filename).split('.')[0]
        test_feature_path = os.path.join(test_feature_dir, f'{songname}.npy')
        np.save(test_feature_path, audio_feats)

        print(f'{filename} feature extrated.')
    
    else:
        raise ValueError(f'Invalid data type: {data_type}, please choose from "test" or "wild"')
