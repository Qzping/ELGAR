from icecream import ic

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

import json
import pickle
import torch
import numpy as np

import subprocess
from multiprocessing import Pool
from multiprocessing import Manager
from collections import Counter
import argparse


def get_gpu_memory_usage():
    try:
        # Execute "nvidia-smi" ,and achieve the output
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Analyze the output
        lines = result.stdout.strip().split('\n')
        gpu_memory_info = []
        for line in lines:
            total, used, free = line.split(', ')
            gpu_memory_info.append({
                'total_memory': int(total),
                'used_memory': int(used),
                'free_memory': int(free)
            })
        return gpu_memory_info
    except FileNotFoundError:
        print("nvidia-smi command not found. Please ensure NVIDIA drivers are installed.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing nvidia-smi: {e}")
        return None

def choose_CUDA_Device4process(allocated_gpus):
    gpu_memory_info = get_gpu_memory_usage()

    if gpu_memory_info is None:
        cuda_device = 'cpu'
    else:
        gpu_free_memory = [gfm['free_memory'] for gfm in gpu_memory_info]
        if len(allocated_gpus) < torch.cuda.device_count():
            # Obtain the usage of memory for each GPU
            available_gpus = [
                i for i, gfm in enumerate(gpu_memory_info) if i not in allocated_gpus
            ]
            
            available_gpus = [
                i for i, gfm in enumerate(gpu_memory_info) if i not in allocated_gpus
            ]
            
            # Choose the first available GPU
            cuda_device = available_gpus[-1]
            allocated_gpus.append(cuda_device)
            print(f"Allocated GPUs: {allocated_gpus}")
        else:
            cuda_device = np.argmax(gpu_free_memory)
        print(f"GPU {cuda_device}: Total Memory: {gpu_memory_info[cuda_device]['total_memory']} MiB, "
              f"Used Memory: {gpu_memory_info[cuda_device]['used_memory']} MiB, "
              f"Free Memory: {gpu_memory_info[cuda_device]['free_memory']} MiB")
    return cuda_device

def solve_nan(human_kp_dir, nan_file_li=['cello69','cello71','cello79']):
    for nan_file in nan_file_li:
        with open(f'{human_kp_dir}/{nan_file}.json','r') as f:
            js_file = json.load(f)
        f.close()
        
        js_np = np.asarray(js_file['body'])
        frames_all = np.arange(js_np.shape[0])
        
        if np.isnan(js_np).any():
            nan_list = np.argwhere(np.isnan(js_np[:,:,0]))
            element_counts = Counter(nan_list[:,1])
        else:
            print(f'No nan in {nan_file}!')
            continue
            
        nan_keys_li = []
        for i,joint_id in enumerate(element_counts.keys()):
            if joint_id in [16,20,21,22]:
                nan_keys_li.append(joint_id)
                nan_keys_li.append(14)
            elif joint_id in [15,17,18,19]:
                nan_keys_li.append(joint_id)
                nan_keys_li.append(13)
        
        nan_keys_li = list(set(nan_keys_li))
        
        frames_nan_id = []
        for joint_id in nan_keys_li:
            frames_nan_id_joints = list(np.argwhere(np.isnan(js_np[:,joint_id,0])).flatten())
            frames_nan_id = list(set(frames_nan_id)|set(frames_nan_id_joints))
        
        frames_nan_id = sorted(list(set(frames_nan_id)))
        frames_no_nan_id = list(set(frames_all)-set(frames_nan_id))
        
        for joint_id in nan_keys_li:
            joints_selected_values = [js_np[i,joint_id,:] for i in frames_no_nan_id]
            
            joints_mean = np.mean(joints_selected_values,axis = 0)
            
            if joint_id in [14,16,20,21,22]:
                guided_kp_id = 14
            elif joint_id in [13,15,17,18,19]:
                guided_kp_id = 13
            
            differences = [list((js_np[i+1,guided_kp_id,:]-js_np[i,guided_kp_id,:])*0.9) if i+1 < js_np.shape[0] else [0,0,0]  for i in frames_nan_id]
            
            for i in range(len(frames_nan_id)):
                if frames_nan_id[i] == 0:
                    js_np[frames_nan_id[i],joint_id,:] = joints_mean
                else:
                    js_np[frames_nan_id[i],joint_id,:] = js_np[frames_nan_id[i]-1,joint_id,:]+differences[i]
        js_file['body'] = js_np.tolist()
        
        with open(f'{human_kp_dir}/{nan_file}.json','w') as f:
            f.write(json.dumps(js_file))
        f.close()

        print(f'{nan_file} nan solved.')

def calc_body_mean(result_dir='results', body_mean_dir='results'):
    files = os.listdir(result_dir)
    pkl_files = sorted([f for f in files if f.endswith('round_1.pkl')])

    body_betas_li = []
    body_scale_li = []
    body_trans_li = []
    body_root_orient_li = []

    for pkl in pkl_files:
        body_smpl_path = os.path.join(result_dir, pkl)
        body_dic = pickle.load(open(body_smpl_path, 'rb'))
        body_betas = body_dic['betas'].mean(axis=0)
        body_betas_li.append(body_betas.tolist())
        body_scale = body_dic['scaling'].mean() /1000
        body_scale_li.append(body_scale.tolist())
        body_trans = body_dic['trans'].mean(axis=0)
        body_trans_li.append(body_trans.tolist())
        body_root_orient = body_dic['root_orient'].mean(axis=0)
        body_root_orient_li.append(body_root_orient.tolist())

    body_betas = np.mean(body_betas_li, axis=0)
    body_scale = np.mean(body_scale_li)
    body_trans = np.mean(body_trans_li, axis=0)
    body_root_orient = np.mean(body_root_orient_li, axis=0)

    data_converted = {
        'body_betas':body_betas.tolist(),
        'body_scale':body_scale.tolist(),
        'body_trans':body_trans.tolist(),
        'body_root_orient':body_root_orient.tolist(),

    }

    with open(f'{body_mean_dir}/body_mean.json', 'w') as json_file:
        json.dump(data_converted, json_file, indent=4)

def ik_joints_optimize_process(folder_name, allocated_gpus, optimizing_time, instrument):
    process_id = os.getpid()
    print(f'Processing folder: {folder_name} -> round {optimizing_time} | with Process ID: {process_id}')

    proj_dir = folder_name

    # instrument = 'cello'
    bm_fname = '../../model/smplx/SMPLX_NEUTRAL.npz'
    vposer = 'human_body_prior/models/V02_05'
    motion_pkl_save_dir = 'results'
    human_kp_dir = f'../train_data/wholebody_normalized'
    print("human_kp_dir: ",human_kp_dir)

    if optimizing_time == 1:
       solve_nan(human_kp_dir=human_kp_dir, nan_file_li=['cello69','cello71','cello79'])
       use_vposer = 1
       only_fit_lh = 0
    elif optimizing_time == 2:
       calc_body_mean(result_dir=motion_pkl_save_dir, body_mean_dir=motion_pkl_save_dir)
       use_vposer = 0
       only_fit_lh = 0
    elif optimizing_time == 3:
        use_vposer = 0
        only_fit_lh = 1

    shell_python_cmd = 'python3'
    if optimizing_time < 3:
        python_script = 'ik_optimize'
    else:
        python_script = 'ik_get_local_wrist'
    
    ik_joints_command = f'{shell_python_cmd} {python_script}.py ' \
                              f'--proj_dir {proj_dir} ' \
                              f'--motion_pkl_save_dir {motion_pkl_save_dir} ' \
                              f'--human_kp_dir {human_kp_dir}'

    if optimizing_time < 3:
        if torch.cuda.is_available():
            cuda_device = str(choose_CUDA_Device4process(allocated_gpus))
        else:
            cuda_device = 'cpu'
        ik_joints_command += f' ' \
                                   f'--instrument {instrument} ' \
                                   f'--use_vposer {use_vposer} ' \
                                   f'--only_fit_lh {only_fit_lh} ' \
                                   f'--bm_fname {bm_fname} ' \
                                   f'--vposer {vposer} ' \
                                   f'--cuda_device {cuda_device} '
    
    os.system(ik_joints_command)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='ik_joints')
    parser.add_argument('--instrument', default='cello', type=str, required=False)
    parser.add_argument('--ik_time', default=1, type=int, required=True)
    
    args = parser.parse_args()

    instrument = args.instrument
    ik_time = args.ik_time

    # instrument = 'cello'
    # ik_time = 1  # set to 1 or 2 or 3
    proj_range = range(1, 86)
    # proj_range = [1]
    
    proj_dirs = [instrument+'0'*(2-len(str(i)))+str(i) for i in proj_range]

    print('proj_dirs:', proj_dirs)

    if ik_time < 3:
        if torch.cuda.is_available():
            process_num = torch.cuda.device_count()
        else:
            process_num = 2
    else:
        process_num = os.cpu_count()
    
    #print("CUDA_device_count:",torch.cuda.device_count())
    with Manager() as manager:
        # Create a shared GPU list by using multiprocessing.Manager
        allocated_gpus = manager.list()
        
        # Create a process pool by using multiprocessing.Pool to ensure that each process can only select a GPU
        with Pool(processes=process_num) as pool:
            pool.starmap(ik_joints_optimize_process, [(folder, allocated_gpus, ik_time, instrument) for folder in proj_dirs])
