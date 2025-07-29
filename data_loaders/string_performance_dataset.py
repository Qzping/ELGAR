import json
import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils.misc import to_torch
from diffusion import logger
from icecream import ic


def get_training_data(split_points, selected_indices, total):
    """
    sample training set from total data

    param:
    - split_points: list, len(sample_num)
    - selected_indices: list, train set index
    - total: np.ndarray: (6665, ...)

    return:
    - training_data: np.ndarray
    """

    selected_subarrays = []
    for idx in selected_indices:
        start_idx = split_points[idx - 1] if idx > 0 else 0
        end_idx = split_points[idx]
        selected_subarrays.append(total[start_idx:end_idx])

    training_data = np.concatenate(selected_subarrays, axis=0)

    return training_data


class StringPerformanceDataset(Dataset):
    dataname = "spd"

    def __init__(self, datapath="data_process/train_data/wholebody_processed", 
                 filename='{"audio": "audio.npy", "motion": "motion.hdf5"}',
                 split='train', **kargs):
        self.datapath = datapath
        self.filename = filename
        self.split = split
        self.cond_mode = kargs.get('cond_mode', 'audio')
        self.cond_info = [self.cond_mode]  # store the given info from condition
        self.train_mode = kargs.get('train_mode', 'total')

        super().__init__()   
        
        if split == 'train':
            condfile = filename[self.cond_mode]
            conddatafilepath = os.path.join(datapath, condfile)
            conddata = np.load(conddatafilepath)
            motionfile = filename['motion']
            motiondatafilepath = os.path.join(datapath, motionfile)
            f = h5py.File(motiondatafilepath, 'r')

            lh_pose = np.array(f['lh_pose'])
            rh_pose = np.array(f['rh_pose'])
            bow = np.array(f['bow'])
            body = np.array(f['body'])

            repeat_time = 1
            noiseaugdata = None
            if 'noise_aug' in filename:
                noiseaugfile = filename['noise_aug']
                noiseaugdatafilepath = os.path.join(datapath, noiseaugfile)
                noiseaugdata = np.load(noiseaugdatafilepath)
                repeat_time += 1
            
            reverbaugdata = None
            if 'reverb_aug' in filename:
                reverbaugfile = filename['reverb_aug']
                reverbaugdatafilepath = os.path.join(datapath, reverbaugfile)
                reverbaugdata = np.load(reverbaugdatafilepath)
                repeat_time += 1

            if self.train_mode == 'partial':
                logger.log('Train with partial data...')
                sample_num = np.array(f['sample_num'])
                split_points = np.cumsum(sample_num)

                test_index = [0, 4, 7, 11, 24, 43, 44, 54, 55]
                all_index = np.arange(81)
                train_index = np.setdiff1d(all_index, test_index)

                conddata = get_training_data(split_points, train_index, conddata)
                lh_pose = get_training_data(split_points, train_index, lh_pose)
                rh_pose = get_training_data(split_points, train_index, rh_pose)
                bow = get_training_data(split_points, train_index, bow)
                body = get_training_data(split_points, train_index, body)

                if noiseaugdata is not None:
                    noiseaugdata = get_training_data(split_points, train_index, noiseaugdata)
                if reverbaugdata is not None:
                    reverbaugdata = get_training_data(split_points, train_index, reverbaugdata)

            train_sample_num = conddata.shape[0]

            logger.log(f'Train with {train_sample_num} samples!')
            
            if noiseaugdata is not None:
                conddata = np.concatenate((conddata, noiseaugdata), axis=0)
            if reverbaugdata is not None:
                conddata = np.concatenate((conddata, reverbaugdata), axis=0)
            lh_pose = lh_pose.repeat(repeat_time, axis=0)
            rh_pose = rh_pose.repeat(repeat_time, axis=0)
            bow = bow.repeat(repeat_time, axis=0)
            body = body.repeat(repeat_time, axis=0)

            data = {
                'lh_pose': lh_pose,
                'rh_pose': rh_pose,
                'bow': bow,
                'body': body,
                f'{self.cond_mode}': conddata
            }
            if 'stft' in filename:
                print('loading stft data...')
                stftfile = filename['stft']
                stftdatafilepath = os.path.join(datapath, stftfile)
                stftdata = np.load(stftdatafilepath)
                if self.train_mode == 'partial':
                    stftdata = get_training_data(split_points, train_index, stftdata)
                    assert stftdata.shape[0] == train_sample_num
                stftdata = stftdata.repeat(repeat_time, axis=0)
                data['stft'] = stftdata
                self.cond_info.append('stft')
                logger.log('stft data loaded')
            if 'mel' in filename:
                print('loading mel data...')
                melfile = filename['mel']
                meldatafilepath = os.path.join(datapath, melfile)
                meldata = np.load(meldatafilepath)
                if self.train_mode == 'partial':
                    meldata = get_training_data(split_points, train_index, meldata)
                    assert meldata.shape[0] == train_sample_num
                meldata = meldata.repeat(repeat_time, axis=0)
                data['mel'] = meldata
                self.cond_info.append('mel')
                logger.log('mel data loaded')
            if 'cp_pos' in f.keys():
                print('loading cp_pos data...')
                cp_pos = np.array(f['cp_pos'])
                if self.train_mode == 'partial':
                    cp_pos = get_training_data(split_points, train_index, cp_pos)
                    assert cp_pos.shape[0] == train_sample_num
                cp_pos = cp_pos.repeat(repeat_time, axis=0)
                data['cp_pos'] = cp_pos
                self.cond_info.append('cp_pos')
                logger.log('cp_pos data loaded')
            if 'cp_info' in f.keys():
                print('loading cp_info data...')
                cp_info = np.array(f['cp_info'])
                if self.train_mode == 'partial':
                    cp_info = get_training_data(split_points, train_index, cp_info)
                    assert cp_info.shape[0] == train_sample_num
                cp_info = cp_info.repeat(repeat_time, axis=0)
                data['cp_info'] = cp_info
                self.cond_info.append('cp_info')
                logger.log('cp_info data loaded')
            if 'used_finger' in f.keys():
                print('loading used_finger data...')
                used_finger = np.array(f['used_finger'])
                if self.train_mode == 'partial':
                    used_finger = get_training_data(split_points, train_index, used_finger)
                    assert used_finger.shape[0] == train_sample_num
                used_finger = used_finger.repeat(repeat_time, axis=0)
                data['used_finger'] = used_finger
                self.cond_info.append('used_finger')
                logger.log('used_finger data loaded')
            if 'instrument' in f.keys():
                print('loading instrument data...')
                instrument = np.array(f['instrument'])
                data['instrument'] = instrument
                self.cond_info.append('instrument')
                logger.log('instrument data loaded')

            
        elif split == 'test':
            filename = filename[self.cond_mode]
            fulldatafilepath = os.path.join(datapath, filename)
            data = {}
            data[self.cond_mode] = np.load(fulldatafilepath)
        

        logger.log(f'data loaded from: {filename}')
        if self.split == 'train':
            logger.log('train data size:', len(data['body']))
        elif self.split == 'test':
            logger.log('test data size:', len(data[self.cond_mode]))
        logger.log(f'condition info includes: {self.cond_info}')

        if 'instrument' in self.cond_mode:
            self._instrument_pos = [np.array(pos) for pos in data["instrument_pos"]]
            self._num_frames = [p.shape[0] for p in self._instrument_pos]

        elif 'audio' in self.cond_mode:
            self._audio = [np.array(audio) for audio in data["audio"]]
            self._num_frames = [p.shape[0] for p in self._audio]
            if 'stft' in data:
                self._stft = [np.array(stft) for stft in data["stft"]]
            if 'mel' in data:
                self._mel = [np.array(mel) for mel in data["mel"]]
            if 'cp_pos' in data:
                self._cp_pos = [np.array(cp_pos) for cp_pos in data["cp_pos"]]
            if 'cp_info' in data:
                self._cp_info = [np.array(cp_info) for cp_info in data["cp_info"]]
            if 'used_finger' in data:
                self._used_finger = [np.array(used_finger) for used_finger in data["used_finger"]]
            if 'instrument' in data:
                self._static_instrument = np.array(data["instrument"])
                
        if self.split == 'train':
            self._lh_pose = [np.array(pose) for pose in data["lh_pose"]]

            self._rh_pose = [np.array(pose) for pose in data["rh_pose"]]

            self._bow = [np.array(b) for b in data["bow"]]

            self._body = [np.array(b) for b in data["body"]]

        self._data_ids = list(range(len(data[self.cond_mode])))

    def _load_instrument(self, data_id, frame_id):
        return self._instrument_pos[data_id][frame_id].reshape(-1, 7, 3)

    def _load_audio(self, data_id, frame_id):
        return self._audio[data_id][frame_id]
    
    def _load_stft(self, data_id, frame_id):
        return self._stft[data_id][frame_id]
    
    def _load_mel(self, data_id, frame_id):
        return self._mel[data_id][frame_id]
    
    def _load_lh_pose(self, data_id, frame_id):
        return self._lh_pose[data_id][frame_id].reshape(-1, 15, 6)
    
    def _load_rh_pose(self, data_id, frame_id):
        return self._rh_pose[data_id][frame_id].reshape(-1, 15, 6)
    
    def _load_bow(self, data_id, frame_id):
        return self._bow[data_id][frame_id].reshape(-1, 3)
    
    def _load_body(self, data_id, frame_id):
        return self._body[data_id][frame_id].reshape(-1, 21, 6)
    
    def _load_cp_pos(self, data_id, frame_id):
        # return self._cp_pos[data_id][frame_id].reshape(-1, 4, 3)
        return self._cp_pos[data_id][frame_id].reshape(-1, 1, 3)
    
    def _load_cp_info(self, data_id, frame_id):
        return self._cp_info[data_id][frame_id].reshape(-1, 2)
    
    def _load_used_finger(self, data_id, frame_id):
        return self._used_finger[data_id][frame_id]
    
    def _load_static_instrument(self):
        return self._static_instrument

    def __len__(self):
        return len(self._data_ids)

    def __getitem__(self,  index):
        data_index = self._data_ids[index]
        nframes = self._num_frames[data_index]
        frame_index = np.arange(nframes)
        
        if self.split == 'test':  # for test
            if 'instrument' in self.cond_mode:
                instrument = self._load_instrument(data_index, frame_index)
                output = {'instrument': instrument}
            elif 'audio' in self.cond_mode:
                audio = self._load_audio(data_index, frame_index) # [nframes, nfeats]
                output = {'audio': audio}
                if 'stft' in self.cond_info:
                    output['stft'] = self._load_stft(data_index, frame_index)
                if 'mel' in self.cond_info:
                    output['mel'] = self._load_mel(data_index, frame_index)
                if 'cp_pos' in self.cond_info:
                    output['cp_pos'] = self._load_cp_pos(data_index, frame_index)
            return output
        
        lh_pose = self._load_lh_pose(data_index, frame_index)  # [nframes, 15joints, 6]
        lh_pose = to_torch(lh_pose)
        lh_pose = lh_pose.permute(1, 2, 0).contiguous().float()  # [15joints, 6, nframes]

        rh_pose = self._load_rh_pose(data_index, frame_index)  # [nframes, 15joints, 6]
        rh_pose = to_torch(rh_pose)

        padded_bow = torch.zeros((rh_pose.shape[0], rh_pose.shape[2]), dtype=rh_pose.dtype)  # [nframes, 6]
        bow = to_torch(self._load_bow(data_index, frame_index))
        padded_bow[:, :3] = bow
        padded_bow = padded_bow.unsqueeze(1) # [nframes, 1, 6]
        rh_pose = torch.cat((rh_pose, padded_bow), 1)  # [nframes, 15joints+1, 6]

        rh_pose = rh_pose.permute(1, 2, 0).contiguous().float()  # [15joints+1, 6, nframes]

        body = self._load_body(data_index, frame_index)  # [nframes, 21joints, 6]
        body = to_torch(body)
        body = body.permute(1, 2, 0).contiguous().float()  # [21joints, nframes, 6]

        if 'instrument' in self.cond_mode:
            instrument = self._load_instrument(data_index, frame_index) # [nframes, npoints, 3]
            # instrument = self._load_instrument()  # static instrument [npoints, 3]  TODO: DYNAMIC INSTRUMENT
            output = {'lh_pose': lh_pose, 'rh_pose': rh_pose, 'body': body, 'instrument': instrument}
        
        elif 'audio' in self.cond_mode:
            audio = self._load_audio(data_index, frame_index) # [nframes, 18]
            output = {'lh_pose': lh_pose, 'rh_pose': rh_pose, 'body': body, 'audio': audio}
            if 'stft' in self.cond_info:
                output['stft'] = self._load_stft(data_index, frame_index)
            if 'mel' in self.cond_info:
                output['mel'] = self._load_mel(data_index, frame_index)
            if 'cp_pos' in self.cond_info:
                output['cp_pos'] = self._load_cp_pos(data_index, frame_index)
            if 'cp_info' in self.cond_info:
                output['cp_info'] = self._load_cp_info(data_index, frame_index)
            if 'used_finger' in self.cond_info:
                output['used_finger'] = self._load_used_finger(data_index, frame_index)
            if 'instrument' in self.cond_info:
                output['instrument'] = self._load_static_instrument()
        
        return output

