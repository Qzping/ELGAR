from data_loaders.string_performance_dataset import StringPerformanceDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from icecream import ic


def lengths_to_mask(lengths, max_len):
    """
    Data will be masked for certain frame from certain batch, when all the joints should be masked.
    Pretty much like 'frame drop'
    """
    # max_len: max frame_num
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    # mask: torch.Size([bs, frames])
    return mask


def collate_tensors(batch):
    # batch: (batch_size, human_kp, 6d, frame_num)

    dims = batch[0].dim() # 3

    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]  # 找到每一个维度(1-3)的最大值
    # max_size: [joints, 6d, max_frames]
    
    size = (len(batch),) + tuple(max_size)
    # size: (bs, joints, 6d, max_frames)

    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        # b.shape: [joints, 6d, frames]
        sub_tensor = canvas[i]
        # sub_tensor.shape: [joints, 6d, max_frames]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))  # sub_tensor与batch对应位置的指针
        sub_tensor.add_(b)  # 将b的值赋给sub_tensor
    
    return canvas  # torch.Size([bs, joints, 6d, max_frames]) 


def collate(batch):
    motion = None  # for test
    cond = {'y':{}}
    
    notnone_batches = [b for b in batch if b is not None]
    
    if 'lh_pose' in batch[0]:  # for train
        
        databatch = []
        for b in notnone_batches:
            body = b['body']  # (bs, 21, 6d, frames)
            lh_pose = b['lh_pose']  # (bs, njoints, 6d, frames)
            rh_pose = b['rh_pose']  # (bs, njoints+1, 6d, frames)
            pose = torch.cat((body, lh_pose, rh_pose), 0)  # (bs, 2njoints+1+21, 6d, frames)
            databatch.append(pose)
        
        # databatch = [b['lh_pose'] for b in notnone_batches]  # (bs, joints, 6d, frames)

        if 'lengths' in notnone_batches[0]:
            lenbatch = [b['lengths'] for b in notnone_batches]
        else:
            lenbatch = [len(b['lh_pose'][0][0]) for b in notnone_batches]  # (bs) * frames

        databatchTensor = collate_tensors(databatch)  # torch.Size([bs, joints, 6d, frames])
        lenbatchTensor = torch.as_tensor(lenbatch)  # torch.Size([bs])
        maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unsqueeze for broadcasting [bs, 1, 1, frames]

        motion = databatchTensor
        cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    # notnone_batches = [b for b in batch if b is not None]
    # databatch = [b['lh_pose'] for b in notnone_batches]  # (bs, joints, 6d, frames)

    # if 'lengths' in notnone_batches[0]:
    #     lenbatch = [b['lengths'] for b in notnone_batches]
    # else:
    #     lenbatch = [len(b['lh_pose'][0][0]) for b in notnone_batches]  # (bs) * frames

    # databatchTensor = collate_tensors(databatch)  # torch.Size([bs, joints, 6d, frames])
    # lenbatchTensor = torch.as_tensor(lenbatch)  # torch.Size([bs])
    # maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unsqueeze for broadcasting
    # # maskbatchTensor: torch.Size([bs, 1, 1, frames])

    # motion = databatchTensor
    # cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

        
    if 'instrument' in notnone_batches[0]:
        instrumentbatch = np.array([b['instrument'] for b in notnone_batches])
        cond['y'].update({'instrument': torch.as_tensor(instrumentbatch)})
    
    if 'audio' in notnone_batches[0]:
        audiobatch = np.array([b['audio'] for b in notnone_batches])
        cond['y'].update({'audio': torch.as_tensor(audiobatch)})

    if 'stft' in notnone_batches[0]:
        stftbatch = np.array([b['stft'] for b in notnone_batches])
        cond['y'].update({'stft': torch.as_tensor(stftbatch)})
        
    if 'mel' in notnone_batches[0]:
        melbatch = np.array([b['mel'] for b in notnone_batches])
        cond['y'].update({'mel': torch.as_tensor(melbatch)})
    
    if 'cp_pos' in notnone_batches[0]:
        cpposbatch = np.array([b['cp_pos'] for b in notnone_batches])
        cond['y'].update({'cp_pos': torch.as_tensor(cpposbatch)})
    
    if 'cp_info' in notnone_batches[0]:
        cpinfobatch = np.array([b['cp_info'] for b in notnone_batches])
        cond['y'].update({'cp_info': torch.as_tensor(cpinfobatch)})
    
    if 'used_finger' in notnone_batches[0]:
        usedfingerbatch = np.array([b['used_finger'] for b in notnone_batches])
        cond['y'].update({'used_finger': torch.as_tensor(usedfingerbatch)})

    if 'instrument' in notnone_batches[0]:
        instrumentbatch = np.array([b['instrument'] for b in notnone_batches])
        cond['y'].update({'instrument': torch.as_tensor(instrumentbatch)})

    return motion, cond


def get_dataset_loader(batch_size, datapath="data_process/train_data/wholebody_processed", filename='{"audio": "audio.npy", "motion": "motion.hdf5"}', split='train', shuffle=True, **kargs):
    dataset = StringPerformanceDataset(datapath=datapath, filename=filename, split=split, **kargs)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, drop_last=False, collate_fn=collate
     )

    return loader