import torch
import numpy as np


def lengths_to_mask(lengths, max_len):
    # lengths: torch.Size([batch_size])
    # max_len: max frame_num
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    # print(mask.shape): torch.Size([64, 60])
    return mask
    

def collate_tensors(batch):
    # print(np.array(batch).shape)
    # (64, 25, 6, 60): (batch_size, human_kp, 6d, frame_num)

    dims = batch[0].dim() # 3

    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]  # find maximum between dim 1-3
    # print(max_size): [25, 6, 60]
    
    size = (len(batch),) + tuple(max_size)
    # print(size): (64, 25, 6, 60)

    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        # b.shape: [25, 6, 60]
        sub_tensor = canvas[i]
        # sub_tensor.shape: [25, 6, 60]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))  # sub_tensor与batch对应位置的指针
        sub_tensor.add_(b)  # set b value to sub_tensor
    
    # print(canvas.shape): torch.Size([64, 25, 6, 60]) 
    return canvas


def collate(batch):
    # print(len(batch))
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]

    # print(np.array(databatch).shape): (64, 25, 6, 60)
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    
    # print(lenbatch): [60] * 64

    databatchTensor = collate_tensors(databatch)  # torch.Size([64, 25, 6, 60])
    lenbatchTensor = torch.as_tensor(lenbatch)  # torch.Size([64])
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unsqueeze for broadcasting
    # print(maskbatchTensor.shape): torch.Size([64, 1, 1, 60])

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})  # shape: torch.Size([64, 1])

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        # print(np.array(action_text).shape): (64,)
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


