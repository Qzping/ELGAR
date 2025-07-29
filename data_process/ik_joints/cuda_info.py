import numpy as np
import torch
import subprocess

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

# Obtain the usage of memory for each GPU
gpu_memory_usage = get_gpu_memory_usage()
if gpu_memory_usage:
    for idx, info in enumerate(gpu_memory_usage):
        print(f"GPU {idx}: Total Memory: {info['total_memory']} MiB, "
              f"Used Memory: {info['used_memory']} MiB, "
              f"Free Memory: {info['free_memory']} MiB")
              

print("gpu_memory_usage: ",gpu_memory_usage)
gpu_free_memory = [gfm['free_memory'] for gfm in gpu_memory_usage]
print("gpu_free_memory: ",gpu_free_memory)
#cuda_device = np.argmax(gpu_memory_usage['free_memory'])