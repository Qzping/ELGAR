import argparse
import json
import math
import os.path
import cv2
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import freq_position as freq_position
import librosa


def draw_fundamental_curve(time_arr, freq_arr, conf_arr, proj, algo):
    fig = plt.figure(figsize=(10, 8))
    # percentage of axes occupied
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.8])
    # axes.plot(time_arr, freq_arr, ls=':', alpha=1, lw=1, zorder=1)
    ax = axes.scatter(time_arr, freq_arr, c=conf_arr, s=1.5, cmap="OrRd")
    axes.set_title('Pitch Curve')
    fig.colorbar(ax)
    # plt.show()

    if not os.path.exists(f'output/{proj}'):
        os.makedirs(f'output/{proj}', exist_ok=True)
    plt.savefig(f'output/{proj}/pitch_curve_{algo}.jpg')


def draw_contact_points(data, file_name, proj):
    """
    data: [n, 4], n: frame number, 4: string number
    """
    string_low = 1
    string_high = 10
    string_length = string_high - string_low

    # string
    string_xloc = [4, 3, 2, 1]

    # Define the data for four long vertical strings with two points each
    string1 = [(string_xloc[0], string_low), (string_xloc[0], string_high)]
    string2 = [(string_xloc[1], string_low), (string_xloc[1], string_high)]
    string3 = [(string_xloc[2], string_low), (string_xloc[2], string_high)]
    string4 = [(string_xloc[3], string_low), (string_xloc[3], string_high)]

    # Extract x and y coordinates for each string
    x1, y1 = zip(*string1)
    x2, y2 = zip(*string2)
    x3, y3 = zip(*string3)
    x4, y4 = zip(*string4)

    if not os.path.exists(f'output/{proj}'):
        os.makedirs(f'output/{proj}', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{proj}/{file_name}.avi', fourcc, fps=30, frameSize=[700, 1000])
    for frame_idx, frame in enumerate(data):
        print(f'Frame {frame_idx + 1}...')
        points = []
        for idx, ratio in enumerate(frame):
            if ratio >= 0:
                points.append([string_xloc[idx], ratio * string_length + string_low])

        # Create a figure and axis without axes
        fig, ax = plt.subplots(figsize=(7, 10))

        ax.set_xlim(0.5, 4.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # matplotlib default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()['color']

        # Plot the four long vertical strings with only two points each
        string1 = ax.plot(x1, y1, marker='o', label='String 1 A', color=colors[0])
        string2 = ax.plot(x2, y2, marker='o', label='String 2 D', color=colors[1])
        string3 = ax.plot(x3, y3, marker='o', label='String 3 G', color=colors[2])
        string4 = ax.plot(x4, y4, marker='o', label='String 4 C', color=colors[4])

        if points:
            for point in points:
                # zorder should be larger than the string number
                ax.scatter(point[0], point[1], color='r', zorder=5)
        else:
            print(f'No points detected in Frame {frame_idx + 1}')

        # Add a title and legend
        ax.set_title(f'Cello Strings (Frame {frame_idx + 1})')
        num1, num2, num3, num4 = 1.03, 0, 3, 0
        ax.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

        # Swap the aspect ratio of the entire image
        ax.set_aspect(1)

        # Display the plot
        # plt.show()

        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(height, width, 3)
        image_array = image_array[:, :, ::-1]  # rgb to bgr
        out.write(image_array)
        plt.close()


def get_sampling_rate_pydub(file_path):
    audio = AudioSegment.from_file(file_path)
    sample_rate = audio.frame_rate
    return sample_rate


# 定义带通滤波器
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# 应用带通滤波器
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    y[np.isnan(y)] = 0
    return np.asarray(y, dtype=np.float32)


def adjust(arr):
    import torch
    import numpy as np
    output_var_type = None
    if isinstance(arr, np.ndarray):
        output_var_type = np.ndarray
    elif isinstance(arr, list):
        output_var_type = list
    elif isinstance(arr, torch.Tensor):
        output_var_type = torch.Tensor
    else:
        output_var_type = np.ndarray

    arr = np.asarray(arr)
    arr_ori_shape = arr.shape

    if len(arr.shape) == 2:
        arr = arr.reshape(-1)
    # print(arr.shape)
    for i in range(arr.shape[0]):  # arr.shape[0]
        if i == 0:
            if abs(arr[i] - arr[i + 1]) > 30 and abs(arr[i] - arr[i + 2]) > 30:
                arr[i] = np.mean([arr[i + 1], arr[i + 2]])
        elif i == arr.shape[0] - 1:  #
            if abs(arr[i] - arr[i - 1]) > 30 and abs(arr[i] - arr[i - 2]) > 30:
                arr[i] = np.mean([arr[i - 1], arr[i - 2]])
        else:
            if abs(arr[i] - arr[i - 1]) > 30 and abs(arr[i] - arr[i + 1]) > 30:
                arr[i] = np.mean([arr[i - 1], arr[i + 1]])

    if output_var_type == np.ndarray:
        return arr.reshape(arr_ori_shape)
    elif output_var_type == list:
        return arr.tolist()
    elif output_var_type == torch.Tensor:
        return torch.tensor(arr.reshape(arr_ori_shape))
    else:
        return arr.reshape(arr_ori_shape)


def pitch_detect_crepe(crepe_backend, proj, instrument='cello', audio_path='wavs/background.wav'):
    # viterbi: smoothing for the pitch curve
    # step_size: 10 milliseconds
    # center: False, don't need to pad!

    if crepe_backend == 'torch':
        import torchcrepe
        import torch
        # audio, sr = torchcrepe.load.audio(audio_path)
        sr = get_sampling_rate_pydub(audio_path)

        audio, sr = librosa.load(audio_path, sr=sr, mono=True)

        if instrument == 'cello':
            freq_range = freq_position.PITCH_RANGES_CELLO
        else:
            freq_range = freq_position.PITCH_RANGES_VIOLIN
        min_freq = np.min(freq_range)
        max_freq = np.max(freq_range)

        audio_filtered = bandpass_filter(audio, min_freq, max_freq, sr, order=2)

        audio_1channel = torch.tensor(audio_filtered).reshape(1, -1)
        sample_num = audio_1channel.shape[1]

        frame_num = math.floor(30 * sample_num / sr)
        frequency, confidence = torchcrepe.predict(audio_1channel,
                                                   sr,
                                                   hop_length=int(sr / 30.),
                                                   return_periodicity=True,
                                                   model='full',
                                                   fmin=min_freq,
                                                   fmax=max_freq,
                                                   batch_size=2048,
                                                   device='cuda:0',  # cuda:0
                                                   decoder=torchcrepe.decode.weighted_argmax)  # torchcrepe.decode.viterbi

        # We'll use a 15 millisecond window assuming a hop length of 5 milliseconds
        win_length = 3

        # Median filter noisy confidence value
        confidence = torchcrepe.filter.median(confidence, win_length)

        frequency = adjust(frequency)

        # Remove inharmonic regions
        frequency = torchcrepe.threshold.At(.21)(frequency, confidence)

        # Optionally smooth pitch to remove quantization artifacts
        frequency = torchcrepe.filter.mean(frequency, win_length)

        frequency = frequency[~torch.isnan(frequency)]

        min_freq = torch.floor(torch.min(frequency))
        max_freq = torch.ceil(torch.max(frequency))

        frequency, confidence = torchcrepe.predict(audio_1channel,
                                                   sr,
                                                   hop_length=int(sr / 30.),
                                                   return_periodicity=True,
                                                   model='full',
                                                   fmin=min_freq,
                                                   fmax=max_freq,
                                                   batch_size=2048,
                                                   device='cuda:0',  # cuda:0
                                                   decoder=torchcrepe.decode.viterbi)  # torchcrepe.decode.viterbi torchcrepe.decode.weighted_argmax

        confidence = torchcrepe.filter.median(confidence, win_length)
        # frequency = torchcrepe.threshold.At(.21)(frequency, confidence)
        frequency = torchcrepe.filter.mean(frequency, win_length)

        frequency = frequency.reshape(-1, )
        confidence = confidence.reshape(-1, )
        time = np.arange(0, 100 / 3. * frequency.size()[0], 100 / 3.).reshape(-1, )[:len(frequency)]

    elif crepe_backend == 'tensorflow':
        import crepe
        sr, audio = wavfile.read(audio_path)
        sample_num = audio.shape[0]
        frame_num = math.floor(30 * sample_num / sr)
        time, frequency, confidence, activation = crepe.predict(
            audio, sr, viterbi=True, step_size=100 / 3, model_capacity='full', center=True)
    else:
        raise Exception('the argument "crepe_backend" is either "tensorflow" or "torch"')

    # draw_fundamental_curve(time, frequency, confidence, proj, 'crepe')

    pitch_results = np.stack((time, frequency, confidence), axis=1)
    # Pitch Data Persistence
    # np.savetxt("pitch.csv", pitch_results, delimiter=",")
    pitch_results = pitch_results[:frame_num, :]
    return pitch_results


if __name__ == '__main__':
    crepe_backend = 'torch'
    instrument = 'cello'

    proj = 'yoyoma'
    wav_path = 'wavs_test/yoyoma.wav'

    pitch_results = pitch_detect_crepe(crepe_backend, proj, instrument, wav_path)
    ic(pitch_results.shape)

    pitch_with_positions = freq_position.get_contact_position(pitch_results, instrument)
    positions = pitch_with_positions[:, -4:]
    ic(positions.shape)
    # draw_contact_points(positions, proj, 'virtual_contact_point')

    ic(positions[30:50])








