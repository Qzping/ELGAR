import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from icecream import ic
import librosa
import soundfile as sf
import torch
import numpy as np


def resample_audio(wav_file, target_sr=44100, mono=False, persist=False):
    """
    Resample the audio to the target sample rate
    :param wav_file: wav file path
    :param target_sr: default target sampling rate 44100
    :param mono: whether to use mono channel
    :param persist: whether to save the audio
    :return:
    """
    # Load audio file
    # librosa version: 0.10.2
    # if mono=False, y = [channels, samples]
    y, sr = librosa.load(wav_file, sr=None, mono=mono)

    if sr != target_sr:
        print(f"Detected sampling rate: {sr}. Resampling to {target_sr}.")
        y_resampled = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        if persist:
            # soundfile version: 0.12.1
            # sf.write(file_name, audio_signal, sample_rate), where audio_signal should be [samples, channels]
            sf.write(wav_file, y_resampled.T, target_sr)
        y = y_resampled
        print(f"Resampled from {sr} to {target_sr}")

    return y, target_sr


def extract_dac_feats(audio_path, seq_len, hop_len):
    """
    Extract features from audio file
    Args:
        audio_path: audio file path
        seq_len: sequence length in seconds
        hop_len: hop length in seconds
    Returns:
        codes: audio features [bs, nframes, nfeats]
    """
    import dac

    current_file_path = os.path.abspath(__file__)

    current_file_directory = os.path.dirname(current_file_path)
    model_path = os.path.join(current_file_directory, 'model/weights_44khz_16kbps_1.0.0.pth')
    model = dac.DAC.load(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    target_sr = 44100
    # y, sr = librosa.load(audio_path, sr=None, mono=True)
    y, sr = resample_audio(audio_path, target_sr=target_sr, mono=True, persist=False)

    audio_seq_len = int(target_sr * seq_len)
    audio_hop_len = int(target_sr * hop_len)

    ##################### if you have a large GPU memery #####################

    # sig_arr = []
    # for start_sample in range(0, y.shape[0], audio_hop_len):
    #     end_sample = start_sample + audio_seq_len
    #     seq_y = y[start_sample:end_sample]
    #     signal = AudioSignal(seq_y, sample_rate=target_sr)
    #     sig_arr.append(signal)
    #
    # batch = AudioSignal.batch(sig_arr, pad_signals=True)
    # batch.to(model.device)
    # x = model.preprocess(batch.audio_data, batch.sample_rate)
    # z, codes, latents, _, _ = model.encode(x)  # codes: [bs, n_q, T]
    # audio_freq = codes.shape[-1]  # if sr == 44100, then it should be (seq_len * 86 + 1)
    # pose_freq = 30 * 5
    # codes = np.asfortranarray(codes.cpu(), dtype=np.float32)
    # codes = librosa.resample(codes, orig_sr=audio_freq, target_sr=pose_freq, axis=2)
    # codes = codes.transpose(0, 2, 1)

    ##################### if you have large GPU memery #####################

    ##################### if you have limited GPU memory #####################

    codes = []
    for start_sample in range(0, y.shape[0], audio_hop_len):
        end_sample = start_sample + audio_seq_len
        seq_y = y[start_sample:end_sample]
        signal = AudioSignal(seq_y, sample_rate=target_sr)
        pad_len = audio_seq_len - signal.signal_length
        signal.audio_data = torch.nn.functional.pad(signal.audio_data, (0, pad_len))  # default pad 0
        signal.to(device)
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        _, code, _, _, _ = model.encode(x)  # codes: [1, n_q, T]
        audio_freq = code.shape[-1]  # if sr == 44100, then it should be (seq_len * 86 + 1)
        pose_freq = 30 * seq_len
        code = np.asfortranarray(code.cpu(), dtype=np.float32)
        code = librosa.resample(code, orig_sr=audio_freq, target_sr=pose_freq, axis=2)
        code = code.transpose(0, 2, 1).squeeze()
        codes.append(code)
    codes = np.stack(codes)
    ic(codes.shape)

    ##################### if you have limited GPU memory #####################

    return codes


def extract_encodec_feats(audio_path, seq_len, hop_len):
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    from pathlib import Path
    import torchaudio

    current_file_path = os.path.abspath(__file__)

    current_file_directory = os.path.dirname(current_file_path)
    model_path = Path(os.path.join(current_file_directory, 'model'))

    target_sr = 24000

    audio_seq_len = int(target_sr * seq_len)
    audio_hop_len = int(target_sr * hop_len)

    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz(repository=model_path)
    model.set_target_bandwidth(24)

    # Load and pre-process the audio waveform
    y, sr = torchaudio.load(audio_path)
    y = convert_audio(y, sr, model.sample_rate, model.channels)
    y = y.squeeze()

    sig_arr = []
    for start_sample in range(0, y.shape[0], audio_hop_len):
        end_sample = start_sample + audio_seq_len
        seq_y = y[start_sample:end_sample]
        signal = AudioSignal(seq_y, sample_rate=target_sr)
        sig_arr.append(signal)

    batch = AudioSignal.batch(sig_arr, pad_signals=True)
    audio_data = batch.audio_data
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(audio_data)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    audio_freq = codes.shape[-1]  # it should be (seq_len * 75)
    pose_freq = 30 * seq_len
    codes = np.asfortranarray(codes.cpu(), dtype=np.float32)
    codes = librosa.resample(codes, orig_sr=audio_freq, target_sr=pose_freq, axis=2)
    codes = codes.transpose(0, 2, 1)

    return codes


def extract_jukebox_feats(audio_path, seq_len, hop_len, model_path='./model'):
    import jukemirlib

    target_sr = jukemirlib.JUKEBOX_SAMPLE_RATE

    audio_seq_len = int(target_sr * seq_len)
    audio_hop_len = int(target_sr * hop_len)

    y = jukemirlib.load_audio(audio_path)

    ##################### if you have a large GPU memery #####################

    # sig_arr = []
    # for start_sample in range(0, y.shape[0], audio_hop_len):
    #     end_sample = start_sample + audio_seq_len
    #     seq_y = y[start_sample:end_sample]
    #     signal = AudioSignal(seq_y, sample_rate=target_sr)
    #     sig_arr.append(signal)
    #
    # batch = AudioSignal.batch(sig_arr, pad_signals=True)
    # audio_data = batch.audio_data
    # audio_data = audio_data.squeeze()
    # audio_data = audio_data.numpy()
    # # audio_data = audio_data[:2]
    # audio_data = np.split(audio_data, audio_data.shape[0], axis=0)
    # audio_data = [data.squeeze() for data in audio_data]
    #
    # reps = jukemirlib.extract(audio_data, layers=[66])
    # codes = reps[66]  # (seq_len, 4800)

    # ic(codes.shape)
    # codes = codes.transpose(0, 2, 1)
    # audio_freq = codes.shape[-1]
    # pose_freq = 30 * seq_len
    # codes = librosa.resample(codes, orig_sr=audio_freq, target_sr=pose_freq, axis=-1)
    # codes = codes.transpose(0, 2, 1)
    # ic(codes.shape)

    ##################### if you have a large GPU memery #####################

    ##################### if you have limited GPU memory #####################

    codes = []
    for start_sample in range(0, y.shape[0], audio_hop_len):
        end_sample = start_sample + audio_seq_len
        seq_y = y[start_sample:end_sample]
        # signal = AudioSignal(seq_y, sample_rate=target_sr)
        pad_len = audio_seq_len - seq_y.shape[0]
        seq_y = np.pad(seq_y, (0, pad_len))  # default pad 0
        reps = jukemirlib.extract(seq_y, layers=[66], cache_dir=model_path)
        code = reps[66]
        code = code.T
        audio_freq = code.shape[-1]
        pose_freq = 30 * seq_len
        code = librosa.resample(code, orig_sr=audio_freq, target_sr=pose_freq, axis=-1)
        code = code.T
        codes.append(code)
    codes = np.stack(codes)
    print('codes:', codes.shape)

    ##################### if you have limited GPU memory #####################

    return codes


def extract_wav2clip_feats(audio_path, seq_len, hop_len):
    import wav2clip

    target_sr = 44100
    audio_seq_len = int(target_sr * seq_len)
    audio_hop_len = int(target_sr * hop_len)
    num_frames = 30
    frame_len = target_sr // num_frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    model = wav2clip.get_model(device = device, frame_length=frame_len, hop_length=frame_len)    # non-overlapping hop

    feats = []
    for start_sample in range(0, y.shape[0], audio_hop_len):
        start_time = time.time()
        end_sample = start_sample + audio_seq_len
        seq_y = y[start_sample:end_sample]
        pad_len = audio_seq_len - seq_y.shape[0]
        seq_y = np.pad(seq_y, (0, pad_len))  # default pad 0
        feat = wav2clip.embed_audio(seq_y, model)   # [channels, dim, frames]
        feat = feat.squeeze(0)  # [dim, frames]
        feats.append(feat)
    feats = np.stack(feats)
    feats = feats.transpose(0, 2, 1)
    return feats


def extract_stft_feats(audio_path, seq_len, hop_len):

    n_fft = 4096
    stft_hop_len = 1472 # sr // num_frames + 1
    stft_win_len = n_fft
    window = "hann"

    target_sr = 44100
    audio_seq_len = int(target_sr * seq_len)
    audio_hop_len = int(target_sr * hop_len)

    y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    print(y.shape)

    feats = []
    for start_sample in range(0, y.shape[0], audio_hop_len):
        # start_time = time.time()
        end_sample = start_sample + audio_seq_len
        seq_y = y[start_sample:end_sample]
        pad_len = audio_seq_len - seq_y.shape[0]
        seq_y = np.pad(seq_y, (0, pad_len))  # default pad 0
        feat = librosa.stft(seq_y, n_fft=n_fft, hop_length=stft_hop_len, win_length=stft_win_len, window=window)
        feat = np.abs(feat)
        feats.append(feat)
        # end_time = time.time()
        # elaps_time = end_time - start_time
        # print(f"Consuming {elaps_time} seconds.")
    feats = np.stack(feats)
    feats = feats.transpose(0, 2, 1)
    # ic(feats.shape)
    return feats


def extract_melspec_feats(audio_path, seq_len, hop_len):

    n_fft = 4096
    stft_hop_len = 1472 # sr // num_frames + 1
    stft_win_len = n_fft
    window = "hann"
    n_mels = 512

    target_sr = 44100
    audio_seq_len = int(target_sr * seq_len)
    audio_hop_len = int(target_sr * hop_len)

    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    print(y.shape)

    feats = []
    for start_sample in range(0, y.shape[0], audio_hop_len):
        # start_time = time.time()
        end_sample = start_sample + audio_seq_len
        seq_y = y[start_sample:end_sample]
        pad_len = audio_seq_len - seq_y.shape[0]
        seq_y = np.pad(seq_y, (0, pad_len))  # default pad 0
        feat = librosa.feature.melspectrogram(y=seq_y, sr=sr, n_fft=n_fft, hop_length=stft_hop_len, win_length=stft_win_len, window=window, n_mels = n_mels)
        feat = librosa.amplitude_to_db(feat, ref=np.max)
        feats.append(feat)
        # end_time = time.time()
        # elaps_tme = end_time - start_time
        # print(f"Consuming {elaps_tme} seconds.")
    feats = np.stack(feats)
    feats = feats.transpose(0, 2, 1)
    return feats
