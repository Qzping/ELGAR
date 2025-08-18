import numpy as np
from data_process.data_manipulation import DataManipulatorCello
import os


if __name__ == "__main__":
    data_manipulator_cello = DataManipulatorCello()
    data_dir = '../dataset/SPD/cello/kp3d'
    save_dir = './mpl_results'

    instrument = 'cello'
    piece_idx = '01'

    file_name = f'{instrument}{piece_idx}'
    file_path = os.path.join(data_dir, f'{file_name}.npy')

    data = np.load(file_path, allow_pickle=True)

    vis_start = 0
    vis_end = 50

    body = data[0:50, 0:23, :]
    left_hand = data[0:50, 91:112, :]
    right_hand = data[0:50, 112:133, :]
    instrument = data[0:50, 133:140, :]
    bow = data[0:50, 140:142, :]

    instrument = data_manipulator_cello.get_second_third_string(instrument)

    vis_dir = os.path.join(save_dir, 'npy_vis')
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, f'{file_name}.mp4')

    data_manipulator_cello.visualize_3dpk(out_file=out_path,
                                          left_hand_3d_position=left_hand,
                                          right_hand_3d_position=right_hand,
                                          instrument_3d_position=instrument,
                                          body_3d_position=body,
                                          bow_3d_position=bow,
                                          plot_static_frame=False,
                                          # hand_type='mano',
                                          hand_type='coco',
                                          # body_type='smpl',
                                          body_type='coco',
                                          view='holistic'
                                          )

