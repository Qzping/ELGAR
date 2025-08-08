import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import glob
import os
import json
import cv2
import time
# import open3d as o3d
# import open3d.visualization.rendering as rendering
from scipy.interpolate import splrep, splev
from scipy.spatial.transform import Rotation, Slerp
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from icecream import ic
from scipy.spatial.transform import Rotation as R
from sympy import rotations
from tqdm import tqdm
import shutil


class DataManipulatorCello:
    def __init__(self, seq_length=150, hop_length=30, bodycolor="blue", hand_type='mano', body_type='smpl'):
        self.seq_length = seq_length
        self.hop_length = hop_length
        self.end_pin_index = 139
        self.lh_wrist_index = 91
        self.instrument_idx_range = (133, 142)
        # self.instrument_links = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 5], [5, 6], [7, 8]]  # with bow
        # self.instrument_links = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 5], [5, 6]]  # without bow
        # self.instrument_links = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 5], [5, 6], [7, 9], [8, 10]]  # with 2nd and 3rd string
        self.instrument_links = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 5], [5, 6], [7, 9], [8, 10],
                                 [1, 7], [7, 8], [2, 8], [3, 9], [9, 10], [4, 10]]  # with 2nd and 3rd string
        self.bow_links = [[0, 1]]

        self.hand_mano_links = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [0, 10],
                                [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [3, 16], [6, 17], [9, 18], [12, 19],
                                [15, 20]]
        self.hand_coco_links = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                                [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19],
                                [19, 20]]

        self.body_coco_links = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                                [6, 8],
                                [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [15, 17],
                                [15, 18],
                                [15, 19], [16, 20], [16, 21], [16, 22]]  # coco

        self.body_smpl_links = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10],
                                [8, 11],
                                [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20],
                                [19, 21]]  # smpl

        # self.hand_links = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
        #                    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19],
        #                    [19, 20]]  # coco links

        if hand_type == 'mano':
            self.hand_links = self.hand_mano_links
        elif hand_type == 'coco':
            self.hand_links = self.hand_coco_links
        else:
            raise ValueError("Hand type is not supported")

        if body_type == 'smpl':
            self.body_links = self.body_smpl_links
        elif body_type == 'coco':
            self.body_links = self.body_coco_links
        else:
            raise ValueError('Body type not supported')

        self.mano_parents_indices = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]
        self.bone_lengths = np.array([0.094, 0.0335, 0.024, 0.094, 0.034, 0.03, 0.081, 0.029, 0.019, 0.087,
                                      0.035, 0.027, 0.039, 0.034, 0.032, 0.019, 0.022, 0.019, 0.022,
                                      0.023])  # in meter
        self.fingerboard_idx = [1, 2, 3, 4]

        self.color_vis_o3d = [90 / 255, 135 / 255, 247 / 255]
        self.color_occ_o3d = [219 / 255, 199 / 255, 123 / 255]
        self.color_gt_o3d = [255 / 255, 102 / 255, 102 / 255]

        self.color_instrument = [139 / 255, 93 / 255, 56 / 255]  # 木质深褐色
        self.color_instrument = [85 / 255, 86 / 255, 82 / 255]  # 木质深褐色
        self.color_instrument = [190 / 255, 200 / 255, 200 / 255]  # 木质深褐色
        self.color_instrument = [146 / 255, 117 / 180, 117 / 255]  # 木质深褐色      # candidate
        # self.color_instrument = [22 / 255, 179 / 180, 11 / 255]                     # candidate
        self.color_instrument = [224 / 255, 179 / 255, 54 / 255]  # candidate
        self.color_instrument = [103 / 255, 68 / 255, 3 / 255]  # candidate
        # self.color_instrument = [194 / 255, 174 / 255, 145 / 255]  # candidate
        self.color_instrument = [120 / 255, 110 / 255, 107 / 255]  # candidate

        # self.color_bow = [177 / 255, 36 / 255, 9 / 255]  # 中性色绿色灰色
        # self.color_bow = [128 / 255, 4 / 255, 4 / 255]  # 中性色绿色灰色
        # self.color_bow = [182 / 255, 105 / 255, 5 / 255]  # 中性色绿色灰色
        # self.color_bow = [145 / 255, 66 / 255, 6 / 255]  # 中性色绿色灰色
        self.color_bow = [110 / 255, 40 / 255, 2 / 255]  # 中性色绿色灰色

        if bodycolor == "purple":
            bodyr, bodyg, bodyb = 90, 31, 117
        elif bodycolor == "blue":
            bodyr, bodyg, bodyb = 61, 94, 191
        elif bodycolor == "pink":
            bodyr, bodyg, bodyb = 148, 55, 109
        elif bodycolor == "orange":
            bodyr, bodyg, bodyb = 179, 58, 34
        else:
            print("NOT VALIDATE COLOR! SET PURPLE AS DEFAULT!")
            bodyr, bodyg, bodyb = 61, 94, 191

        # self.color_bow = [200 / 255, 260 / 255, 12 / 255]
        # self.color_body = [193 / 255, 199 / 255, 210 / 255]  # 柔和灰色
        # self.color_body_joint = [255 / 255, 169 / 255, 169 / 255]
        # self.color_body_joint = [185 / 255, 190 / 255, 200 / 255]
        # self.color_body = [61 / 255, 94 / 255, 191 / 255]  # 柔和灰色
        self.color_body = [bodyr/255, bodyg/255, bodyb/255]

        # self.color_body_joint = [32 / 255, 52 / 255, 122 / 255]
        self.color_body_joint = [(bodyr-5)/255, (bodyg-5)/255, (bodyb-5)/255]
        # self.color_cp = [133 / 255, 252 / 255, 7 / 255]
        self.color_cp = [252 / 255, 0 / 255, 0 / 255]

        # self.color_lefthand = [10 / 255, 80 / 255, 235 / 255]  # 深蓝色
        # self.color_righthand = [10 / 255, 80 / 255, 235 / 255]  # 深蓝色
        # self.color_lefthand_joint = [178 / 255, 127 / 255, 190 / 255]
        # self.color_righthand_joint =  [178 / 255, 127 / 255, 190 / 255]

        # self.color_lefthand = [223 / 255, 166 / 255, 117 / 255]  # 深蓝色
        # self.color_righthand = [223 / 255, 166 / 255, 117 / 255]  # 深蓝色
        # self.color_lefthand_joint = [247 / 255, 160 / 255, 100 / 255]
        # self.color_righthand_joint = [247 / 255, 160 / 255, 100 / 255]

        self.color_lefthand = [249 / 255, 174 / 255, 120 / 255]  # 深蓝色
        self.color_righthand = [249 / 255, 174 / 255, 120 / 255]  # 深蓝色
        self.color_lefthand_joint = [249 / 255, 174 / 255, 120 / 255]
        self.color_righthand_joint = [249 / 255, 174 / 255, 120 / 255]

        # handr = 123
        # handg = 63
        # handb = 0
        # self.color_lefthand = [handr / 255, handg / 255, handb / 255]  # 深蓝色
        # self.color_righthand = [handr / 255, handg / 255, handb / 255]  # 深蓝色
        # self.color_lefthand_joint = [handr / 255, handg / 255, handb / 255]
        # self.color_righthand_joint = [handr / 255, handg / 255, handb / 255]


    def get_train_data(self, npy_dir, out_file):
        """
        :param npy_dir: npy directory storing all the cello data in 3dkp
        :param out_file: output file name, in json
        """
        data_all = {}
        instrument_sequences = []
        lh_trans_sequences = []
        lh_pose_sequences = []

        print("Getting train data in progress ...")
        data_path_list = self._find_npy_files(npy_dir)

        for data_path in data_path_list:
            data = np.load(data_path)
            data_norm, lh_trans = self._normalize_keypoints_cello(data)

            data_instrument = data_norm[:, self.instrument_idx_range[0]:self.instrument_idx_range[1], :]

            # match wih the corresponding rot vec
            piece_idx = data_path.split('.')[1][-2:]  # if current is cello01, then return '01'
            rot_file = f"./ik_result/cello/cello{piece_idx}/hand_rot_vec.json"
            with open(rot_file, 'r') as file:
                data_lh_pose = np.array(json.load(file)['hand_rot_vec'])[:, :16, :]
            assert len(data_lh_pose) == len(data_norm)

            for start_frame in range(0, len(data_norm), self.hop_length):
                end_frame = start_frame + self.seq_length
                seq_instru = data_instrument[start_frame: end_frame]
                seq_lh_trans = lh_trans[start_frame: end_frame]
                seq_lh_pose = data_lh_pose[start_frame: end_frame]

                if len(seq_instru) < self.seq_length:
                    padding_length = self.seq_length - len(seq_instru)
                    seq_instru = np.pad(seq_instru, ((0, padding_length), (0, 0), (0, 0)), mode='constant',
                                        constant_values=0.)  # (90, 9, 3)
                    seq_lh_trans = np.pad(seq_lh_trans, ((0, padding_length), (0, 0), (0, 0)), mode='constant',
                                          constant_values=0.)  # (90, 1, 3)
                    seq_lh_pose = np.pad(seq_lh_pose, ((0, padding_length), (0, 0), (0, 0)), mode='constant',
                                         constant_values=0.)  # (90, 16, 3)
                seq_instru = np.reshape(seq_instru, (len(seq_instru), seq_instru.shape[1] * seq_instru.shape[2]))
                seq_lh_trans = np.reshape(seq_lh_trans,
                                          (len(seq_lh_trans), seq_lh_trans.shape[1] * seq_lh_trans.shape[2]))

                # for hand pose, convert rot vector to 6d
                seq_lh_pose_flat = seq_lh_pose.reshape(-1, 3)  # shape in (90*16, 3) for batch conversion
                seq_lh_pose_matrix = Rotation.from_rotvec(seq_lh_pose_flat).as_matrix()  # shape in (90*16, 3, 3)
                # ic(seq_lh_pose_matrix[0])
                seq_lh_pose_6d = seq_lh_pose_matrix[:, :2, :].reshape(len(seq_lh_pose), seq_lh_pose.shape[
                    1] * 6)  # 这里很必要，matrix转换6是[:, :2, :], 不是[:, :, :2]!!
                # ic(seq_lh_pose_6d[0])
                instrument_sequences.append(seq_instru)
                lh_trans_sequences.append(seq_lh_trans)
                lh_pose_sequences.append(seq_lh_pose_6d)
                # break
            # break

        data_all['instrument_pos'] = np.array(instrument_sequences).tolist()
        data_all['lh_trans'] = np.array(lh_trans_sequences).tolist()
        data_all['lh_pose'] = np.array(lh_pose_sequences).tolist()

        ic(np.array(data_all['instrument_pos']).shape)
        ic(np.array(data_all['lh_trans']).shape)
        ic(np.array(data_all['lh_pose']).shape)

        with open(out_file, 'w') as file:
            json.dump(data_all, file, indent=4)
        print(f"Output file is {out_file}")

    def rot2position_hand(self, hand_trans, hand_6d_96, hand_type='left', smooth=False, customize_bone=False):
        """
        :param hand_trans: numpy array in the shape of (batch_size, 3)
        :param hand_6d_96: numpy array in the shape of (batch_size, 96)
        :return: left_hand_3d_position: numpy array in the shape of (batch_size, 21, 3), 3d location of each joint.
        """
        # print("6d to position converting ...")
        # convert 6d to rotation matrix
        hand_6d_flat = hand_6d_96.reshape(-1, 6)
        hand_matrices = self.rotation_6d_to_R(hand_6d_flat).reshape(-1, 16, 3, 3)
        if smooth:
            hand_matrices = self.smooth_rot(hand_matrices)
        # from rotation matrices to 3d position by Forward Kinematics
        if hand_type == 'left':
            init_pose_batch = self._get_init_pose_batch(hand_trans, 'left', mano=not customize_bone)
        elif hand_type == 'right':
            init_pose_batch = self._get_init_pose_batch(hand_trans, 'right', mano=not customize_bone)
        else:
            raise TypeError("hand_type must be either 'left' or 'right'")

        if customize_bone:
            hand_3d_position = self._get_joint_positions_batch(init_pose_batch, hand_matrices,
                                                               self.mano_parents_indices,
                                                               bone_lengths=self.bone_lengths)
        else:
            hand_3d_position = self._get_joint_positions_batch(init_pose_batch, hand_matrices,
                                                               self.mano_parents_indices)

        return hand_3d_position

    def visualize_3dpk(self, out_file, left_hand_3d_position=None, right_hand_3d_position=None,
                       instrument_3d_position=None, bow_3d_position=None, body_3d_position=None,
                       cp_3d_position=None, plot_static_frame=False, plot_hand_indices=True,
                       hand_type='mano', body_type='smpl', view='instrument', highlight=False,
                       string_audio_position=None, string_lh_position=None,
                       string_rh_position=None):
        """
        Output video for visualization
        :param out_file: .mp4 file
        :param left_hand_3d_position: numpy array in the shape of (frame_size, 21, 3), without visualizing when leaving as 'None'
        :param instrument_3d_position: numpy array in the shape of (frame_size, 9, 3), without visualizing when leaving as 'None'
        :param plot_static_frame: bool
        :param view: str in ['finger', 'instrument', 'holistic']
        """
        print("Visualizing ...")

        # visualization settings
        frame_size = (1600, 1200)
        fps = 30

        # video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_file, fourcc, fps, frame_size)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        # plot for each frame, if unequal length for instru and hand, then take the longer one
        frame_num = max(instrument_3d_position.shape[0] if instrument_3d_position is not None else 0,
                        left_hand_3d_position.shape[0] if left_hand_3d_position is not None else 0,
                        right_hand_3d_position.shape[0] if right_hand_3d_position is not None else 0,
                        bow_3d_position.shape[0] if bow_3d_position is not None else 0,
                        body_3d_position.shape[0] if body_3d_position is not None else 0)
        for f in range(frame_num):
            ax.clear()  # clear the former results
            plot_data = []
            points_lh = np.array([])
            points_instru = np.array([])
            # instruments points
            if instrument_3d_position is not None:
                points_instru = instrument_3d_position[f]
                x_instru = points_instru[:, 0]
                y_instru = points_instru[:, 1]
                z_instru = points_instru[:, 2]

                colors_instru = ['b'] * len(points_instru)
                colors_instru[-2] = 'k'  # frog
                colors_instru[-1] = 'c'  # tip
                ax.scatter(x_instru, y_instru, z_instru, c=colors_instru, zorder=2)

                lines_instru = [[points_instru[link[0]], points_instru[link[1]]] for link in self.instrument_links]
                line_collection_instru = Line3DCollection(lines_instru, edgecolors='saddlebrown', linewidths=2,
                                                          zorder=2)
                ax.add_collection(line_collection_instru)
                if len(plot_data) == 0:
                    plot_data = points_instru
                else:
                    plot_data = np.concatenate((plot_data, points_instru), axis=0)

            if bow_3d_position is not None:
                points_bow = bow_3d_position[f]
                x_bow = points_bow[:, 0]
                y_bow = points_bow[:, 1]
                z_bow = points_bow[:, 2]
                colors_bow = ['b'] * len(points_bow)
                colors_bow[-2] = 'k'  # frog
                colors_bow[-1] = 'c'  # tip
                ax.scatter(x_bow, y_bow, z_bow, c=colors_bow, zorder=2)

                lines_bow = [[points_bow[0], points_bow[1]]]
                line_collection_bow = Line3DCollection(lines_bow, edgecolors='saddlebrown', linewidths=2,
                                                       zorder=2)
                ax.add_collection(line_collection_bow)
                if len(plot_data) == 0:
                    plot_data = points_bow
                else:
                    plot_data = np.concatenate((plot_data, points_bow), axis=0)

            if left_hand_3d_position is not None:
                if hand_type == 'mano':
                    hand_links = self.hand_mano_links
                elif hand_type == 'coco':
                    hand_links = self.hand_coco_links
                else:
                    raise ValueError("Hand type is not supported")

                points_lh = left_hand_3d_position[f]
                x_lh = points_lh[:, 0]
                y_lh = points_lh[:, 1]
                z_lh = points_lh[:, 2]
                # ax.scatter(x_lh, y_lh, z_lh, c='r', s=5)
                ax.scatter(x_lh, y_lh, z_lh, c='b', s=5, zorder=3)

                if plot_hand_indices:
                    indices = np.arange(len(x_lh))
                    for i in range(len(x_lh)):
                        ax.text(x_lh[i], y_lh[i], z_lh[i], str(indices[i]), color='black')

                # set hand colors
                hand_line_colors = ['b'] * 20
                # thumb in red
                # hand_line_colors[12:15] = ['r'] * 3
                # hand_line_colors[19] = 'r'
                # index finger in yellow
                # hand_line_colors[0:3] = ['y'] * 3
                # hand_line_colors[15] = 'y'

                lines_lh = [[points_lh[link[0]], points_lh[link[1]]] for link in hand_links]
                line_collection_lh = Line3DCollection(lines_lh, edgecolors=hand_line_colors, linewidths=2, zorder=3)
                ax.add_collection(line_collection_lh)

                if len(plot_data) == 0:
                    plot_data = points_lh
                else:
                    plot_data = np.concatenate((plot_data, points_lh), axis=0)

            if right_hand_3d_position is not None:
                if hand_type == 'mano':
                    hand_links = self.hand_mano_links
                elif hand_type == 'coco':
                    hand_links = self.hand_coco_links
                else:
                    raise ValueError("Hand type is not supported")
                points_rh = right_hand_3d_position[f]
                x_rh = points_rh[:, 0]
                y_rh = points_rh[:, 1]
                z_rh = points_rh[:, 2]
                # ax.scatter(x_rh, y_rh, z_rh, c='r', s=5)
                ax.scatter(x_rh, y_rh, z_rh, c='b', s=5, zorder=3)

                if plot_hand_indices:
                    indices = np.arange(len(x_rh))
                    for i in range(len(x_rh)):
                        ax.text(x_rh[i], y_rh[i], z_rh[i], str(indices[i]), color='black')

                # set hand colors
                hand_line_colors = ['b'] * 20
                # thumb in red
                # hand_line_colors[12:15] = ['r'] * 3
                # hand_line_colors[19] = 'r'
                # index finger in yellow
                # hand_line_colors[0:3] = ['y'] * 3
                # hand_line_colors[15] = 'y'

                lines_rh = [[points_rh[link[0]], points_rh[link[1]]] for link in hand_links]
                line_collection_rh = Line3DCollection(lines_rh, edgecolors=hand_line_colors, linewidths=2, zorder=3)
                ax.add_collection(line_collection_rh)

                if len(plot_data) == 0:
                    plot_data = points_rh
                else:
                    plot_data = np.concatenate((plot_data, points_rh), axis=0)

            if body_3d_position is not None:
                if body_type == 'smpl':
                    body_links = self.body_smpl_links
                elif body_type == 'coco':
                    body_links = self.body_coco_links
                else:
                    raise ValueError('Body type not supported')
                points_body = body_3d_position[f]
                x_body = points_body[:, 0]
                y_body = points_body[:, 1]
                z_body = points_body[:, 2]
                ax.scatter(x_body, y_body, z_body, c='b', s=4, zorder=0)

                body_line_colors = ['b'] * 20
                lines_body = [[points_body[link[0]], points_body[link[1]]] for link in body_links]
                line_collection_body = Line3DCollection(lines_body, edgecolors=body_line_colors, linewidths=2, zorder=0)
                ax.add_collection(line_collection_body)

                if len(plot_data) == 0:
                    plot_data = points_body
                else:
                    plot_data = np.concatenate((plot_data, points_body), axis=0)

            if cp_3d_position is not None:
                points_cp = cp_3d_position[f]
                x_cp = points_cp[:, 0]
                y_cp = points_cp[:, 1]
                z_cp = points_cp[:, 2]
                ax.scatter(x_cp, y_cp, z_cp, c='purple', s=5, zorder=5)

            if highlight:
                assert string_audio_position is not None, 'string_audio_position is required for highlighting string!'
                points_string_audio = string_audio_position[f]

                lines_string_audio = [[points_string_audio[0], points_string_audio[1]]]
                line_collection_string_audio = Line3DCollection(lines_string_audio, edgecolors='green', linewidths=2,
                                                                zorder=2)
                ax.add_collection(line_collection_string_audio)

                wrong_on_same_string = False
                if string_lh_position is not None and string_rh_position is not None:
                    points_string_lh = string_lh_position[f]
                    points_string_rh = string_rh_position[f]
                    if not np.array_equal(points_string_audio, points_string_lh):
                        if np.array_equal(points_string_lh, points_string_rh):
                            wrong_on_same_string = True
                            lines_string_lh = [[points_string_lh[0], points_string_lh[1]]]
                            line_collection_string_lh = Line3DCollection(lines_string_lh, edgecolors='red',
                                                                         linewidths=2,
                                                                         zorder=2)
                            ax.add_collection(line_collection_string_lh)

                if string_lh_position is not None and not wrong_on_same_string:
                    points_string_lh = string_lh_position[f]

                    if not np.array_equal(points_string_audio, points_string_lh):
                        lines_string_lh = [[points_string_lh[0], points_string_lh[1]]]
                        line_collection_string_lh = Line3DCollection(lines_string_lh, edgecolors='yellow', linewidths=2,
                                                                     zorder=2)
                        ax.add_collection(line_collection_string_lh)

                if string_rh_position is not None and not wrong_on_same_string:
                    points_string_rh = string_rh_position[f]

                    if not np.array_equal(points_string_audio, points_string_rh):
                        lines_string_rh = [[points_string_rh[0], points_string_rh[1]]]
                        line_collection_string_rh = Line3DCollection(lines_string_rh, edgecolors='orange', linewidths=2,
                                                                     zorder=2)
                        ax.add_collection(line_collection_string_rh)

            # set axis lim
            ax.view_init(elev=-10, azim=110, roll=-60)  # 正面
            # ax.view_init(elev=-20, azim=10, roll=10)    # 侧面
            # ax.set_xlim(0, 1000)
            # ax.set_ylim(0, 800)
            # ax.set_zlim(0, 1200)

            if view == 'finger':
                if left_hand_3d_position is None:
                    raise Exception('Hand is required for finger view!')
                ax.view_init(elev=-10, azim=110, roll=-60)  # 正面
                if f == 0:
                    xlim, ylim, zlim = self._compute_axis_lim(points_lh, scale_factor=0.8)

            elif view == 'instrument':
                ax.view_init(elev=-10, azim=110, roll=-60)  # 正面
                if f == 0:
                    xlim, ylim, zlim = self._compute_axis_lim(points_instru[self.fingerboard_idx], scale_factor=1.2)

            elif view == 'instrument_bottom':
                ax.view_init(elev=-57, azim=-151, roll=-100)
                if f == 0:
                    xlim, ylim, zlim = self._compute_axis_lim(points_instru[self.fingerboard_idx], scale_factor=2.5)

            elif view == 'instrument_lateral':
                ax.view_init(elev=174, azim=-164, roll=-150)
                if f == 0:
                    xlim, ylim, zlim = self._compute_axis_lim(points_instru[self.fingerboard_idx], scale_factor=1.5)

            elif view == 'holistic':
                ax.view_init(elev=8, azim=115, roll=-50)
                # ax.view_init(elev=125, azim=-137, roll=-50)
                if f == 0:
                    xlim, ylim, zlim = self._compute_axis_lim(plot_data, scale_factor=1)
            else:
                raise Exception('View must be either "finger", "instrument" or "holistic"!')
                # origin is set to [0, 0, 0]
                # xlim = (0, xlim[1])
                # ylim = (0, ylim[1])
                # zlim = (0, zlim[1])

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame Num: {f + 1}')

            ax.set_box_aspect([2, 2, 2])

            # save current frame
            plt.draw()
            if plot_static_frame:
                plt.show()
            else:
                # Uncomment it when unknown shape bug occurs
                import matplotlib
                matplotlib.use("Agg")
            plt.pause(0.01)

            plt.switch_backend('agg')

            # convert Matplotlib to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, frame_size)

            # write in the video
            video_writer.write(img)
            print(f'frame {f} done.')

        video_writer.release()
        plt.close()


    def visualize_3dkp_o3d(self, output_path, shot_path, audioname=None, lefthand_3d_position=None,
                           righthand_3d_position=None, instrument_3d_position=None, bow_3d_position=None,
                           body_3d_position=None, cp_3d_position=None, plot_static_frame=False, view_file=None,
                           view_type='front', lookat_point=None, offset=0, if_vis=None,
                           bodycolor="purple", get_obj=False, get_OffScreenRender=False):
        print("Visualizing ...")

        import open3d as o3d
        import open3d.visualization.rendering as rendering

        if if_vis is None:
            if_vis = [False, True, True, True, True, True]

        if view_file is not None:
            view_info = json.load(open(view_file, 'r'))
            intrinsic = view_info['intrinsic']
            H, W = intrinsic['height'], intrinsic['width']
        else:
            H, W = 1920, 1080

        # render2image
        render = rendering.OffscreenRenderer(W, H)
        c_lh = rendering.MaterialRecord()
        c_rh = rendering.MaterialRecord()
        c_ins = rendering.MaterialRecord()
        c_bow = rendering.MaterialRecord()
        c_lhj = rendering.MaterialRecord()
        c_rhj = rendering.MaterialRecord()
        c_body = rendering.MaterialRecord()
        c_bodyj = rendering.MaterialRecord()
        c_cp = rendering.MaterialRecord()
        skin = [249 / 255, 174 / 255, 120 / 255, 1.0]
        c_lh.base_color = skin
        c_rh.base_color = skin
        c_ins.base_color = [120 / 255, 110 / 255, 107 / 255, 1.0]
        c_bow.base_color = [110 / 255,  40 / 255,   2 / 255, 1.0]
        c_lhj.base_color = skin
        c_rhj.base_color = skin
        if bodycolor == "purple":
            bodyr, bodyg, bodyb = 90, 31, 117
        elif bodycolor == "blue":
            bodyr, bodyg, bodyb = 61, 94, 191
        elif bodycolor == "pink":
            bodyr, bodyg, bodyb = 148, 55, 109
        elif bodycolor == "orange":
            bodyr, bodyg, bodyb = 179, 58, 34
        else:
            print("NOT VALIDATE COLOR! SET PURPLE AS DEFAULT!")
            bodyr, bodyg, bodyb = 61, 94, 191

        c_body.base_color = [bodyr/255, bodyg/255, bodyb/255, 1.0]
        c_bodyj.base_color = [(bodyr-5)/255, (bodyg-5)/255, (bodyb-5)/255, 1.0]

        c_cp.base_color = self.color_cp + [1.0]

        c_lh.shader = "defaultLit"
        c_rh.shader = "defaultLit"
        c_ins.shader = "defaultLit"
        c_bow.shader = "defaultLit"
        c_lhj.shader = "defaultLit"
        c_rhj.shader = "defaultLit"
        c_body.shader = "defaultLit"
        c_bodyj.shader = "defaultLit"
        c_cp.shader = "defaultLit"

        # flat bridge example
        # instrument_3d_position = self.get_second_third_string(instrument_3d_position[:,:7])

        # 创建视图
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=W, height=H, visible=False)
        # 绘制坐标系
        coordination = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # 创建地板
        # ground = o3d.geometry.TriangleMesh.create_box(width=4.0, height=4.0, depth=0.001)
        # ground.paint_uniform_color([0.8, 0.8, 0.8])  # 地板颜色
        # vis.add_geometry(coordination)
        # vis.add_geometry(ground)

        # 添加初始帧
        skeleton_instrument_list = []  # 1. 琴
        skeleton_lefthand_list = []  # 2. 左手
        skeleton_righthand_list = []  # 3. 右手
        skeleton_bow_list = []  # 4. 弓
        skeleton_body_list = []  # 5. 人体

        # 创建视频writer
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        # video_render = cv2.VideoWriter(output_render, fourcc, fps, (W*2, H*2))

        # 逐帧更新
        for frame_idx in tqdm(range(len(lefthand_3d_position))):
            render.scene.clear_geometry()
            mesh = o3d.geometry.TriangleMesh()

            # set background color
            # render_bg = vis.get_render_option()
            # render_bg.background_color = np.array([187 / 255, 189 / 255, 188 / 255])

            # set 地板
            if frame_idx == 0 and if_vis[0]:
                self._set_ground(vis, instrument_3d_position[0])
                # self._set_square_ground(vis, instrument_3d_position[0])
                # render_option = vis.get_render_option()
                # render_option.background_color = [0.4,0.4,0.4]

            if frame_idx % 60 == 0 and get_OffScreenRender:
                OffScreenTick = True
            else:
                OffScreenTick = False


            # 更新instrument位置
            if if_vis[1]:
                skeleton_instrument_list, whole_mesh = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_instrument_list,
                                                                  kp3d_cur_frame=instrument_3d_position[frame_idx],
                                                                  # kp3d_cur_frame=instrument_3d_position[frame_idx, 1:],
                                                                  links=self.instrument_links, color=self.color_instrument,
                                                                  if_OffScreenRender=OffScreenTick, renderer=render, rcolor_joint=c_ins, rcolor_limb=c_ins, attribute="ins",
                                                                  cylinder_radius=0.0023, sphere_radius=0.0023, color_node=self.color_instrument
                                                                  )
                mesh += whole_mesh

                # 更新cp位置
                if cp_3d_position is not None:
                    # ic(instrument_3d_position[frame_idx])
                    # ic(cp_3d_position[frame_idx])
                    # ic(frame_idx)
                    # self._vis_cp(vis_obj=vis, cp_location=cp_3d_position[frame_idx], color=self.color_cp, size=0.006)
                    if not np.isnan(cp_3d_position[frame_idx]).all():

                        skeleton_cp_list, whole_mesh = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_cp_list,
                                                                              kp3d_cur_frame=cp_3d_position[frame_idx],
                                                                              links=[], color=self.color_cp,
                                                                              if_OffScreenRender=OffScreenTick, renderer=render, rcolor_joint=c_cp,
                                                                              rcolor_limb=c_cp, attribute="cp",
                                                                              cylinder_radius=0.0023, sphere_radius=0.006,
                                                                              color_node=self.color_cp
                                                                              )

            # 更新左手位置
            if if_vis[2]:
                skeleton_lefthand_list, whole_mesh = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_lefthand_list,
                                                                kp3d_cur_frame=lefthand_3d_position[frame_idx],
                                                                if_OffScreenRender=OffScreenTick, renderer=render, rcolor_joint=c_lhj, rcolor_limb=c_lh, attribute="lh",
                                                                links=self.hand_links, color=self.color_lefthand, cylinder_radius=0.0045,
                                                                sphere_radius=0.0045, color_node=self.color_lefthand_joint)
                mesh += whole_mesh
            # 更新右手位置
            if if_vis[3]:
                skeleton_righthand_list, whole_mesh = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_righthand_list,
                                                                 kp3d_cur_frame=righthand_3d_position[frame_idx],
                                                                 if_OffScreenRender=OffScreenTick, renderer=render, rcolor_joint=c_rhj, rcolor_limb=c_rh, attribute="rh",
                                                                 links=self.hand_links, color=self.color_righthand, cylinder_radius=0.0045,
                                                                 sphere_radius=0.0045, color_node=self.color_righthand_joint)
                mesh += whole_mesh
            # 更新bow位置
            if if_vis[4]:
                skeleton_bow_list, whole_mesh = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_bow_list,
                                                           kp3d_cur_frame=bow_3d_position[frame_idx], links=self.bow_links,
                                                           if_OffScreenRender=OffScreenTick, renderer=render, rcolor_joint=c_bow, rcolor_limb=c_bow, attribute="bow",
                                                           cylinder_radius=0.0025, color=self.color_bow,
                                                           sphere_radius=0.0025, color_node=self.color_bow)
                mesh += whole_mesh
            # 更新body位置
            if if_vis[5]:
                skeleton_body_list, whole_mesh = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_body_list,
                                                            # kp3d_cur_frame=body_3d_position[frame_idx, 16:],      # no body (except arms)
                                                            kp3d_cur_frame=body_3d_position[frame_idx], links=self.body_links,
                                                            if_OffScreenRender=OffScreenTick, renderer=render, rcolor_joint=c_bodyj, rcolor_limb=c_body, attribute="body",
                                                            cylinder_radius=0.018, color=self.color_body,
                                                            sphere_radius=0.018, color_node=self.color_body_joint)
                mesh += whole_mesh

            # 生成obj
            if get_obj:
                temp_dir = r"../validation/TEMP_OBJ/TEMP/"
                os.makedirs(temp_dir, exist_ok=True)
                file = os.path.join(temp_dir, f'{frame_idx}.obj')
                # file = os.path.join(temp_ply_dir, f"{frame_idx}.ply")
                if (frame_idx-10) % 30 == 0 or frame_idx == 0:
                    print(f"Frame {frame_idx} DONE")
                    o3d.io.write_triangle_mesh(file, mesh)



            if view_type == "front":
                # set 正面视角
                self._set_viewpoint_front(vis, instrument_3d_position[0])
            elif view_type == "lateral":
                # set 侧面视角
                self._set_viewpoint_lateral(vis, instrument_3d_position[0])
            elif view_file != None:
                if os.path.isfile(view_file):
                    # set 特定视角
                    ctr = vis.get_view_control()
                    param = o3d.io.read_pinhole_camera_parameters(view_file)
                    extrinsic_matrix = param.extrinsic
                    camera_position = extrinsic_matrix[:3, 3]
                    camera_direction = extrinsic_matrix[:3, 2]
                    lookat_position = camera_position - camera_direction
                    up_vector = extrinsic_matrix[:3, 1]

                    focal_length = param.intrinsic.intrinsic_matrix[0, 0]
                    image_width = param.intrinsic.width
                    fov_x = 2 * np.arctan(image_width / (2 * focal_length)) * 180 / np.pi

                    render.scene.set_background([10.0, 10.0, 10.0, 10.0])
                    render.setup_camera(param.intrinsic, param.extrinsic)
                    # render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
                    render.scene.scene.set_sun_light([-0.3, 0.3, -0.3], [1.0, 1.0, 1.0], 150000)
                    render.scene.scene.enable_sun_light(True)
                    # rendered_image = render.render_to_image()
                    # print("Saving image at testasdasdasd.png")
                    # o3d.io.write_image("testasdasdasd.png", rendered_image, 9)

                    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
                    # ctr.set_lookat(lookat_point)  # 目标点
                    # ctr.set_zoom(0.8)
                else:
                    print("No view file founded.")

            vis.poll_events()
            # vis.update_renderer()
            # render_bg = vis.get_render_option()
            # render_bg.background_color = np.array([199 / 255, 192 / 255, 191 / 255])
            # vis.run()
            while plot_static_frame and frame_idx == 0:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)  # 防止 CPU 占用过高

            image = vis.capture_screen_float_buffer(do_render=False)
            image = (255 * np.asarray(image)).astype(np.uint8)
            image = cv2.resize(image, (W, H))
            frame_text = f"Frame: {frame_idx + offset}"
            # cv2.putText(image, frame_text, (W-150, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.7*255, 0.7*255, 0.7*255), 1, cv2.LINE_AA)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以符合OpenCV的要求
            video_writer.write(image_bgr)

            # rendered_image = render.render_to_image()
            # rendered_image = np.asarray(rendered_image)
            # rendered_image = cv2.resize(rendered_image, (W, H))
            # rendered_bgr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
            # video_render.write(rendered_bgr)
            if frame_idx % 60 == 0:
                selected_shot = os.path.join(shot_path, f'{audioname}_{frame_idx+offset}.png')
                vis.capture_screen_image(selected_shot, do_render=False)
                # cv2.imwrite(selected_shot, image_bgr[380:1400, 100:])
                # cv2.imwrite(selected_shot, image_bgr[200:1770, 70:1200])
                # cv2.imwrite(selected_shot, image_bgr[235:1725, 153:1225])

                if OffScreenTick:
                    rendered_image = render.render_to_image()
                    rendered_shot = os.path.join(shot_path, f'rendered_{audioname}_{frame_idx + offset}.png')
                    o3d.io.write_image(rendered_shot, rendered_image, 9)
                # ic(image_bgr.shape)

            # print(skeleton_body_list)
            # temp = self.convert_to_assimp_mesh(skeleton_body_list)
            # assimp_meshes.append(self.convert_to_assimp_mesh(mesh))

        vis.destroy_window()
        video_writer.release()
        # video_render.release()
        # pyassimp.export_mesh(assimp_meshes, "o3d2fbx_test.fbx")

    def visualize_3dkp_o3d_view_select(self, lefthand_3d_position=None, righthand_3d_position=None,
                                       instrument_3d_position=None, bow_3d_position=None, body_3d_position=None,
                                       cp_3d_position=None, view_type='manual_view', if_vis=None,
                                       manual_view_file='manual_view.json',  selected_frame=0):
        print("Visualizing ...")

        import open3d as o3d
        import open3d.visualization.rendering as rendering

        if if_vis is None:
            if_vis = [False, True, True, True, True, True]

        # instrument_3d_position = self._get_23string(instrument_3d_position)

        # 创建视图
        vis = o3d.visualization.Visualizer()
        H, W = 1920, 1440
        # H, W = 1440, 1440   # performer
        # H, W = 1920, 1280
        # H, W = 1440, 1702

        vis.create_window(width=W, height=H)
        # 绘制坐标系
        coordination = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # 创建地板
        # ground = o3d.geometry.TriangleMesh.create_box(width=4.0, height=4.0, depth=0.001)
        # ground.paint_uniform_color([0.8, 0.8, 0.8])  # 地板颜色
        vis.add_geometry(coordination)
        # vis.add_geometry(ground)

        if view_type == "manual_view":
            skeleton_instrument_list = []  # 1. 琴
            skeleton_lefthand_list = []  # 2. 左手
            skeleton_righthand_list = []  # 3. 右手
            skeleton_bow_list = []  # 4. 弓
            skeleton_body_list = []  # 5. 人体

            if if_vis[0]:
                self._set_ground(vis, instrument_3d_position[selected_frame])
            if if_vis[1]:
                skeleton_instrument_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_instrument_list,
                                                                  kp3d_cur_frame=instrument_3d_position[selected_frame],
                                                                  links=self.instrument_links, color=self.color_instrument,
                                                                  cylinder_radius=0.0023, sphere_radius=0.0023,
                                                                  color_node=self.color_instrument)
                if cp_3d_position is not None:
                    if not np.isnan(cp_3d_position[selected_frame]).all():
                        self._vis_cp(vis_obj=vis, cp_location=cp_3d_position[selected_frame], color=self.color_cp, size=0.006)

            # 更新左手位置
            if if_vis[2]:
                skeleton_lefthand_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_lefthand_list,
                                                                kp3d_cur_frame=lefthand_3d_position[selected_frame],
                                                                links=self.hand_links, color=self.color_lefthand,
                                                                cylinder_radius=0.0055,
                                                                sphere_radius=0.0055, color_node=self.color_lefthand_joint)
            # 更新右手位置
            if if_vis[3]:
                skeleton_righthand_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_righthand_list,
                                                                 kp3d_cur_frame=righthand_3d_position[selected_frame],
                                                                 links=self.hand_links, color=self.color_righthand,
                                                                 cylinder_radius=0.0055,
                                                                 sphere_radius=0.0055,
                                                                 color_node=self.color_righthand_joint)
            # 更新bow位置
            if if_vis[4]:
                skeleton_bow_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_bow_list,
                                                           kp3d_cur_frame=bow_3d_position[selected_frame], links=self.bow_links,
                                                           cylinder_radius=0.0025, color=self.color_bow,
                                                           sphere_radius=0.0025, color_node=self.color_bow)
            # # 更新body位置
            if if_vis[5]:
                skeleton_body_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_body_list,
                                                            kp3d_cur_frame=body_3d_position[selected_frame],
                                                            links=self.body_links, color=self.color_body,
                                                            cylinder_radius=0.018, sphere_radius=0.018,
                                                            color_node=self.color_body_joint)

            ctr = vis.get_view_control()
            # ctr.set_lookat(lookat_point)
            # ctr.set_lookat([0,0,0.5])
            # ctr.set_front([0,1,0])
            # # ctr.set_eye([0,-0.7,0.7])
            # ctr.set_up([0, 1, 0])
            # render_bg = vis.get_render_option()
            # render_bg.background_color = np.array([199/255, 192/255, 191/255])
            vis.run()  # 改变视角以确定要存储的视角, press ‘q’ 退出
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(manual_view_file, param)
            vis.destroy_window()

        base_name, ext = os.path.splitext(manual_view_file)
        backup_manual_view = f"{base_name}_backup{ext}"
        if not os.path.exists(backup_manual_view):
            shutil.copy(manual_view_file, backup_manual_view)
            print(f"Backup created: {backup_manual_view}")
        else:
            print(f"Backup already exists: {backup_manual_view}")

        # vis.create_window(width=W, height=H)
        # # 添加初始帧
        # skeleton_instrument_list = []  # 1. 琴
        # skeleton_lefthand_list = []  # 2. 左手
        # skeleton_righthand_list = []  # 3. 右手
        # skeleton_bow_list = []  # 4. 弓
        # skeleton_body_list = []  # 5. 人体
        #
        # # 创建视频writer
        # fps = 30
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        #
        # # set 地板
        # self._set_ground(vis, instrument_3d_position[0])
        #
        # # 更新instrument位置
        # skeleton_instrument_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_instrument_list,
        #                                                   kp3d_cur_frame=instrument_3d_position[selected_frame],
        #                                                   links=self.instrument_links, color=self.color_instrument,
        #                                                   cylinder_radius=0.0023, sphere_radius=0.0023,
        #                                                   color_node=self.color_instrument)
        # # 更新左手位置
        # skeleton_lefthand_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_lefthand_list,
        #                                                 kp3d_cur_frame=lefthand_3d_position[selected_frame],
        #                                                 links=self.hand_links, color=self.color_lefthand,
        #                                                 cylinder_radius=0.0055,
        #                                                 sphere_radius=0.0055, color_node=self.color_lefthand_joint)
        # # 更新右手位置
        # skeleton_righthand_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_righthand_list,
        #                                                  kp3d_cur_frame=righthand_3d_position[selected_frame],
        #                                                  links=self.hand_links, color=self.color_righthand,
        #                                                  cylinder_radius=0.0055,
        #                                                  sphere_radius=0.0055,
        #                                                  color_node=self.color_righthand_joint)
        # # 更新bow位置
        # skeleton_bow_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_bow_list,
        #                                            kp3d_cur_frame=bow_3d_position[selected_frame], links=self.bow_links,
        #                                            cylinder_radius=0.0045, color=self.color_bow,
        #                                            sphere_radius=0.0045, color_node=self.color_bow)
        # # 更新body位置
        # skeleton_body_list = self._vis_frame_update(vis_obj=vis, skeleton_list=skeleton_body_list,
        #                                             kp3d_cur_frame=body_3d_position[selected_frame],
        #                                             links=self.body_links, color=self.color_body,
        #                                             cylinder_radius=0.018, sphere_radius=0.018,
        #                                             color_node=self.color_body_joint)
        # # 更新cp位置
        # ic(cp_3d_position)
        # if cp_3d_position is not None:
        #     ic(selected_frame)
        #     self._vis_cp(vis_obj=vis, cp_location=cp_3d_position[selected_frame], color=self.color_cp, size=10)
        #
        # if view_type == "front":
        #     # set 正面视角
        #     self._set_viewpoint_front(vis, instrument_3d_position[0])
        #     ctr = vis.get_view_control()
        #     ctr.set_lookat(lookat_point)
        # elif view_type == "lateral":
        #     # set 侧面视角
        #     self._set_viewpoint_lateral(vis, instrument_3d_position[0])
        #     ctr = vis.get_view_control()
        #     ctr.set_lookat(lookat_point)
        # elif view_type == "manual_view":
        #     ctr = vis.get_view_control()
        #     param = o3d.io.read_pinhole_camera_parameters(manual_view_file)
        #     ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        #     # ctr.set_lookat(lookat_point)
        # elif view_file != None:
        #     if os.path.isfile(view_file):
        #         # set 特定视角
        #         ctr = vis.get_view_control()
        #         param = o3d.io.read_pinhole_camera_parameters(view_file)
        #         ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        #         # ctr.set_lookat(lookat_point)  # 目标点
        #     else:
        #         print("No view file founded.")
        #
        # vis.poll_events()
        # vis.update_renderer()
        #
        # image = vis.capture_screen_float_buffer()
        # image = (255 * np.asarray(image)).astype(np.uint8)
        # image = cv2.resize(image, (W, H))
        # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以符合OpenCV的要求
        # video_writer.write(image_bgr)
        #
        # vis.capture_screen_image("test_image_3.png")
        #
        # vis.destroy_window()



    @staticmethod
    def rotation_6d_to_R(d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2, axis=-1)
        return np.stack((b1, b2, b3), axis=-2)

    @staticmethod
    def apply_rotations(target_points, rotation_matrices):
        """
        Applies n rotation matrices to n sets of key points.

        Parameters:
        target_points (numpy array): Shape (n, m, 3), representing the positions of m key points for n hands.
        rotation_matrices (numpy array): Shape (n, 3, 3), representing n rotation matrices.

        Returns:
        numpy array: Shape (n, m, 3), representing the rotated hand key points.
        """
        n = target_points.shape[0]

        # Initialize an array to hold the rotated points
        rotated_points = np.zeros_like(target_points)

        # Apply the corresponding rotation matrix to each set of points
        for i in range(n):
            # (n, 3) is row vectors, not column vectors
            rotated_points[i] = np.dot(target_points[i], rotation_matrices[i].T)

        return rotated_points

    @staticmethod
    def smooth_rot(rot_matrix, njoints=16, window_length=11, polyorder=6):
        # rot6d = rot6d.reshape(-1, 6)
        # rot_matrix = self._rotation_6d_to_R(rot6d).reshape(-1, 3, 3)
        rot_matrix = rot_matrix.reshape(-1, 3, 3)
        rotation = Rotation.from_matrix(rot_matrix)
        rot_quat = rotation.as_quat().reshape(-1, njoints, 4)  # [nframes, njoints, 4]
        smoothed_quat = np.empty_like(rot_quat)
        nframes, njoints, nquats = rot_quat.shape

        ############################## START: Smooth with Savgol Filter ##############################

        # q and -q represents the same rotation, the interp between the positive and negative could cause errors
        dot_products = np.einsum('ijk,ijk->ij', rot_quat[1:], rot_quat[:-1])
        flip_mask = (np.cumsum(dot_products < 0, axis=0) % 2).astype(bool)
        rot_quat[1:][flip_mask] *= -1
        smoothed_quat = signal.savgol_filter(rot_quat, window_length=window_length, polyorder=polyorder, axis=0,
                                            mode='interp')

        norms = np.linalg.norm(smoothed_quat, axis=-1)
        smoothed_quat = smoothed_quat / norms[:, :, np.newaxis]  # normalization

        ############################## END: Smooth with Savgol Filter ##############################

        ############################## START: Smooth with B Spline ##############################

        # time_points = np.linspace(0, 1, nframes)
        # s = 0.08  # smooth factor
        #
        # for joint in range(njoints):
        #     for quat in range(nquats):
        #         spline = splrep(time_points, rot_quat[:, joint, quat], s=s, k=3)
        #         smoothed_quat[:, joint, quat] = splev(time_points, spline)
        #
        # norms = np.linalg.norm(smoothed_quat, axis=-1)
        # smoothed_quat = smoothed_quat / norms[:, :, np.newaxis]

        ############################## END: Smooth with B Spline ##############################

        ############################## START: Smooth with Slerp ##############################

        # times = np.arange(nframes)
        # smoothing_factor = 0.65
        #
        # for joint in range(njoints):
        #     # Extract the rotations for the current joint
        #     joint_rotations = rotation[joint::njoints]  # 获取每个关节的旋转
        #
        #     # Create Slerp for the current joint
        #     slerp = Slerp(times, joint_rotations)
        #
        #     # Generate smoothed rotations with Slerp
        #     smoothed_rotation = slerp(times * smoothing_factor)
        #
        #     # Store the results
        #     smoothed_quat[:, joint] = smoothed_rotation.as_quat()

        ############################## END: Smooth with Slerp ##############################

        smoothed_quat = smoothed_quat.reshape(-1, 4)
        smooothed_rot_matrices = Rotation.from_quat(smoothed_quat).as_matrix().reshape(-1, njoints, 3, 3)
        return smooothed_rot_matrices

    @staticmethod
    def get_second_third_string(instru_7):
        """
        :param instru_7: shape in (seq_len, 7, 3)
        :return: shape in (seq_len, 11, 3), with the end points of the 2nd and 3rd string
        """
        batch_size = instru_7.shape[0]
        instru_11 = np.zeros((batch_size, 11, 3))
        instru_11[:, :7, :] = instru_7

        instru_11[:, 7, :] = instru_7[:, 1, :] * (2 / 3.) + instru_7[:, 2, :] * (1 / 3.)
        instru_11[:, 8, :] = instru_7[:, 1, :] * (1 / 3.) + instru_7[:, 2, :] * (2 / 3.)
        instru_11[:, 9, :] = instru_7[:, 3, :] * (2 / 3.) + instru_7[:, 4, :] * (1 / 3.)
        instru_11[:, 10, :] = instru_7[:, 3, :] * (1 / 3.) + instru_7[:, 4, :] * (2 / 3.)

        return instru_11

    @staticmethod
    def calculate_cp_coordinates(cp_info, instru_11):
        """
        get cp coordinates from the cp info
        :param cp_info: (batch_size, 2)  [activation_string, vibrating_length]
        :param instru_11: (batch_size, 11, 3)
        :return: cp coordinates (batch_size, 3)
        """
        cp_coordinates = np.zeros((cp_info.shape[0], 3))

        # match the 3dkp's index of each string
        string_points = {
            0: (2, 4),  # 1弦
            1: (8, 10),  # 2弦
            2: (7, 9),  # 3弦
            3: (1, 3)  # 4弦
        }

        # get cp coordinate
        for i, (string_num, position_ratio) in enumerate(cp_info):
            # -1 for no string activating for current frame
            if string_num == -1:
                cp_coordinates[i] = np.array([np.nan, np.nan, np.nan])
                # cp_coordinates[i] = np.array([0, 0, 0])
                continue
            start_idx, end_idx = string_points[string_num]
            start_point = instru_11[i, start_idx]
            end_point = instru_11[i, end_idx]
            # position_ratio *= 1.03
            if position_ratio == -1.:
                cp_coordinates[i] = np.array([0, 0, 0])
                continue
            cp_coordinates[i] = position_ratio * start_point + (1 - position_ratio) * end_point

        return cp_coordinates

    @staticmethod
    def smooth_loc(loc_3d, sf=35):
        for i in range(loc_3d.shape[1]):
            loc_3d[:, i] = signal.savgol_filter(loc_3d[:, i], window_length=sf, polyorder=2, axis=0, mode='interp')
        return loc_3d

    def _rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def _compute_axis_lim(self, points, scale_factor=1):
        # ic(triangulated_points.shape)
        # triangulated_points in shape [num_frame, num_keypoint, 3 axis]
        xlim, ylim, zlim = None, None, None
        # ic(triangulated_points.shape)
        minmax = np.nanpercentile(points, q=[0, 100], axis=0).T
        # ic(minmax)
        minmax *= 1.
        minmax_range = (minmax[:, 1] - minmax[:, 0]).max() / 2
        if xlim is None:
            mid_x = np.mean(minmax[0])
            xlim = mid_x - minmax_range / scale_factor, mid_x + minmax_range / scale_factor
        if ylim is None:
            mid_y = np.mean(minmax[1])
            ylim = mid_y - minmax_range / scale_factor, mid_y + minmax_range / scale_factor
        if zlim is None:
            mid_z = np.mean(minmax[2])
            zlim = mid_z - minmax_range / scale_factor, mid_z + minmax_range / scale_factor
        return xlim, ylim, zlim

    def _find_npy_files(self, npy_dir):
        npy_files = glob.glob(os.path.join(npy_dir, '**', '*.npy'), recursive=True)
        return npy_files

    def _normalize_keypoints_cello(self, data):
        data = np.array(data)
        frame_num, num_keypoints, coords = data.shape
        # make sure 155 in dim2, 3 in dim3
        assert num_keypoints == 155 and coords == 3
        # end_pin_index = 139, LH_WIRST_INDEX = 91
        end_pin_location = data[:, self.end_pin_index, :]
        normalized_data = data - end_pin_location[:, np.newaxis, :]
        return normalized_data, normalized_data[:, self.lh_wrist_index:self.lh_wrist_index + 1, :]
        # LH_WIRST_INDEX:LH_WIRST_INDEX+1 是为了保持维度不变，相当于mp.mewaxis

    def _get_init_pose_batch(self, hand_trans, hand_type='left', mano=False):
        if mano:
            filepath = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset/SPD/mano_hand_info/J3_{hand_type}.txt"
        else:
            filepath = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset/SPD/custom_hand_info/J3_{hand_type}.txt"
        with open(filepath) as f:
            lines = f.readlines()
        init_pose = np.zeros((21, 3))
        for l in range(len(lines)):
            init_pose[l] = np.array([float(x) for x in lines[l].rstrip().split(' ')])
        init_pose = init_pose - init_pose[0]
        init_pose_batch = np.array([init_pose + trans for trans in hand_trans])
        return init_pose_batch

    def _get_joint_positions_batch(self, init_positions, rotations, parent_indices, bone_lengths=None):
        """
        - init_positions: shape (batch_size, 21, 3), the initial joint positions for each frame.
        - rotations:  shape (batch_size, 16, 3, 3), the rotation matrices for the hand joints for each sample.
        - bone_lengths: numpy array of shape (20,)

        - positions: numpy array of shape (batch_size, 21, 3)
        """

        positions = np.zeros_like(init_positions)  # shape (batch_size, 21, 3)
        positions[:, 0, :] = init_positions[:, 0, :]  # The root joint (wrist) remains the same

        # Loop over each joint (1 to 20), and calculate its position relative to its parent joint
        for i in range(1, 21):
            parent_index = parent_indices[i]

            # We start from the parent and accumulate all rotation matrices down to the current joint
            R = rotations[:, parent_index]  # Get the parent's rotation matrix for the entire batch

            # Accumulate rotation matrices from the root to the parent
            current_parent_index = parent_index
            while current_parent_index != 0:
                current_parent_index = parent_indices[current_parent_index]
                R = np.einsum('bij,bjk->bik', rotations[:, current_parent_index], R)  # Batch matrix multiplication

            # Calculate the vector between the parent and the current joint in the initial pose
            original_parent_position = init_positions[:, parent_index, :]  # (batch_size, 3)
            original_self_position = init_positions[:, i, :]  # (batch_size, 3)

            if bone_lengths is None:
                original_vector = original_self_position - original_parent_position
            else:
                # Bone length for the current joint
                bone_length = bone_lengths[i - 1]
                # ATTENTION the way of normalizing! in terms of each row rather than the whole array
                original_vector = bone_length * self._normalize_vector_batch(
                    original_self_position - original_parent_position)

            # Apply the accumulated rotation to this vector
            relative_position = np.einsum('bij,bj->bi', R, original_vector)  # Batch matrix-vector multiplication
            # The new global position of the current joint is the parent's new position plus the relative displacement
            positions[:, i, :] = relative_position + positions[:, parent_index, :]

        return positions

    def _normalize_vector_batch(self, v_batch):
        """
        Input shape: (batch_size, 3), normalize each row
        """

        norms = np.linalg.norm(v_batch, axis=1, keepdims=True)
        # prevent from divided by 0
        norms[norms == 0] = 1
        # normalize each batch
        normalized_arr = v_batch / norms
        return normalized_arr


    def _vis_skeleton(self, joints, limbs, cylinder_radius=0.006, sphere_radius=0.003,
                     add_trans=None, color_vis=None, color_node=None,
                     if_OffScreenRender=False, renderer=None, rcolor_joint=None, rcolor_limb=None, attribute=None):
        """
        Input:
            joints: numpy [n_joints, 3], joint positions
            limbs: limb topology
            add_trans: numpy [3], additional translation for skeleton
            mask_scheme: occlusion mask scheme, 'lower'/'full'/'video'
            start/end: start/end frame for full-body occlusion mask if mask_scheme=='full'
            t: current timestep (for full-body occlusion visualization)
        Output:
            skeleton_list: open3d body skeleton (a list of open3d arrows)
        """
        import open3d as o3d

        skeleton_list = []
        whole_mesh = o3d.geometry.TriangleMesh()

        for idx, joint in enumerate(joints):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)  # radius is half the diameter
            if add_trans is not None:
                joint_position = joint + add_trans  # apply translation to joint position
            else:
                joint_position = joint  # no translation if add_trans is None
            sphere.translate(joint_position)  # translate the sphere to the joint position
            sphere.paint_uniform_color(color_node)  # paint the sphere with the specified color
            sphere.compute_vertex_normals()
            whole_mesh += sphere
            if if_OffScreenRender:
                sklt_name = f"{attribute}_joint_{idx}"
                renderer.scene.add_geometry(sklt_name, sphere, rcolor_joint)
            skeleton_list.append(sphere)  # add the sphere to the skeleton list

        for idx, limb in enumerate(limbs):
            bone_length = np.linalg.norm(joints[[limb[0]]] - joints[[limb[1]]], axis=-1)
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=cylinder_radius, cone_radius=0.001,
                                                           cylinder_height=bone_length, cone_height=0.001)
            mat = self._rotation_matrix_from_vectors(np.array([0, 0, 1]), joints[limb[1]] - joints[limb[0]])
            transformation = np.identity(4)
            transformation[0:3, 0:3] = mat
            transformation[:3, 3] = joints[limb[0]]
            if add_trans is not None:
                transformation[0:3, 3] += add_trans
            arrow.transform(transformation)
            arrow.paint_uniform_color(color_vis)
            arrow.compute_vertex_normals()
            whole_mesh += arrow
            if if_OffScreenRender:
                sklt_name = f"{attribute}_limb_{idx}"
                renderer.scene.add_geometry(sklt_name, arrow, rcolor_limb)
            skeleton_list.append(arrow)
        return skeleton_list, whole_mesh

    def _set_ground(self, vis_obj, instrument7_frame0):
        import open3d as o3d

        P6 = instrument7_frame0[5]      # tailgut
        P7 = instrument7_frame0[6]      # endpin

        # 向量 tailgut - endpin
        v67 = P7 - P6
        v67_norm = np.linalg.norm(v67)
        v67_unit = v67 / v67_norm

        # 创建圆柱体地面
        radius = 0.6
        height = 0.01
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=1000)
        cylinder.paint_uniform_color([0.9, 0.9, 0.9])

        # 将圆柱体的底面中心移动到endpin的位置
        cylinder.translate([-0.03,0.3,0])

        # 计算旋转轴：使得圆柱体的轴（默认是z轴）与向量tailgut-endpin
        # 默认的圆柱体轴向为z轴，即[0, 0, 1]
        default_axis = np.array([0, 0, 1])

        rotation_axis = np.cross(default_axis, v67_unit)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm != 0:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.dot(default_axis, v67_unit))

            # 创建旋转矩阵
            rotation = R.from_rotvec(rotation_axis * angle)  # 使用旋转轴和角度创建旋转矩阵
            rotation_matrix = rotation.as_matrix()

            # 旋转圆柱体，保持其中心位置不变
            # cylinder.rotate(rotation_matrix, center=P7)

        vis_obj.get_render_option().show_coordinate_frame = False
        vis_obj.add_geometry(cylinder)

    def _set_square_ground(self, vis_obj, instrument7_frame0):
        import open3d as o3d

        P6 = instrument7_frame0[5]      # tailgut
        P7 = instrument7_frame0[6]      # endpin

        # 向量 tailgut - endpin
        v67 = P7 - P6
        v67_norm = np.linalg.norm(v67)
        v67_unit = v67 / v67_norm

        # 创建圆柱体地面
        width = 2.0
        depth = 2.0
        height = 0.01
        ground = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        ground.paint_uniform_color([0.9, 0.9, 0.9])

        # 将圆柱体的底面中心移动到endpin的位置
        ground.translate([-0.03,0.3,0])

        # 计算旋转轴：使得圆柱体的轴（默认是z轴）与向量tailgut-endpin
        # 默认的圆柱体轴向为z轴，即[0, 0, 1]
        default_axis = np.array([0, 0, 1])

        rotation_axis = np.cross(default_axis, v67_unit)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm != 0:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.dot(default_axis, v67_unit))

            # 创建旋转矩阵
            rotation = R.from_rotvec(rotation_axis * angle)  # 使用旋转轴和角度创建旋转矩阵
            rotation_matrix = rotation.as_matrix()

            # 旋转圆柱体，保持其中心位置不变
            # cylinder.rotate(rotation_matrix, center=P7)

        vis_obj.get_render_option().show_coordinate_frame = False
        vis_obj.add_geometry(ground)

    def _vis_frame_update(self, vis_obj, skeleton_list, kp3d_cur_frame, links, color, color_node,
                          if_OffScreenRender=False, renderer=None, rcolor_joint=None, rcolor_limb=None, attribute=None,
                          cylinder_radius=0.006, sphere_radius=0.003, ):
            for arrow in skeleton_list:
                vis_obj.remove_geometry(arrow, reset_bounding_box=False)  # 清空前一帧
            skeleton_list, whole_mesh = self._vis_skeleton(joints=kp3d_cur_frame, limbs=links, color_vis=color, cylinder_radius=cylinder_radius, sphere_radius=sphere_radius, color_node=color_node,
                                                           if_OffScreenRender=if_OffScreenRender, renderer=renderer, rcolor_joint=rcolor_joint, rcolor_limb=rcolor_limb, attribute=attribute)
            for arrow in skeleton_list:
                vis_obj.add_geometry(arrow)
            return skeleton_list, whole_mesh

    def _set_viewpoint_front(self, vis_obj, instrument7_frame0):

        nut_l = instrument7_frame0[1]      # nut_l
        nut_r = instrument7_frame0[2]      # nut_r
        bridge_l = instrument7_frame0[3]      # bridge_l
        bridge_r = instrument7_frame0[4]      # bridge_r

        # 计算法向量
        v1 = nut_r - nut_l
        v2 = bridge_l - nut_l
        normal_vector = np.cross(v1, v2)  # 向量叉积得到法向量

        # 归一化法向量
        normal_unit = normal_vector / np.linalg.norm(normal_vector)

        # 计算第1和第2个点的距离
        distance = np.linalg.norm(bridge_l - nut_r)

        # 目标点（平面内的点，通常为第1个点）
        lookat_point = 0.5 * (bridge_l + bridge_r)

        # 计算观察点位置（根据右手法则，沿法向量方向偏移5倍距离）
        camera_position = lookat_point + 1 * distance * normal_unit

        # 设置"上方向"为标准的y轴向量
        up_vector = np.array([0, 1, 0])

        ctr = vis_obj.get_view_control()
        # 设置视角
        ctr.set_lookat(lookat_point)  # 目标点
        ctr.set_front(lookat_point - camera_position)  # 观察方向
        ctr.set_up(up_vector)  # "上方向"
        # ctr.set_zoom(0.5)

    def _set_viewpoint_lateral(self, vis_obj, instrument7_frame0):

        nut_l = instrument7_frame0[1]      # nut_l
        nut_r = instrument7_frame0[2]      # nut_r
        bridge_l = instrument7_frame0[3]      # bridge_l

        # 目标点（平面内的点，通常为第1个点）
        # lookat_point = 1 / 3. * nut_l + 2/3. * nut_r
        lookat_point = bridge_l

        # 设置观察方向为从第1个关键点指向第2个关键点的向量
        front_vector = nut_l - nut_r  # 观察方向

        # 计算从第3个点指向第1个点的方向，并旋转60度使其斜向上
        v3 = nut_l - bridge_l
        v3 = instrument7_frame0[4]- instrument7_frame0[5]
        angle = np.radians(8)  # 将60度转换为弧度

        # 绕Y轴旋转60度
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

        # 旋转v3向量
        up_vector = rotation_matrix.dot(v3)
        up_vector = up_vector / np.linalg.norm(up_vector)

        ctr = vis_obj.get_view_control()
        # 设置视角
        ctr.set_lookat(lookat_point)
        ctr.set_front(front_vector)
        ctr.set_up(up_vector)
        ctr.set_zoom(0.5)

    def _vis_cp(self, vis_obj, cp_location, color, size):
        import open3d as o3d

        cp_location = cp_location.reshape(3, )

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.paint_uniform_color(color)
        sphere.translate(cp_location)

        # 移除前一帧的cp
        if hasattr(self, 'sphere_geometry'):
            vis_obj.remove_geometry(self.sphere_geometry, reset_bounding_box=False)
        vis_obj.add_geometry(sphere)

        # 保存当前的cp对象，便于下一帧移除
        self.sphere_geometry = sphere
