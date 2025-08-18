from scipy.io import wavfile
from pitch_detect import pitch_detect_crepe
import freq_position as freq_position
from icecream import ic
from data_process.postprocess import get_original_motion, get_final_gen_motion
from data_process.data_alignment import *


class Validation():
    def __init__(self, generation_motion_file, original_motion_file, original_body_file, input_audio_file, vis=False):

        self.input_audio_file = input_audio_file

        # generated motion
        (self.generation_lh_3d, self.generation_rh_3d, self.generation_body_3d,
         self.generation_instrument_3d, self.generation_bow_3d, self.generation_R_s) = get_final_gen_motion(generation_motion_file)
        self.generation_num_frames = self.generation_lh_3d.shape[0]
        # ic(self.generation_num_frames)

        # original motion
        (self.original_lh_3d, self.original_rh_3d, self.original_body_3d, self.original_instrument_3d,
         self.original_bow_3d, self.original_cp_3d, self.cp_info, self.used_finger_idx, self.original_R_s) = get_original_motion(original_motion_file, original_body_file)
        self.original_num_frames = self.original_lh_3d.shape[0]
        # ic(self.original_num_frames)

        # original audio pitch & info
        self.pitch_results = pitch_detect_crepe(crepe_backend='torch', proj=None, audio_path=input_audio_file)
        self.audio_frame_num = self.pitch_results.shape[0]

        # tip index
        self.tip_idx = [16, 17, 19, 18]

        # instrument index
        self.string_points = {
            0: (2, 4),  # 1弦
            1: (8, 10),  # 2弦
            2: (7, 9),  # 3弦
            3: (1, 3)  # 4弦
        }

        if not vis == True:
            # valid num frame, slice motion data
            self.valid_num_frame = min(self.generation_num_frames, self.audio_frame_num)
            # self.valid_num_frame = self.generation_num_frames
            vars_to_slice = ['generation_lh_3d', 'generation_rh_3d', 'generation_body_3d', 'generation_instrument_3d', 'generation_bow_3d',
                             'original_lh_3d', 'original_rh_3d', 'original_body_3d', 'original_instrument_3d', 'original_bow_3d', 'original_cp_3d',
                             'cp_info', 'used_finger_idx', 'original_R_s']
            for var in vars_to_slice:
                setattr(self, var, getattr(self, var)[:self.valid_num_frame])


    def _detect_note_changes_from_audio(self, confidence_threshold=0.5, pitch_change_threshold=50, min_frames=5):

        # 得到每一帧的音高结果
        pitch_results = self.pitch_results[:self.valid_num_frame, :]

        # 过滤掉置信度低的检测
        pitch_results[pitch_results[:, 2] < confidence_threshold, 1] = 0

        # 初始化变量
        note_changes = []
        current_pitch = 0
        change_duration = 0
        last_change_frame_idx = -1

        for i in range(0, len(pitch_results)):
            current_freq = pitch_results[i, 1]

            # 如果当前帧的音高频率为0Hz，跳过
            if current_freq == 0:
                continue

            # 计算音高变化（音分为单位）
            freq_change = 1200 * np.log2(current_freq / (current_pitch + 1e-8))

            # 如果音高变化大于50音分，更新换音持续计数
            if abs(freq_change) > pitch_change_threshold:
                change_duration += 1
            else:
                change_duration = 0

            # 如果音高变化持续超过5帧，认为发生了换音
            if change_duration >= min_frames:
                last_change_frame_idx = i - (min_frames - 1)
                note_changes.append(last_change_frame_idx)
                current_pitch = current_freq  # 更新当前音符的起始频率
                change_duration = 0  # 重置换音计数器

        # ic(len(note_changes))

        return note_changes

    def _detect_note_changes_from_audio11(self, confidence_threshold=0.9, pitch_change_threshold=50, min_frames=5):
        # 得到每一帧的音高结果
        pitch_results = self.pitch_results[:self.valid_num_frame, :]
        flag_column = np.ones((pitch_results.shape[0], 1))
        pitch_results = np.hstack((pitch_results, flag_column))

        # 过滤掉置信度低的检测
        pitch_results[pitch_results[:, 2] < confidence_threshold, 3] = 0
        ic(pitch_results[30:60])

        # 初始化变量
        note_changes = []
        current_pitch = 0
        change_duration = 0
        last_change_frame_idx = -1

        for i in range(0, len(pitch_results)):
            current_freq = pitch_results[i, 1]
            current_flag = pitch_results[i, 3]

            # 如果当前帧的音高频率为0Hz，跳过
            if current_flag == 0:
                change_duration = 0
                continue

            # 计算音高变化（音分为单位）
            freq_change = 1200 * np.log2(current_freq / (current_pitch + 1e-8))

            # 如果音高变化大于50音分，更新换音持续计数
            if abs(freq_change) > pitch_change_threshold:
                change_duration += 1
            else:
                change_duration = 0

            # 如果音高变化持续超过5帧，认为发生了换音
            if change_duration >= min_frames:
                last_change_frame_idx = i - (min_frames - 1)
                note_changes.append(last_change_frame_idx)
                current_pitch = current_freq  # 更新当前音符的起始频率
                change_duration = 0  # 重置换音计数器
            # 如果上一个换音帧与当前帧之间存在被置0的帧，且当前帧音高变化不满足50音分，但已持续超过5帧，也认为发生了换音
            # elif last_change_frame_idx != -1:
            #     frames_between_changes = pitch_results[last_change_frame_idx + 1:i, 3]
            #     zero_frames = np.sum(frames_between_changes == 0)
            #     if zero_frames > 2:
            #         freq_change_temp_list = []
            #         # for temp_i in range(1, min_frames):
            #         #     current_freq_next = pitch_results[min(i+temp_i, len(pitch_results)-1), 1]
            #         #     freq_change_temp = 1200 * np.log2(current_freq_next / (current_freq + 1e-8))
            #         #     freq_change_temp_list.append(freq_change_temp)
            #         # if all(x <= pitch_change_threshold for x in freq_change_temp_list):
            #         freq_change_temp = 1200 * np.log2(current_freq / (current_pitch + 1e-8))
            #         if abs(freq_change_temp) < pitch_change_threshold:
            #             last_change_frame_idx = i - (min_frames - 1)
            #             note_changes.append(last_change_frame_idx)
            #             current_pitch = current_freq
            #             change_duration = 0


        ic(len(note_changes))

        return note_changes


    def _get_bow_changing_from_motion(self, bow_motion_clip, instrument_motion_clip, min_bow_change_interval=0, max_valid_frame=np.inf):

        # 提取弓根的坐标（bow_motion_clip[:, 0, :]）和桥的坐标（instrument_motion_clip[:, 4, :]）
        bow_root_positions = bow_motion_clip[:max_valid_frame, 0, :]
        bridge_r_positions = instrument_motion_clip[:max_valid_frame, 4, :]

        # 计算弓根与bridge_r的欧氏距离
        distances = np.linalg.norm(bow_root_positions - bridge_r_positions, axis=1)

        # 计算距离的变化
        distance_diff = np.diff(distances)

        # 检测距离变化的反转点
        change_bow_indices = np.where(np.sign(distance_diff[:-1]) != np.sign(distance_diff[1:]))[0] + 1

        # 用来存储已选择的换弓点的索引和方向
        valid_bow_changes = []
        bow_directions = []

        # 处理换弓反转点，排除短周期内的连续换弓
        for i, idx in enumerate(change_bow_indices):
            if i == 0:
                # 初始时，直接选择第一个反转点
                valid_bow_changes.append(idx)
                bow_directions.append("downbow" if distances[idx] < distances[idx - 1] else "upbow")
            else:
                # 计算反转点与前一个反转点的时间间隔，排除短周期的连续换弓
                interval = idx - change_bow_indices[i - 1]
                if interval > min_bow_change_interval:
                    # 如果时间间隔大于设定阈值，才认为是有效的换弓
                    valid_bow_changes.append(idx)
                    bow_directions.append("downbow" if distances[idx] < distances[idx - 1] else "upbow")

        # 返回每一帧对应的弓根和桥之间的距离、换弓时机的帧数索引和换弓时机对应的方向
        return distances, valid_bow_changes, bow_directions


    def _get_bow_changing_from_motion_0(self, bow_motion_clip, instrument_motion_clip, min_bow_change_interval=0, max_valid_frame=np.inf):

        # 提取弓根的坐标（bow_motion_clip[:, 0, :]）
        bow_root_positions = bow_motion_clip[:max_valid_frame, 0, :]
        bridge_l_positions = instrument_motion_clip[0, 3, :]
        bridge_r_positions = instrument_motion_clip[0, 4, :]
        down_bow_vector = bridge_l_positions - bridge_r_positions

        # 用来存储已选择的换弓点的索引和方向
        valid_bow_changes = []
        bow_directions = []

        for f in range(1, len(bow_motion_clip)-1):
            bow_root_previous = bow_root_positions[f-1]
            bow_root_current = bow_root_positions[f]
            bow_root_next = bow_root_positions[f+1]

            bow_vector_1 = bow_root_current - bow_root_previous
            bow_vector_2 = bow_root_next - bow_root_current

            if f in [255, 256, 257, 258, 259,260, 261, 262, 263]:
                ic(np.dot(bow_vector_1, bow_vector_2))

            if np.dot(bow_vector_1, bow_vector_2) < 0:
                valid_bow_changes.append(f)
                bow_directions.append("downbow" if np.dot(bow_vector_2, down_bow_vector) < 0 else "upbow")

        # 计算弓根与bridge_r的欧氏距离
        distances = np.linalg.norm(bow_root_positions - bridge_r_positions, axis=1)

        # 返回每一帧对应的弓根和桥之间的距离、换弓时机的帧数索引和换弓时机对应的方向
        return distances, valid_bow_changes, bow_directions

    def _calculate_loudness(self, wav_file_path, frame_size, hop_size):
        """
        计算音频信号的响度（均方根，RMS），输入为WAV文件路径。

        :param wav_file_path: WAV文件路径
        :param frame_size: 每帧大小（采样点数）
        :param hop_size: 帧之间的间隔（采样点数）
        :return: 音频信号的响度曲线（每帧的RMS值）
        """
        # 读取WAV文件
        sample_rate, audio_signal = wavfile.read(wav_file_path)

        # 如果音频是立体声（多通道），则选择单个通道（例如左声道）
        if len(audio_signal.shape) > 1:
            audio_signal = audio_signal[:, 0]  # 选择第一个通道

        # 计算帧数
        num_frames = (len(audio_signal) - frame_size) // hop_size + 1
        loudness = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            frame = audio_signal[start:end]

            # 计算均方根（RMS）
            loudness[i] = np.sqrt(np.mean(frame ** 2))

        return loudness


    def _plot_bow_change(self, distances, change_bow_indices_distance, bow_directions_distance, note_changes):
        """
        绘制弓根与琴桥的欧氏距离图，并标出上弓和下弓的帧数
        :param distances: 每一帧的弓根和琴桥之间的欧氏距离
        :param change_bow_indices_distance: 换弓时机的帧数索引
        :param bow_directions_distance: 每个换弓时机的方向（'upbow' 或 'downbow'）
        """
        plt.figure(figsize=(10, 6))

        # 绘制距离曲线
        plt.plot(range(len(distances)), distances, label="Distance between Bow Root and Bridge_r", color='green')

        # 用红色标出上弓的帧
        for idx, direction in zip(change_bow_indices_distance, bow_directions_distance):
            if direction == "upbow":
                plt.scatter(idx, distances[idx], color='red', zorder=5)  # 标出上弓的帧
                plt.text(idx, distances[idx], str(idx), color='red', fontsize=9, verticalalignment='bottom',
                         horizontalalignment='center')

        # 用蓝色标出下弓的帧
        for idx, direction in zip(change_bow_indices_distance, bow_directions_distance):
            if direction == "downbow":
                plt.scatter(idx, distances[idx], color='blue', zorder=5)  # 标出下弓的帧
                plt.text(idx, distances[idx], str(idx), color='blue', fontsize=9, verticalalignment='top',
                         horizontalalignment='center')

        # 用细线标出音符变化的帧
        for idx in note_changes:
            plt.vlines(x=idx, ymin=min(distances), ymax=max(distances), color='black', linestyle='dashed',
                       linewidth=1)  # 画出细线
            plt.text(idx, max(distances), str(idx), color='black', fontsize=6, verticalalignment='bottom',
                     horizontalalignment='center')

        # 设置图表标题和标签
        plt.xlabel('Frame Number')
        plt.ylabel('Distance')
        plt.title('Distance between Bow Root and Bridge_r with Bow Changes')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def _plot_bow_change_compare(self, distances1, change_bow_indices_distance1, bow_directions_distance1,
                         distances2, change_bow_indices_distance2, bow_directions_distance2, note_changes):
        """
        绘制两个弓根与琴桥的欧氏距离图，并标出上弓和下弓的帧数（分别标出两个弓的数据）

        :param distances1: 第一个弓的每一帧的弓根和琴桥之间的欧氏距离
        :param change_bow_indices_distance1: 第一个弓的换弓时机的帧数索引
        :param bow_directions_distance1: 第一个弓的每个换弓时机的方向（'upbow' 或 'downbow'）
        :param distances2: 第二个弓的每一帧的弓根和琴桥之间的欧氏距离
        :param change_bow_indices_distance2: 第二个弓的换弓时机的帧数索引
        :param bow_directions_distance2: 第二个弓的每个换弓时机的方向（'upbow' 或 'downbow'）
        :param note_changes: 音符变化的帧索引
        """
        plt.figure(figsize=(10, 6))

        # 绘制第一个弓的距离曲线
        plt.plot(range(len(distances1)), distances1, label="Generated", color='green')

        # 绘制第二个弓的距离曲线
        plt.plot(range(len(distances2)), distances2, label="Original", color='orange')

        # 用红色标出第一个弓的上弓的帧
        for idx, direction in zip(change_bow_indices_distance1, bow_directions_distance1):
            if direction == "upbow":
                plt.scatter(idx, distances1[idx], color='green', zorder=5)  # 标出上弓的帧
                plt.text(idx, distances1[idx], str(idx), color='green', fontsize=9, verticalalignment='bottom',
                         horizontalalignment='center')

        # 用蓝色标出第一个弓的下弓的帧
        for idx, direction in zip(change_bow_indices_distance1, bow_directions_distance1):
            if direction == "downbow":
                plt.scatter(idx, distances1[idx], color='green', zorder=5)  # 标出下弓的帧
                plt.text(idx, distances1[idx], str(idx), color='green', fontsize=9, verticalalignment='top',
                         horizontalalignment='center')

        # 用红色标出第二个弓的上弓的帧
        for idx, direction in zip(change_bow_indices_distance2, bow_directions_distance2):
            if direction == "upbow":
                plt.scatter(idx, distances2[idx], color='orange', zorder=5)  # 标出上弓的帧
                plt.text(idx, distances2[idx], str(idx), color='orange', fontsize=9, verticalalignment='bottom',
                         horizontalalignment='center')

        # 用蓝色标出第二个弓的下弓的帧
        for idx, direction in zip(change_bow_indices_distance2, bow_directions_distance2):
            if direction == "downbow":
                plt.scatter(idx, distances2[idx], color='orange', zorder=5)  # 标出下弓的帧
                plt.text(idx, distances2[idx], str(idx), color='orange', fontsize=9, verticalalignment='top',
                         horizontalalignment='center')

        # 用细线标出音符变化的帧
        for idx in note_changes:
            plt.vlines(x=idx, ymin=min(distances1), ymax=max(distances1), color='black', linestyle='dashed',
                       linewidth=1)  # 画出细线
            plt.text(idx, max(distances1), str(idx), color='black', fontsize=6, verticalalignment='bottom',
                     horizontalalignment='center')

        # 设置图表标题和标签
        plt.xlabel('Frame Number')
        plt.ylabel('Distance')
        plt.title('Distance between Bow Root and Bridge_r with Bow Changes')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def _plot_bow_change_with_loudness(self, distances1, change_bow_indices_distance1, bow_directions_distance1,
                                       distances2, change_bow_indices_distance2, bow_directions_distance2,
                                       note_changes, loudness):
        """
        绘制两个弓根与琴桥的欧氏距离图，并标出上弓和下弓的帧数，及响度曲线。

        :param distances1: 第一个弓的每一帧的弓根和琴桥之间的欧氏距离
        :param change_bow_indices_distance1: 第一个弓的换弓时机的帧数索引
        :param bow_directions_distance1: 第一个弓的每个换弓时机的方向（'upbow' 或 'downbow'）
        :param distances2: 第二个弓的每一帧的弓根和琴桥之间的欧氏距离
        :param change_bow_indices_distance2: 第二个弓的换弓时机的帧数索引
        :param bow_directions_distance2: 第二个弓的每个换弓时机的方向（'upbow' 或 'downbow'）
        :param note_changes: 音符变化的帧索引
        :param loudness: 音频的响度曲线
        :param fps: 帧率，默认为30
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 绘制第一个弓的距离曲线
        ax1.plot(range(len(distances1)), distances1, label="Generated Bow", color='green')

        # 绘制第二个弓的距离曲线
        ax1.plot(range(len(distances2)), distances2, label="Original Bow", color='orange')

        # 用红色标出第一个弓的上弓的帧
        for idx, direction in zip(change_bow_indices_distance1, bow_directions_distance1):
            if direction == "upbow":
                ax1.scatter(idx, distances1[idx], color='green', zorder=5)
                ax1.text(idx, distances1[idx], str(idx), color='green', fontsize=9, verticalalignment='bottom',
                         horizontalalignment='center')

        # 用蓝色标出第一个弓的下弓的帧
        for idx, direction in zip(change_bow_indices_distance1, bow_directions_distance1):
            if direction == "downbow":
                ax1.scatter(idx, distances1[idx], color='green', zorder=5)
                ax1.text(idx, distances1[idx], str(idx), color='green', fontsize=9, verticalalignment='top',
                         horizontalalignment='center')

        # 用红色标出第二个弓的上弓的帧
        for idx, direction in zip(change_bow_indices_distance2, bow_directions_distance2):
            if direction == "upbow":
                ax1.scatter(idx, distances2[idx], color='orange', zorder=5)
                ax1.text(idx, distances2[idx], str(idx), color='orange', fontsize=9, verticalalignment='bottom',
                         horizontalalignment='center')

        # 用蓝色标出第二个弓的下弓的帧
        for idx, direction in zip(change_bow_indices_distance2, bow_directions_distance2):
            if direction == "downbow":
                ax1.scatter(idx, distances2[idx], color='orange', zorder=5)
                ax1.text(idx, distances2[idx], str(idx), color='orange', fontsize=9, verticalalignment='top',
                         horizontalalignment='center')

        # 用细线标出音符变化的帧
        for idx in note_changes:
            ax1.vlines(x=idx, ymin=min(distances1), ymax=max(distances1), color='black', linestyle='dashed',
                       linewidth=1)
            ax1.text(idx, max(distances1), str(idx), color='black', fontsize=6, verticalalignment='bottom',
                     horizontalalignment='center')

        # 设置左侧y轴的标签和标题
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Distance', color='black')
        ax1.set_title('Bow Change and Loudness with Frame Number')
        ax1.grid(True)

        # 创建第二个y轴，用于响度曲线
        ax2 = ax1.twinx()
        ax2.plot(np.linspace(0, self.valid_num_frame, len(loudness)), loudness, label="Loudness (RMS)", color='blue', linewidth=1.5)
        ax2.set_ylabel('Loudness (RMS)', color='blue')

        # 设置右侧y轴的标签
        ax2.tick_params(axis='y', labelcolor='blue')

        # 显示图例
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()

    def _compute_min_distance(self, arr1, arr2):
        arr1_expanded = arr1[:, :, np.newaxis, :]
        arr2_expanded = arr2[:, np.newaxis, :, :]

        distances = np.linalg.norm(arr1_expanded - arr2_expanded, axis=-1)

        min_distances = np.min(distances, axis=(1, 2))

        min_distances = min_distances[~np.isnan(min_distances)]

        return min_distances

    def _compute_min_distance0(self, cp_coordinates, hand_tips):
        min_distance = []
        min_indices = []

        for i in range(cp_coordinates.shape[0]):
            # 当前帧的目标和手指坐标
            cp_coords_sample = cp_coordinates[i]
            hand_tips_sample = hand_tips[i]

            # 计算距离矩阵
            distances = np.linalg.norm(
                cp_coords_sample[:, np.newaxis, :] - hand_tips_sample[np.newaxis, :, :],
                axis=-1
            )

            # 将 NaN 值掩码设置到距离矩阵
            nan_mask = (
                    np.isnan(cp_coords_sample).any(axis=1)[:, np.newaxis] |
                    np.isnan(hand_tips_sample).any(axis=1)[np.newaxis, :]
            )
            distances[nan_mask] = np.nan

            # 找到非 NaN 的最小值及其位置
            if np.isnan(distances).all():
                min_distance.append(None)
                min_indices.append(None)
            else:
                min_value = np.nanmin(distances)  # 忽略 NaN 值计算最小值
                min_index = np.unravel_index(np.nanargmin(distances), distances.shape)  # 获取最小值的位置
                min_distance.append(min_value)
                min_indices.append(min_index)
        # 转换为 NumPy 数组以便进一步处理
        min_distance = np.array(min_distance)
        min_indices = np.array(min_indices)

        return min_distance, min_indices

    def find_correct_cp_from4cp(self, cp_coordinates, string_indices):
        cp_coordinates_1 = cp_coordinates[np.arange(cp_coordinates.shape[0]), string_indices]
        return cp_coordinates_1

    def compute_fourtip_distance(self, cp_coordinates_1, hand_tips):
        distances = np.linalg.norm(hand_tips - cp_coordinates_1[:, np.newaxis, :], axis=-1)
        valid_frames = np.array([x for x in distances if not np.all(np.isnan(x))])
        average_distances_4finger_allframe = np.mean(valid_frames, axis=0)
        min_distance = valid_frames.min(axis=1)
        return average_distances_4finger_allframe, distances, min_distance


    def get_string_audio(self, cps, tips):
        cps_expanded = cps[:, :, np.newaxis, :]
        tips_expanded = tips[:, np.newaxis, :, :]

        distances = np.linalg.norm(cps_expanded - tips_expanded, axis=-1)

        all_nan_mask = np.all(np.isnan(distances), axis=(1, 2))

        min_indices = np.full(distances.shape[0], np.nan)

        valid_batches = ~all_nan_mask
        if np.any(valid_batches):
            flat_indices = np.nanargmin(distances[valid_batches].reshape(-1, 16), axis=1)
            min_indices[valid_batches] = flat_indices // 4

        min_indices = np.nan_to_num(min_indices, nan=-1).astype(int)

        return min_indices, all_nan_mask


    def _cal_cp_coordinates(self, max_len, instrument_pos):
        data_manip = DataManipulatorCello()
        pitch_results = self.pitch_results

        pitch_with_positions = freq_position.get_contact_position(pitch_results)
        positions = pitch_with_positions[:, -4:]
        positions = positions[:max_len]

        cp_info = []
        for string_id, ratio in enumerate(zip(*positions)):
            ratio_arr = np.array(ratio)
            col_info = np.stack((np.full(ratio_arr.shape, string_id), ratio_arr), axis=1)
            cp_info.append(col_info)

        cp_info = np.array(cp_info)  # (string_index, frame_num, ratio)

        values = cp_info[:, :, 1]
        all_negative = np.all(values == -1, axis=0)
        cp_info[:, all_negative, 0] = -1

        cp_coordinates = []
        for string_idx, string_cp_info in enumerate(cp_info):
            string_cp_coordinates = data_manip.calculate_cp_coordinates(string_cp_info, instrument_pos)
            cp_coordinates.append(string_cp_coordinates)

        cp_coordinates = np.array(cp_coordinates)

        cp_coordinates = cp_coordinates.transpose(1, 0, 2)
        return cp_coordinates


    def _cal_bowing_metrics(self, bow_pred_dict, bow_gt_dict, tolerance, direction_match=False):
        gt_bow_changes = sorted(bow_gt_dict.keys())
        pred_bow_changes = sorted(bow_pred_dict.keys())

        gt_matches = []
        pred_matches = []

        for gt in gt_bow_changes:
            closest_pred = None
            min_frame_distance = float('inf')

            for pred in pred_bow_changes:
                if pred not in pred_matches and (pred - tolerance <= gt <= pred + tolerance):
                    if direction_match:
                        if bow_pred_dict[pred] == bow_gt_dict[gt]:
                            distance = abs(gt - pred)
                            if distance < min_frame_distance:
                                closest_pred = pred
                                min_frame_distance = distance
                    else:
                        distance = abs(gt - pred)
                        if distance < min_frame_distance:
                            closest_pred = pred
                            min_frame_distance = distance

            if closest_pred is not None:
                gt_matches.append(gt)
                pred_matches.append(closest_pred)


        # ic(gt_matches)
        # ic(pred_matches)

        precision = len(pred_matches) / len(pred_bow_changes) if pred_bow_changes else 0
        recall = len(gt_matches) / len(gt_bow_changes) if gt_bow_changes else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        details = {"num_true_pred": len(pred_matches),
                   "num_pred": len(pred_bow_changes),
                   "num_matched_gt": len(gt_matches),
                   "num_gt": len(gt_bow_changes)}

        return precision, recall, f1, details



    def validate_bowing_from_audio(self,):
        # get note changing from audio
        note_changing = self._detect_note_changes_from_audio()
        # get generated bowing from motion
        frog_distances, bow_changing, bow_directions = self._get_bow_changing_from_motion(self.generation_bow_3d,
                                                                                          self.generation_instrument_3d,
                                                                                          max_valid_frame=self.valid_num_frame)
        frog_distances_original, bow_changing_original, bow_directions_original = self._get_bow_changing_from_motion(
                                                                                          self.original_bow_3d,
                                                                                          self.original_instrument_3d,
                                                                                          max_valid_frame=self.valid_num_frame)

        bow_change_dict = {bow_changing[i] : bow_directions[i] for i in range(len(bow_changing))}
        bow_change_original_dict = {bow_changing_original[i]: bow_directions_original[i] for i in range(len(bow_changing_original))}

        precision, recall, f1, details = self._cal_bowing_metrics(bow_change_dict, bow_change_original_dict, tolerance=3, direction_match=True)

        # ic(precision, recall, f1)



        # self._plot_bow_change_compare(frog_distances, bow_changing, bow_directions,
        #                               frog_distances_original, bow_changing_original, bow_directions_original,
        #                               note_changing)

        # self._plot_bow_change_with_loudness(frog_distances, bow_changing, bow_directions,
        #                               frog_distances_original, bow_changing_original, bow_directions_original, note_changing,
        #                               loudness)
        # loudness = self._calculate_loudness(self.input_audio_file, frame_size=2048, hop_size=512)
        # ic(loudness.shape)
        # self._plot_bow_change(frog_distances, bow_changing, bow_directions, note_changing)
        return precision, recall, f1, details

    def cal_cosine_similarity(self, pred_bow, gt_bow, instrument_pose, frog_tip=False):
        num_frames = gt_bow.shape[0]  # 获取帧数

        if frog_tip:
            # vector 是 frog 到 tip
            vector_gt = gt_bow[:, 1] - gt_bow[:, 0]
            vector_pred = pred_bow[:, 1] - pred_bow[:, 0]
        else:
            # vector 是 origin 到 frog
            # vector_gt = gt_bow[:, 0]
            # vector_pred = pred_bow[:, 0]

            # bridge 到 frog
            vector_gt = gt_bow[:, 0] - instrument_pose[:, 3]
            vector_pred = pred_bow[:, 0] - instrument_pose[:, 3]


        # 计算每一帧的余弦相似度
        cosine_similarities = np.zeros(num_frames)
        for i in range(num_frames):
            # frog
            cosine_similarities[i] = self._cosine_similarity(vector_gt[i], vector_pred[i])

        # 返回所有帧的平均余弦相似度
        return np.mean(cosine_similarities)

    def cal_cosine_similarity_coordinates(self, pred_bow, gt_bow, instrument_pose, frog_tip=False):
        num_frames = gt_bow.shape[0]  # 获取帧数
        total_cs = 0.0

        for i in range(num_frames):
            frame_pred = pred_bow[i, 0]
            frame_gt = gt_bow[i, 0]

            for j in range(3):
                total_cs += self._cosine_similarity(frame_pred[j], frame_gt[j])
        avg_cs = total_cs / (num_frames * 3)

        return avg_cs

    def cal_cosine_similarity_framewise(self, pred_bow, gt_bow, instrument_pose):
        """弓根在x、y、z轴上坐标的时间序列的vector的余弦相似度，再cross coordinates取平均"""
        assert gt_bow.shape[0] == pred_bow.shape[0]
        num_frames = gt_bow.shape[0]  # 获取帧数

        vector_x_pred = []
        vector_x_gt = []
        vector_y_pred = []
        vector_y_gt = []
        vector_z_pred = []
        vector_z_gt = []

        for frame_i in range(num_frames):
            vector_x_pred.append(pred_bow[frame_i, 0, 0] - instrument_pose[frame_i, 3, 0])    # dim=1中 index=0为frog， index=1为bow tip， dim=2中 代表x\y\z轴上的坐标分量
            vector_x_gt.append(gt_bow[frame_i, 0, 0] - instrument_pose[frame_i, 3, 0])
            vector_y_pred.append(pred_bow[frame_i, 0, 1] - instrument_pose[frame_i, 3, 1])
            vector_y_gt.append(gt_bow[frame_i, 0, 1] - instrument_pose[frame_i, 3, 1])
            vector_z_pred.append(pred_bow[frame_i, 0, 2] - instrument_pose[frame_i, 3, 2])
            vector_z_gt.append(gt_bow[frame_i, 0, 2] - instrument_pose[frame_i, 3, 2])

        vector_x_pred = np.array(vector_x_pred)
        vector_x_gt = np.array(vector_x_gt)
        vector_y_pred = np.array(vector_y_pred)
        vector_y_gt = np.array(vector_y_gt)
        vector_z_pred = np.array(vector_z_pred)
        vector_z_gt = np.array(vector_z_gt)

        cs_x = self._cosine_similarity(vector_x_pred, vector_x_gt)
        cs_y = self._cosine_similarity(vector_y_pred, vector_y_gt)
        cs_z = self._cosine_similarity(vector_z_pred, vector_z_gt)

        cs_average = (cs_x + cs_y + cs_z) / 3

        ic(vector_x_pred.shape)
        ic(vector_x_gt.shape)
        ic(cs_x)
        ic(cs_y)
        ic(cs_z)
        ic(cs_average)




    def _cosine_similarity(self, a, b):
        """
        计算两个向量之间的余弦相似度
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def _line_to_line_distance(self, target_string, target_bow):
        """
        计算两条直线之间的距离。

        :param target_string: tuple((x1, y1, z1), (x2, y2, z2))，第一条直线的起点和终点
        :param target_bow: tuple((x3, y3, z3), (x4, y4, z4))，第二条直线的起点和终点
        :return: 两条直线之间的最短距离
        """
        # 起点和终点
        p1, p2 = target_string
        q1, q2 = target_bow

        # 方向向量
        d1 = np.array(p2) - np.array(p1)  # 第一条直线的方向向量
        d2 = np.array(q2) - np.array(q1)  # 第二条直线的方向向量

        # 两条直线的起点向量
        r = np.array(q1) - np.array(p1)

        # 叉乘和点乘
        cross_d1_d2 = np.cross(d1, d2)
        norm_cross_d1_d2 = np.linalg.norm(cross_d1_d2)

        if norm_cross_d1_d2 == 0:
            # 若两直线平行，计算点到直线的垂直距离
            # 垂直距离公式：|r x d1| / |d1|
            distance = np.linalg.norm(np.cross(r, d1)) / np.linalg.norm(d1)
        else:
            # 若两直线不平行，计算最短距离
            distance = abs(np.dot(r, cross_d1_d2)) / norm_cross_d1_d2

        return distance

    def cal_gt_string_bow_distance(self, bow_motion, instrument_motion, cp_info):
        num_frame = len(cp_info)
        cnt = num_frame
        distance_total = 0

        for frame_idx in range(num_frame):
            if cp_info[frame_idx, 0] == -1:     # 当前帧无cp
                cnt -= 1
            else:                               # 有cp的帧
                target_string = (instrument_motion[frame_idx, self.string_points[cp_info[frame_idx, 0]][0]],
                                 instrument_motion[frame_idx, self.string_points[cp_info[frame_idx, 0]][1]])
                target_bow = (bow_motion[frame_idx, 0],
                              bow_motion[frame_idx, 1])
                distance_idx = self._line_to_line_distance(target_string, target_bow)
                distance_total += distance_idx

        bow_string_avg_distance = distance_total / cnt

        return bow_string_avg_distance, distance_total, cnt

    def cal_finger_string_distance(self, lefthand_motion, cp_3d, used_finger_index):

        cp_3d = cp_3d.squeeze()
        used_finger_3d = np.zeros((self.valid_num_frame, 3))
        for frame_i in range(len(used_finger_index)):
            if used_finger_index[frame_i] == -1:
                used_finger_3d[frame_i] = np.array([np.nan, np.nan, np.nan])
            else:
                used_finger_3d[frame_i] = lefthand_motion[frame_i, self.tip_idx[used_finger_index[frame_i]]]
        # ic(cp_3d.shape)
        # ic(used_finger_3d.shape)

        diff = cp_3d - used_finger_3d
        squared_diff = diff ** 2
        distances = np.sqrt(np.sum(squared_diff, axis=1))
        nan_mask = np.any(np.isnan(cp_3d), axis=1) | np.any(np.isnan(used_finger_3d), axis=1)
        distances[nan_mask] = np.nan

        return distances


    def validate_contact_deviation(self,):
        hand_tips = self.generation_lh_3d[:self.valid_num_frame, self.tip_idx]
        # hand_tips = self.original_lh_3d[:self.valid_num_frame, self.tip_idx]

        # Audio infered CP, 最早，4根弦根据距离筛1跟
        cp_coordinates = self._cal_cp_coordinates(self.valid_num_frame, self.generation_instrument_3d)
        min_distances_old = self._compute_min_distance(cp_coordinates, hand_tips)

        # Audio infered CP, 最早，4根弦直接通过音频筛出唯一cp
        string_idices, _ = self.get_string_audio(cp_coordinates, hand_tips)
        cp_coordinates_1 = self.find_correct_cp_from4cp(cp_coordinates, string_idices)
        fourtip_avg_distance, distance_allframe, min_distances_audio = self.compute_fourtip_distance(cp_coordinates_1, hand_tips)

        # MOCAP CP
        cp_coordinates_1_mocap = self.original_cp_3d.squeeze()
        fourtip_avg_distance, distance_allframe, min_distances_mocap = self.compute_fourtip_distance(cp_coordinates_1_mocap, hand_tips)
        # cnt = len(cp_coordinates_1_mocap) - np.sum(np.all(np.isnan(cp_coordinates_1_mocap), axis=1))
        # ic(min_distances_mocap.shape)

        # min_distances, min_indices = self._compute_min_distance(cp_coordinates, hand_tips)
        # ic(min_distances.shape)

        lh_avg_distance = np.mean(min_distances_old)
        # lh_avg_distance = np.mean(min_distances_audio)
        # lh_avg_distance = np.mean(min_distances_mocap)
        # ic(lh_avg_distance)

        cnt = len(string_idices)
        distance_total = 0
        for frame_idx, string in enumerate(string_idices):
            if string == -1:
                cnt -= 1
            else:
                target_string = (self.generation_instrument_3d[frame_idx, self.string_points[string][0]],
                                 self.generation_instrument_3d[frame_idx, self.string_points[string][1]])
                target_bow = (self.generation_bow_3d[frame_idx, 0],
                              self.generation_bow_3d[frame_idx, 1])
                # target_bow = (self.original_bow_3d[frame_idx, 0],
                #               self.original_bow_3d[frame_idx, 1])
                distance_idx = self._line_to_line_distance(target_string, target_bow)
                distance_total += distance_idx
        bow_string_avg_distance = distance_total / cnt
        # ic(bow_string_avg_distance)
        # ic(cnt)

        return lh_avg_distance, bow_string_avg_distance, cnt, distance_allframe, fourtip_avg_distance
