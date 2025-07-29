import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import glob
import json

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import signal
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
from data_process.data_manipulation import DataManipulatorCello
from scipy.optimize import minimize

ORIGIN_INDEX = 139  # tail_gut = 138, end_pin = 139
LH_WRIST_INDEX = 91
RH_WRIST_INDEX = 112
CONTACT_POINT_INDEX = 150
# CELLO_IDX_RANGE = (133, 142)  # with bow
CELLO_IDX_RANGE = (133, 140)  # without bow
BOW_IDX_RANGE = (140, 142)
SEQ_LEN = 150  # number of frames per sequence
HOP_LEN = 30  # gap size between consecutive sequence
FRAME_RATE = 30
SAMPLE_NUMBER = 300000000000
BODY_IDX_RANGE = (0, 23)
USED_FINGER_TIP_IDX = 154
FINGER_TIP_IDX = [99, 103, 107, 111]

INSTRUMENT_LINKS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 5], [5, 6], [7, 8]]
HAND_LINKS = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [0, 10],
              [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [3, 16], [6, 17], [9, 18], [12, 19], [15, 20]]
MANO_PARENTS_INDICES = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]
BONE_LENGTHS = np.array([0.094, 0.0335, 0.024, 0.094, 0.034, 0.03, 0.081, 0.029, 0.019, 0.087,
                         0.035, 0.027, 0.039, 0.034, 0.032, 0.019, 0.022, 0.019, 0.022, 0.023])


def normalize_keypoints_cello(data):
    data = np.array(data)
    frame_num, num_keypoints, coords = data.shape
    # make sure 155 in dim2, 3 in dim3
    assert num_keypoints == 155 and coords == 3
    # end_pin_index = 139, LH_WIRST_INDEX = 91
    origin_location = data[:, ORIGIN_INDEX, :]
    normalized_data = data - origin_location[:, np.newaxis, :]
    lh_trans = normalized_data[:, LH_WRIST_INDEX:LH_WRIST_INDEX + 1, :]
    rh_trans = normalized_data[:, RH_WRIST_INDEX:RH_WRIST_INDEX + 1, :]
    return normalized_data, lh_trans, rh_trans
    # LH_WRIST_INDEX:LH_WRIST_INDEX+1 是为了保持维度不变，相当于np.newaxis


def find_npy_files(directory):
    npy_files = sorted(glob.glob(os.path.join(directory, '**', '*.npy'), recursive=True))
    return npy_files


def kabsch_algorithm(P, Q):
    """
    Compute the optimal rotation matrix using the Kabsch algorithm to align point set Q to point set P.

    P: reference point set (6, 3)
    Q: target point set (6, 3)

    Returns the rotation matrix (3, 3)
    """
    # Compute the covariance matrix
    H = np.dot(Q.T, P)

    # Perform singular value decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation (determinant of R = 1, no reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R


def align_cello_keypoints(cello_data, reference_cello):
    """
    Align cellos by rotating them around the tail_gut (6th key point) so that they best fit the first cello.

    cello_data: numpy array of shape (n, 6, 3), where n is the number of cellos, and each has 7 key points in 3D

    reference_cello: numpy array of shape (6, 3), the target with which the cello_data to calibrate

    Returns:
    - aligned_cello_data: numpy array of the same shape as cello_data, but with the cellos aligned to the first one.
    - rotation_matrices: list of rotation matrices applied to each cello (except the first, which doesn't change)
    """
    n = cello_data.shape[0]

    cello_data = cello_data[:, 1:5, :]

    # Extract the reference points (only includes fingerboard)
    ref_keypoints = reference_cello[1:5]  # Shape (4, 3)

    aligned_cello_data = np.copy(cello_data)
    rotation_matrices = []

    loop_align = tqdm(range(0, n))
    for i in loop_align:
        loop_align.set_description(f'Aligning [{i}/{n}]')
        target_keypoints = cello_data[i]

        # Compute the optimal rotation matrix to align the current cello to the reference
        R = kabsch_algorithm(ref_keypoints, target_keypoints)
        rotation_matrices.append(R)

        # Apply the rotation to the key points (excluding the tail gut, which stays at the origin)
        rotated_keypoints = np.dot(target_keypoints, R.T)

        # Store the rotated cello back in the aligned_cello_data array
        aligned_cello_data[i] = rotated_keypoints

    rotation_matrices = np.array(rotation_matrices)

    return aligned_cello_data, rotation_matrices


def compute_average_shape(cello_data):
    """
    Compute the average shape of n aligned cellos, where each cello is represented by 7 key points.
    """
    # Calculate the mean across the first axis (the n cellos)
    average_shape = np.mean(cello_data, axis=0)

    return average_shape


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


def calculate_rotation_change(R_base, R_target_set):
    """
    Calculates the change in rotation matrices from R_base to each matrix in R_target_set.

    Parameters:
    R_base (np.array): Base rotation matrix of shape (3, 3).
    R_target_set (np.array): Target rotation matrices of shape (n, 3, 3).

    Returns:
    np.array: The set of change rotation matrices of shape (n, 3, 3).
    """
    # Compute the transpose (inverse) of the base rotation matrix
    R_base_inv = R_base.T

    # Compute the change in rotation matrices for each target matrix
    R_change_set = np.einsum('ijk,kl->ijl', R_target_set, R_base_inv)

    return R_change_set


def plot_cellos(cello_data):
    """
    Plots n cellos in 3D space with different colors.

    cello_data: numpy array of shape (n, 7, 3), representing the positions of 7 key points on n cellos.
    """
    n = cello_data.shape[0]

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate a list of unique colors for each cello
    colors = plt.cm.jet(np.linspace(0, 1, n))

    for i in range(n):
        cello = cello_data[i]
        color = colors[i]

        # Plot the key points of the cello
        ax.scatter(cello[..., 0], cello[..., 1], cello[..., 2], color=color, label=f'Cello {i + 1}')

        # Draw the links between the key points
        for link in INSTRUMENT_LINKS:
            p1, p2 = link
            ax.plot([cello[p1, 0], cello[p2, 0]],
                    [cello[p1, 1], cello[p2, 1]],
                    [cello[p1, 2], cello[p2, 2]], color=color, linewidth=0.3)

    # Labeling the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show legend and plot
    ax.legend()
    plt.show()


def set_equal_scale(ax, data):
    x_limits = [np.min(data[:, 0]), np.max(data[:, 0])]
    y_limits = [np.min(data[:, 1]), np.max(data[:, 1])]
    z_limits = [np.min(data[:, 2]), np.max(data[:, 2])]

    # Determine the overall range
    max_range = np.array([x_limits[1] - x_limits[0],
                          y_limits[1] - y_limits[0],
                          z_limits[1] - z_limits[0]]).max() / 2.0

    # Find the midpoints of the ranges
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    # Set the axes limits equally based on the midpoints and max_range
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_cellos_and_hands(cello_data, hand_pose):
    """
    Plots n cellos and corresponding hand poses in 3D space with different colors for each cello-hand pair.
    Ensures that the x, y, z axes have the same scaling.

    Parameters:
    cello_data (numpy array): Shape (n, 7, 3), representing the positions of 7 key points on n cellos.
    hand_pose (numpy array): Shape (n, 21, 3), representing the positions of 21 key points on n hands.
    """
    n = cello_data.shape[0]

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate a list of unique colors for each cello-hand pair
    colors = plt.cm.jet(np.linspace(0, 1, n))

    for i in range(n):
        cello = cello_data[i]
        hand = hand_pose[i]
        color = colors[i]

        # Plot the key points of the cello
        ax.scatter(cello[:, 0], cello[:, 1], cello[:, 2], color=color, label=f'Cello {i + 1}')
        # Draw the links between the key points of the cello
        for link in INSTRUMENT_LINKS:
            p1, p2 = link
            ax.plot([cello[p1, 0], cello[p2, 0]],
                    [cello[p1, 1], cello[p2, 1]],
                    [cello[p1, 2], cello[p2, 2]], color=color, linewidth=0.3)

        # Plot the key points of the hand
        ax.scatter(hand[:, 0], hand[:, 1], hand[:, 2], color=color, s=4, label=f'Hand {i + 1}')
        # Draw the links between the key points of the hand
        for link in HAND_LINKS:
            p1, p2 = link
            ax.plot([hand[p1, 0], hand[p2, 0]],
                    [hand[p1, 1], hand[p2, 1]],
                    [hand[p1, 2], hand[p2, 2]], color=color, linewidth=0.3)

    # Combine the cello and hand data for scaling
    combined_data = np.concatenate((cello_data.reshape(-1, 3), hand_pose.reshape(-1, 3)), axis=0)

    # Set equal scaling for the axes
    set_equal_scale(ax, combined_data)

    # Labeling the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show legend and plot
    ax.legend()
    plt.show()


def normalize_vector_batch(v_batch):
    """
    输入shape：(batch_size, 3)，对每一行分别归一化
    """
    # 计算每一行的模
    norms = np.linalg.norm(v_batch, axis=1, keepdims=True)
    # 防止除以0的情况，保留原值
    norms[norms == 0] = 1
    # 对每一行进行归一化
    normalized_arr = v_batch / norms
    return normalized_arr


def pad_with_identity(original_array, n):
    m = original_array.shape[0]

    # Create an empty array of shape (n, 3, 3)
    padded_array = np.zeros((n, 3, 3))

    # Copy the original array into the padded array
    padded_array[:m] = original_array

    # Fill the remaining rows with identity matrices
    padded_array[m:n] = np.eye(3)[None, :, :].repeat(n - m, axis=0)

    return padded_array


def batch_calculate_wrist_displacement_and_rotation(wrist_positions, batch_R_old, batch_R_origin):
    """
    Calculates the displacement and new rotation R_new for each wrist node to achieve a specified new pose.

    Parameters:
    - wrist_positions: numpy array of shape (n, 3), representing the initial positions of n wrist nodes.
    - batch_R_old: numpy array of shape (n, 3, 3), representing the initial orientation of each wrist node.
    - batch_R_origin: numpy array of shape (n, 3, 3), representing the rotation matrix for each wrist node around the origin. (given from the instrument rotations)

    Returns:
    - translations: numpy array of shape (n, 3), representing the displacement required for each wrist node.
    - R_new_matrices: numpy array of shape (n, 3, 3), representing the rotation matrix needed for each wrist node
                      to reach the new orientation after the displacement.
    """
    new_trans = np.einsum('nij,nj->ni', batch_R_origin, wrist_positions)

    new_R = np.einsum('nij,njk->nik', batch_R_origin, batch_R_old)

    return new_trans, new_R

def replace_nan_quaternions(quat_array, default_quat=np.array([1, 0, 0, 0])):
    nan_indices = np.isnan(quat_array).any(axis=1)
    quat_array[nan_indices] = default_quat
    return quat_array


def replace_zero_6d(array_6d, default_6d=np.array([1, 0, 0, 0, 1, 0])):
    norms = np.linalg.norm(array_6d, axis=1)
    array_6d[norms == 0] = default_6d
    return array_6d


def rotation_6d_to_R(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def find_cp_position(cp, instrument_7):
    """
    In terms of each frame, record the corresponding cp in the format of (activation_string, vibrating length)
    :param cp:  contact point of current frame, (3, )
    :param instrument_7:    (7, 3)
    :return:   activation string, vibrating length (0 - 1, 1 for open string)
    """
    # get the start and end points of the 1st and 4th strings
    start_1, start_4, end_1, end_4 = instrument_7[2], instrument_7[1], instrument_7[4], instrument_7[3]

    # calculate the start and end of each string
    strings = [
        (start_1, end_1),  # 1
        (start_1 * (2 / 3) + start_4 * (1 / 3), end_1 * (2 / 3) + end_4 * (1 / 3)),  # 2
        (start_1 * (1 / 3) + start_4 * (2 / 3), end_1 * (1 / 3) + end_4 * (2 / 3)),  # 3
        (start_4, end_4)  # 4
    ]

    min_distance = float('inf')
    activation_string = -1
    vibrating_length = None

    for i, (start, end) in enumerate(strings):
        # vector
        line_vec = end - start
        point_vec = cp - start
        line_len = np.linalg.norm(line_vec)

        # projection ratio
        proj_ratio = np.dot(point_vec, line_vec) / (line_len ** 2)
        proj_ratio = np.clip(proj_ratio, 0, 1)

        # calculate projection point
        proj_point = start + proj_ratio * line_vec

        # calculate the cp's positipn to each string
        distance = np.linalg.norm(cp - proj_point)

        # find the activation string
        if distance < min_distance:
            min_distance = distance
            activation_string = i
            vibrating_length = 1 - proj_ratio  # end:0, start:1

    return [activation_string, vibrating_length]


def get_used_finger_idx(used_finger_tip, finger_tips):
    """
    Args:
        used_finger_tip: (seq_len, 3)
        finger_tips: (seq_len, 4, 3) four fingertips, excluding thumb
    Returns: used_finger_idx: (seq_len,)
    """
    # ic(used_finger_tip.shape, finger_tips.shape)
    seq_len = used_finger_tip.shape[0]
    nan_used_indices = np.isnan(used_finger_tip).any(axis=1)

    used_finger_idx = np.zeros((seq_len,), dtype=int)
    for i in range(seq_len):
        if nan_used_indices[i]:
            used_finger_idx[i] = -1
            continue
        distances = np.linalg.norm(used_finger_tip[i] - finger_tips[i], axis=1)
        closest_index = np.argmin(distances)
        used_finger_idx[i] = closest_index

    return used_finger_idx


def get_bow_trans(p1, p2, p3, p4):
    """
    Calculate the shortest distance and translation vector for each frame
    between two skew lines L1 and L2, where points are in the shape (nframes, 3).

    Parameters:
    - p1: A set of points on line L1 (nframes, 3)
    - p2: A set of points on line L1 (nframes, 3)
    - p3: A set of points on line L2 (nframes, 3)
    - p4: A set of points on line L2 (nframes, 3)

    Returns:
    - distance: The shortest distance between the two lines for each frame
    - translation_vector: The translation vector for each frame to move L1 to the shortest point on L2
    """
    # Ensure the input points have the correct shape (nframes, 3)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)

    # Direction vectors of the lines L1 and L2
    d1 = p2 - p1  # Direction vector of L1 (nframes, 3)
    d2 = p4 - p3  # Direction vector of L2 (nframes, 3)

    # Vector from p1 on L1 to p3 on L2 (nframes, 3)
    ap = p3 - p1

    # Cross product of the direction vectors d1 and d2 (nframes, 3)
    cross_product = np.cross(d1, d2, axis=1)

    # If the cross product is zero, the lines are parallel
    cross_norm = np.linalg.norm(cross_product, axis=1)
    parallel_mask = cross_norm == 0

    # Shortest distance between the two skew lines (nframes,)
    distance = np.zeros(len(p1))

    # Calculate distance for frames where the lines are not parallel
    distance[~parallel_mask] = np.abs(np.sum(ap * cross_product, axis=1) / cross_norm[~parallel_mask])

    # Calculate translation vector to move L1 to the shortest point on L2 (nframes, 3)
    translation_vector = np.zeros_like(p1)
    translation_vector[~parallel_mask] = (np.sum(ap * cross_product, axis=1) / (cross_norm[~parallel_mask] ** 2))[:,
                                         None] * cross_product[~parallel_mask]

    return distance, translation_vector


def get_string_end_points(cp_info, instru_11):
    """
    get cp coordinates from the cp info
    :param cp_info: (batch_size, 2)  [activation_string, vibrating_length]
    :param instru_11: (batch_size, 11, 3)
    :return: start_points, end_points
    """

    start_points = np.zeros((cp_info.shape[0], 3))
    end_points = np.zeros((cp_info.shape[0], 3))

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
            start_points[i] = np.array([np.nan, np.nan, np.nan])
            end_points[i] = np.array([np.nan, np.nan, np.nan])
            continue
        start_idx, end_idx = string_points[string_num]
        start_points[i] = instru_11[i, start_idx]
        end_points[i] = instru_11[i, end_idx]

    return start_points, end_points


def linear_interpolate(arr):
    isnan = np.isnan(arr)
    nan_indices = np.flatnonzero(isnan)
    not_nan_indices = np.flatnonzero(~isnan)
    not_nan_values = arr[~isnan]
    arr[nan_indices] = np.interp(nan_indices, not_nan_indices, not_nan_values)
    return arr


# 计算两条线段之间的最短距离（用于优化目标）
def segment_distance(B_prime, A, P, Q):
    # 计算直线 L1 的方向向量 v1 和直线 L2 的方向向量 v2
    v1 = B_prime - A
    v2 = Q - P

    # 计算向量 A1 到 B1 的向量 diff
    diff = Q - A

    # 计算 v1 和 v2 的叉积
    cross_product = np.cross(v1, v2)

    # 计算叉积的模长
    cross_product_norm = np.linalg.norm(cross_product)

    # 计算点积 diff · (v1 x v2)
    dot_product = np.dot(diff, cross_product)

    # 最短距离公式
    distance = np.abs(dot_product) / cross_product_norm
    return distance


# 目标函数：最小化旋转后的点 B' 与 PQ 线段的最短距离
def objective(R, B, A, P, Q):
    B_prime = np.dot(R, B)  # 对 B 进行旋转
    return segment_distance(B_prime, A, P, Q)


# 旋转矩阵从欧拉角或四元数构造（示例中使用欧拉角）
def euler_to_rotation_matrix(euler_angles):
    pitch, yaw, roll = euler_angles
    # 欧拉角到旋转矩阵的转换（绕 Z，Y，X 轴旋转）
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    return np.dot(Rz, np.dot(Ry, Rx))

# 主优化过程
def rotate_segment(A, B, P, Q):
    # 初始旋转矩阵（单位矩阵表示无旋转）
    initial_angles = np.array([0.0, 0.0, 0.0])

    # 优化目标函数
    result = minimize(
        lambda angles: objective(euler_to_rotation_matrix(angles), B, A, P, Q),
        initial_angles,
        method='Nelder-Mead'
    )

    # 得到优化后的旋转矩阵
    optimal_rotation = euler_to_rotation_matrix(result.x)

    # 计算旋转后的 B 点
    B_prime = np.dot(optimal_rotation, B)
    B_trans = B_prime - B

    return B_trans


def interpolate_rotation_quaternions(quaternions, B):
    nframes = quaternions.shape[0]
    rot_matrices = np.zeros((nframes, 3, 3))
    B_rotated = B.copy()

    # 对缺失的旋转矩阵进行插值
    for i in range(1, nframes - 1):
        if quaternions[i] is None:  # 如果当前帧是空的
            # 找到前后的有效旋转矩阵
            prev_idx = i - 1
            next_idx = i + 1
            while prev_idx > 0 and (quaternions[prev_idx] is None or np.linalg.norm(quaternions[prev_idx]) == 0):
                prev_idx -= 1
            while next_idx < nframes - 1 and (
                    quaternions[next_idx] is None or np.linalg.norm(quaternions[next_idx]) == 0):
                next_idx += 1

            if prev_idx == 0 and next_idx == nframes - 1:
                raise ValueError("Cannot interpolate at the ends if both are None.")

            # 对四元数进行SLERP插值
            q1 = R.from_quat(quaternions[prev_idx])
            q2 = R.from_quat(quaternions[next_idx])
            t = (i - prev_idx) / (next_idx - prev_idx)  # 计算插值比例

            rots = R.from_quat(np.vstack([q1.as_quat(), q2.as_quat()]))
            # 使用Slerp进行插值
            slerp = Slerp([prev_idx, next_idx], rots)
            q_interpolated = slerp(t)  # 获取插值后的四元数
            quaternions[i] = q_interpolated.as_quat()  # 存储插值后的四元数

    ic(quaternions.shape)
    quaternions = smooth_quaternions(quaternions)
    # 将插值后的四元数转换回旋转矩阵
    for i in range(nframes):
        if quaternions[i] is not None and np.linalg.norm(quaternions[i]) == 0:
            rot_matrices[i] = R.from_quat(quaternions[i]).as_matrix()
            B_rotated[i] = np.dot(rot_matrices[i], B[i])
    return B_rotated


def smooth_quaternions(rot_quat):
    dot_products = np.einsum('ij,ij->i', rot_quat[1:], rot_quat[:-1])
    flip_mask = (np.cumsum(dot_products < 0, axis=0) % 2).astype(bool)
    rot_quat[1:][flip_mask] *= -1
    smoothed_quat = signal.savgol_filter(rot_quat, window_length=11, polyorder=9, axis=0, mode='interp')

    norms = np.linalg.norm(smoothed_quat, axis=-1)
    smoothed_quat = smoothed_quat / norms[:, np.newaxis]

    return smoothed_quat


# 处理多个帧数据
def get_rotated_bow_tip(A, B, P, Q):
    nframes = A.shape[0]
    B_trans = np.zeros((nframes, 3))

    for i in range(nframes):
        # 检查 P 和 Q 是否包含 NaN
        if np.any(np.isnan(P[i])) or np.any(np.isnan(Q[i])):
            # 如果 P 或 Q 包含 NaN，则保留原始 B 点
            B_trans[i] = [np.nan, np.nan, np.nan]
        else:
            # 否则，进行优化并旋转 B 点
            B_trans[i] = rotate_segment(A[i], B[i], P[i], Q[i])

    return B_trans


from scipy.ndimage import gaussian_filter1d


def smooth_directions_gaussian(vectors, sigma=1):
    # 高斯滤波（每个分量分别滤波）
    smoothed_vectors = np.zeros_like(vectors)
    for i in range(3):  # 对每个分量进行滤波
        smoothed_vectors[:, i] = gaussian_filter1d(vectors[:, i], sigma=sigma)

    # 重新归一化向量（保证单位向量）
    smoothed_vectors = smoothed_vectors / np.linalg.norm(smoothed_vectors, axis=1, keepdims=True)

    return smoothed_vectors


if __name__ == "__main__":
    print("In Progress ...")

    # get cello data from SPD
    kp3d_dir_path = "../dataset/SPD/cello/kp3d/"  # the of the directory containing all the 3d kp files (.npy)
    rot_vec_dir_path = "../dataset/SPD/cello/hand_rot/"
    save_dir = "./train_data"
    data_manip = DataManipulatorCello()
    smooth_rot = data_manip.smooth_rot
    data_path_list = find_npy_files(kp3d_dir_path)[:2]
    ic(data_path_list)

    if not data_path_list:
        raise FileNotFoundError('Please follow the instrution to download the SPD and put them in the right place!')

    reference_cello = np.load('instrument/instrument.npy')

    data_instrument = []
    data_bow = []
    data_contactpoint = []
    data_lh_trans = []
    data_lh_pose_R = []
    data_rh_trans = []
    data_rh_pose_R = []
    data_cp_info = []
    data_body_coco = []
    data_used_finger_idx = []

    frame_num_recording = [0]  # flat all the frame in all the pieces, but record the starting nodes
    frame_pointer = 0
    cnt = 0
    piece_idx_list = []
    # loop all the data piece from cello01 -- cello85
    loop_data_path_list = tqdm(data_path_list, total=len(data_path_list), leave=True)
    for data_path in loop_data_path_list:
        data_piece = np.load(data_path)
        frame_pointer += len(data_piece)
        frame_num_recording.append(frame_pointer)

        # for the instrument recording in 3dkp format, set the "END PIN" as the original point
        data_norm_piece, data_lh_trans_piece, data_rh_trans_piece = normalize_keypoints_cello(data_piece)
        data_instrument_piece = data_norm_piece[:, CELLO_IDX_RANGE[0]:CELLO_IDX_RANGE[1], :]
        data_bow_piece = data_norm_piece[:, BOW_IDX_RANGE[0]:BOW_IDX_RANGE[1], :]
        data_contactpoint_piece = data_norm_piece[:, CONTACT_POINT_INDEX, :]
        data_body_coco_piece = data_norm_piece[:, BODY_IDX_RANGE[0]:BODY_IDX_RANGE[1], :]

        data_used_finger_tip_pos_piece = data_norm_piece[:, USED_FINGER_TIP_IDX, :]
        data_tips_pos_piece = data_norm_piece[:, FINGER_TIP_IDX, :]
        data_used_finger_idx_piece = get_used_finger_idx(data_used_finger_tip_pos_piece, data_tips_pos_piece)

        # for hand pose recording in rot vec (.json), match corresponding file with current 3dkp file (.npy)
        piece_idx = data_path.split('.')[-2][-2:]  # get current piece index, if current is cello01, then return '01'
        loop_data_path_list.set_description(f'Data Piece Loading [{piece_idx}/{len(data_path_list)}]')
        piece_idx_list.append(piece_idx)

        rot_file = os.path.join(rot_vec_dir_path + f"cello{piece_idx}/hand_rot_vec.json")
        with open(rot_file, 'r') as file:
            data_piece = np.array(json.load(file)['hand_rot_vec'])
            data_lh_pose_piece = data_piece[:, :16, :]  # left hand, the first 16
            data_rh_pose_piece = data_piece[:, 16:, :]  # right hand, the rest
        assert len(data_lh_pose_piece) == len(data_norm_piece), \
            f'pose: {len(data_lh_pose_piece)} != norm: {len(data_norm_piece)}'

        # loop all the frames in the current piece
        for frame_idx in range(len(data_norm_piece)):
            frame_instrument = data_instrument_piece[frame_idx]
            frame_bow = data_bow_piece[frame_idx]
            frame_contactpoint = data_contactpoint_piece[frame_idx]
            frame_lh_trans = data_lh_trans_piece[frame_idx].squeeze()
            frame_rh_trans = data_rh_trans_piece[frame_idx].squeeze()
            frame_lh_pose_rotvec = data_lh_pose_piece[frame_idx]
            frame_lh_pose_matrix = R.from_rotvec(frame_lh_pose_rotvec).as_matrix()
            frame_rh_pose_rotvec = data_rh_pose_piece[frame_idx]
            frame_rh_pose_matrix = R.from_rotvec(frame_rh_pose_rotvec).as_matrix()
            cp_info = find_cp_position(frame_contactpoint,
                                       frame_instrument)  # a list, [activation_string, vibrating_length]
            frame_body_coco = data_body_coco_piece[frame_idx]
            frame_used_finger_idx = data_used_finger_idx_piece[frame_idx]

            # convert from mm to m
            frame_instrument = frame_instrument / 1000.
            frame_bow = frame_bow / 1000.
            frame_contactpoint = frame_contactpoint / 1000.
            frame_lh_trans = frame_lh_trans / 1000.
            frame_rh_trans = frame_rh_trans / 1000.
            frame_body_coco = frame_body_coco / 1000.

            data_instrument.append(frame_instrument)
            data_bow.append(frame_bow)
            data_contactpoint.append(frame_contactpoint)
            data_cp_info.append(cp_info)
            data_lh_trans.append(frame_lh_trans)
            data_rh_trans.append(frame_rh_trans)
            data_lh_pose_R.append(frame_lh_pose_matrix)
            data_rh_pose_R.append(frame_rh_pose_matrix)
            data_body_coco.append(frame_body_coco)
            data_used_finger_idx.append(frame_used_finger_idx)

            # break
        # cnt += 1
        # if cnt >= 2:
        #     break

    data_instrument = np.array(data_instrument)  # (frame_num, 9, 3) or (frame_num, 7, 3)
    data_bow = np.array(data_bow)  # (frame_num, 2, 3)
    data_contactpoint = np.array(data_contactpoint)  # (frame_num, 3)
    data_cp_info = np.array(data_cp_info)  # (frame_num, 2)
    data_lh_trans = np.array(data_lh_trans)  # (frame_num, 3)
    data_lh_pose_R = np.array(data_lh_pose_R)  # (frame_num, 16, 3, 3)
    data_rh_trans = np.array(data_rh_trans)  # (frame_num, 3)
    data_rh_pose_R = np.array(data_rh_pose_R)  # (frame_num, 16, 3, 3)
    data_body_coco = np.array(data_body_coco)  # (frame_num, 23, 3)
    data_used_finger_idx = np.array(data_used_finger_idx)  # (frame_num,)

    # set the cello of the fist frame as the "stardard cello pose"
    # reference_cello = data_instrument[0]  # (7, 3)
    # aligning the cellos of all the frames, targeting to the stardard one
    aligned_mainpart_cellos, rotations = align_cello_keypoints(data_instrument, reference_cello)
    # aligned_mainpart_cellos, rotations = align_cello_keypoints(data_instrument[:, :-1, :], reference_cello[:-1, :])           # excluding the end pin
    # aligned_mainpart_cellos, rotations = align_cello_keypoints(data_instrument[:, :-3, :], reference_cello[:-3, :])         # if with bow, -3 for end pin
    # aligned_remaining_cellos = apply_rotations(data_instrument[:, -1:, :], rotations)
    # aligned_cellos = np.concatenate((aligned_mainpart_cellos, aligned_remaining_cellos), axis=1)

    # aligned_cellos = apply_rotations(data_instrument, rotations)
    # average_cello = compute_average_shape(aligned_cellos).reshape(7, 3)
    average_cello = reference_cello

    # ic(frame_num_recording)
    # ic(aligned_mainpart_cellos.shape)

    aligned_samples_cello = np.stack(
        [aligned_mainpart_cellos[idx] for idx in frame_num_recording[:-1]], axis=0
    )  # [piece_num, 4, 3]

    aligned_samples_cello_bridge_middle = (aligned_samples_cello[:, 2] + aligned_samples_cello[:, 3]) / 2
    # ic(aligned_samples_cello_bridge_middle.shape)
    aligned_samples_cello_trans = aligned_samples_cello_bridge_middle[0] - aligned_samples_cello_bridge_middle

    # aligning the bow, with the same rotation as the cello, in terms of the current frame
    aligned_bows = apply_rotations(data_bow, rotations)
    # aligning the body, with the same rotation as the cello, in terms of the current frame
    aligned_bodies = apply_rotations(data_body_coco, rotations)
    # aligning the whole hand, with the same rotation as the cello, in terms of the current frame
    # init_lh_positions = get_init_pose_batch(data_lh_trans, 'left')
    # original_lh_positions = get_joint_positions_batch(init_lh_positions, data_lh_pose_R, BONE_LENGTHS, MANO_PARENTS_INDICES).reshape(-1, 21, 3)
    # aligned_lh_positions = apply_rotations(original_lh_positions, rotations)   # whole hand rotation based on the tail gut
    # init_rh_positions = get_init_pose_batch(data_rh_trans, 'right')
    # original_rh_positions = get_joint_positions_batch(init_rh_positions, data_rh_pose_R, BONE_LENGTHS, MANO_PARENTS_INDICES).reshape(-1, 21, 3)
    # aligned_rh_positions = apply_rotations(original_rh_positions, rotations)

    # replacing the "hand trans" by the "hand trans new" which is the trans after cello aligning (taking tail gut as the original point)
    lh_trans_new, lh_wrist_R_new = batch_calculate_wrist_displacement_and_rotation(data_lh_trans, data_lh_pose_R[:, 0],
                                                                                   rotations)
    # replacing the wirst rotation, the new rotation combining the former self-rotation and the global-rotation towards the tail gut
    lh_pose_R_new = data_lh_pose_R.copy()
    lh_pose_R_new[:, 0] = lh_wrist_R_new
    # init_lh_positions_new = get_init_pose_batch(lh_trans_new, 'left')
    # aligned_lh_positions_new = get_joint_positions_batch(init_lh_positions_new, lh_pose_R_new, BONE_LENGTHS, MANO_PARENTS_INDICES).reshape(-1, 21, 3)
    # as same as the previous "aligned_hand_position", for validating the correctness of the process

    rh_trans_new, rh_wrist_R_new = batch_calculate_wrist_displacement_and_rotation(data_rh_trans, data_rh_pose_R[:, 0],
                                                                                   rotations)
    rh_pose_R_new = data_rh_pose_R.copy()
    rh_pose_R_new[:, 0] = rh_wrist_R_new
    # init_rh_positions_new = get_init_pose_batch(rh_trans_new, 'right')
    # aligned_rh_positions_new = get_joint_positions_batch(init_rh_positions_new, rh_pose_R_new, BONE_LENGTHS, MANO_PARENTS_INDICES).reshape(-1, 21, 3)
    # as same as the previous "aligned_hand_position", for validating the correctness of the process

    # plot things, for checking
    # samples_idx = [0, 800, 2000]
    #
    # original_instrument_all = np.concatenate((data_instrument, data_bow), axis=1)
    # aligned_instrument_all = np.concatenate((aligned_cellos, aligned_bows), axis=1)
    #
    # original_samples_cello = np.stack(
    #     (original_instrument_all[samples_idx[0]], original_instrument_all[samples_idx[1]], original_instrument_all[samples_idx[2]]), axis=0)
    # aligned_samples_cello = np.stack(
    #     (aligned_instrument_all[samples_idx[0]], aligned_instrument_all[samples_idx[1]], aligned_instrument_all[samples_idx[2]]), axis=0)
    #
    # original_samples_hand = np.stack((original_hand_position[samples_idx[0]], original_hand_position[samples_idx[1]],
    #                                  original_hand_position[samples_idx[2]]), axis=0)
    # aligned_samples_hand = np.stack((aligned_hand_position[samples_idx[0]], aligned_hand_position[samples_idx[1]],
    #                                  aligned_hand_position[samples_idx[2]]), axis=0)
    # aligned_samples_hand_new = np.stack((aligned_hand_position_new[samples_idx[0]], aligned_hand_position_new[samples_idx[1]],
    #                                  aligned_hand_position_new[samples_idx[2]]),axis=0)

    # plot_cellos_and_hands(aligned_samples_cello, aligned_samples_hand)
    # plot_cellos_and_hands(aligned_samples_cello, aligned_samples_hand_new)

    # plot_cellos_and_hands(original_samples_cello, original_samples_hand)
    # plot_cellos_and_hands(aligned_samples_cello, aligned_samples_hand_new)

    # creat training data
    data_bow = aligned_bows
    data_body_coco = aligned_bodies

    data_lh_trans = lh_trans_new
    data_lh_pose_R = lh_pose_R_new
    data_lh_pose_R_flat = data_lh_pose_R.reshape(-1, 3, 3)
    data_lh_pose_6d = data_lh_pose_R_flat[:, :2, :].reshape(len(data_lh_pose_R), data_lh_pose_R.shape[1] * 6)

    data_rh_trans = rh_trans_new
    data_rh_pose_R = rh_pose_R_new
    data_rh_pose_R_flat = data_rh_pose_R.reshape(-1, 3, 3)
    data_rh_pose_6d = data_rh_pose_R_flat[:, :2, :].reshape(len(data_rh_pose_R), data_rh_pose_R.shape[1] * 6)

    loop_train_gen = tqdm(range(1, len(frame_num_recording)))
    for piece_i in loop_train_gen:
        loop_train_gen.set_description(f'Training Piece Generating [{piece_i}/{len(frame_num_recording) - 1}]')
        piece_start = frame_num_recording[piece_i - 1]
        piece_end = frame_num_recording[piece_i]

        piece_len = piece_end - piece_start

        rotation_piece_base = rotations[piece_start]
        rotation_target_set = rotations[piece_start: piece_end]
        rotation_change_set = calculate_rotation_change(rotation_piece_base, rotation_target_set)

        seq_bow = data_bow[piece_start: piece_end]
        seq_body = data_body_coco[piece_start: piece_end]
        seq_lh_trans = data_lh_trans[piece_start: piece_end]
        seq_lh_pose = data_lh_pose_6d[piece_start: piece_end]
        seq_rh_trans = data_rh_trans[piece_start: piece_end]
        seq_rh_pose = data_rh_pose_6d[piece_start: piece_end]
        seq_rotation = rotations[piece_start: piece_end]
        seq_rotation_6d = seq_rotation[:, :2, :].reshape(len(seq_rotation), 6)
        seq_rotation_withinpiece_6d = rotation_change_set[:, :2, :].reshape(len(rotation_change_set), 6)
        # seq_contactpoint = data_contactpoint[piece_start: piece_end]
        seq_cp_info = data_cp_info[piece_start: piece_end]

        seq_lh_pose_6d = seq_lh_pose.reshape(-1, 6)
        seq_lh_pose_6d = replace_zero_6d(seq_lh_pose_6d)

        seq_lh_pose_matrix = rotation_6d_to_R(seq_lh_pose_6d).reshape(-1, 3, 3)
        seq_lh_pose_matrix = smooth_rot(seq_lh_pose_matrix)
        seq_lh_pose_matrix_flat = seq_lh_pose_matrix.reshape(-1, 3, 3)
        seq_lh_pose_6d = seq_lh_pose_matrix_flat[:, :2, :].reshape(seq_lh_pose_matrix.shape[0],
                                                                   seq_lh_pose_matrix.shape[1], 6)
        seq_lh_pose = seq_lh_pose_6d.reshape(seq_lh_pose.shape)

        seq_rh_pose_6d = seq_rh_pose.reshape(-1, 6)
        seq_rh_pose_6d = replace_zero_6d(seq_rh_pose_6d)

        # seq_rh_pose_matrix = rotation_6d_to_R(seq_rh_pose_6d).reshape(-1, 3, 3)
        # seq_rh_pose_matrix = smooth_rot(seq_rh_pose_matrix, window_length=91, polyorder=5)
        # seq_rh_pose_matrix_flat = seq_rh_pose_matrix.reshape(-1, 3, 3)
        # seq_rh_pose_6d = seq_rh_pose_matrix_flat[:, :2, :].reshape(seq_rh_pose_matrix.shape[0],
        #                                                            seq_rh_pose_matrix.shape[1], 6)

        seq_rh_pose_matrix = rotation_6d_to_R(seq_rh_pose_6d).reshape(-1, 16, 3, 3)
        seq_rh_wrist_pose_matrix = seq_rh_pose_matrix[:, :1]
        seq_rh_without_wrist_pose_matrix = seq_rh_pose_matrix[:, 1:]
        seq_rh_wrist_pose_matrix = smooth_rot(seq_rh_wrist_pose_matrix, njoints=1, window_length=19, polyorder=6)
        seq_rh_without_wrist_pose_matrix = smooth_rot(seq_rh_without_wrist_pose_matrix, njoints=15, window_length=91,
                                                      polyorder=6)
        seq_rh_pose_matrix = np.concatenate((seq_rh_wrist_pose_matrix, seq_rh_without_wrist_pose_matrix), axis=1)
        seq_rh_pose_matrix_flat = seq_rh_pose_matrix.reshape(-1, 3, 3)
        seq_rh_pose_6d = seq_rh_pose_matrix_flat[:, :2, :].reshape(seq_rh_pose_matrix.shape[0],
                                                                   seq_rh_pose_matrix.shape[1], 6)

        seq_rh_pose = seq_rh_pose_6d.reshape(seq_rh_pose.shape)

        seq_rotation_withinpiece = rotation_6d_to_R(seq_rotation_withinpiece_6d)
        seq_instrument = average_cello[np.newaxis, :, :].repeat(piece_len, axis=0)
        # seq_instrument = apply_rotations(seq_instrument, seq_rotation_withinpiece)

        seq_rh_trans = signal.savgol_filter(seq_rh_trans, window_length=19, polyorder=5, axis=0, mode='interp')
        seq_rh_trans += aligned_samples_cello_trans[piece_i - 1]

        seq_body += aligned_samples_cello_trans[piece_i - 1]

        seq_bow += aligned_samples_cello_trans[piece_i - 1]
        seq_bow_start = seq_bow[:, 0].copy()
        seq_bow_end = seq_bow[:, 1].copy()

        seq_bow_start = signal.savgol_filter(seq_bow_start, window_length=91, polyorder=5, axis=0, mode='interp')
        seq_bow_end = signal.savgol_filter(seq_bow_end, window_length=119, polyorder=5, axis=0, mode='interp')
        seq_bow_vec = normalize_vector_batch(seq_bow_end - seq_bow_start)

        seq_rh_position = data_manip.rot2position_hand(seq_rh_trans, seq_rh_pose, hand_type='right')

        # seq_elbow = seq_body[:, 8]
        # seq_new_bow_start = seq_elbow + (seq_rh_trans - seq_elbow) * 1.1

        hand_joints_bow = [5, 6, 11, 12, 14, 15]
        seq_new_bow_start = np.mean(seq_rh_position[:, hand_joints_bow], axis=1)
        seq_new_bow_end = seq_new_bow_start + seq_bow_vec * 0.8

        # seq_bow_start = seq_bow[:, 0]
        # seq_bow_end = seq_bow[:, 1]
        # bow_trans = seq_new_bow_start - seq_bow_start
        # seq_new_bow_end = seq_bow_end + bow_trans
        # seq_bow[:, 0] = seq_new_bow_start
        # seq_bow[:, 1] = seq_new_bow_end
        # seq_bow_vec = normalize_vector_batch(seq_new_bow_end - seq_new_bow_start)

        # seq_bow_vec = smooth_directions_gaussian(seq_bow_vec, sigma=50)

        # seq_bow_vec = seq_new_bow_end - seq_new_bow_start
        # ic(seq_bow_vec.shape)
        string_start, string_end = get_string_end_points(seq_cp_info, seq_instrument)
        # ic(string_start.shape)
        # ic(string_end.shape)

        seq_bow_vec = normalize_vector_batch(seq_new_bow_end - seq_new_bow_start)
        # ic(seq_bow_vec.shape)

        # seq_bow_end = seq_bow_end + seq_bow_end_trans
        # seq_bow_vec = normalize_vector_batch(seq_bow_end - seq_bow_start)

        # seq_bow[:, 0] = seq_bow_start
        # seq_bow[:, 1] = seq_bow_vec
        # ic(seq_new_bow_end.shape)
        # ic(seq_new_bow_end[0])

        seq_used_finger_idx = data_used_finger_idx[piece_start: piece_end]
        seq_lh_position = data_manip.rot2position_hand(seq_lh_trans, seq_lh_pose, hand_type='left')
        tip2mano = {-1: -1, 0: 16, 1: 17, 2: 19, 3: 18}
        seq_used_finger_tip_idx = np.vectorize(tip2mano.get)(seq_used_finger_idx)
        seq_used_finger_tip_pos = np.full((seq_used_finger_tip_idx.shape[0], 3), np.nan)
        valid_indices = seq_used_finger_tip_idx != -1
        seq_used_finger_tip_pos[valid_indices] = seq_lh_position[valid_indices, seq_used_finger_tip_idx[valid_indices]]
        # ic(seq_used_finger_tip_pos)

        # seq_instrument = data_manip.get_second_third_string(seq_instrument)
        seq_cp_pos = data_manip.calculate_cp_coordinates(seq_cp_info, seq_instrument)
        seq_lh_trans_amend = seq_cp_pos - seq_used_finger_tip_pos

        # seq_lh_trans_amend = np.nan_to_num(seq_lh_trans_amend, nan=0)

        # ic(seq_cp_info)
        # ic((np.isnan(seq_cp_pos))[30:100])
        # ic(seq_cp_pos[30:100])

        # from scipy.interpolate import interp1d, CubicSpline
        # def spline_interpolate(arr):
        #     x = np.arange(len(arr))
        #     mask = ~np.isnan(arr)
        #     spline = interp1d(x[mask], arr[mask], kind='cubic', fill_value='extrapolate')
        #     arr[~mask] = spline(x[~mask])
        #     return arr
        #
        # def cubic_spline_interpolation(arr):
        #     x = np.arange(len(arr))
        #     mask = ~np.isnan(arr)
        #     cs = CubicSpline(x[mask], arr[mask])
        #     arr[~mask] = cs(x[~mask])
        #     return arr

        seq_lh_trans_amend = np.apply_along_axis(linear_interpolate, axis=0, arr=seq_lh_trans_amend)
        seq_lh_trans_amend = signal.savgol_filter(seq_lh_trans_amend, window_length=27, polyorder=5, axis=0,
                                                  mode='interp')
        seq_lh_trans = seq_lh_trans + seq_lh_trans_amend

        seq_body[:, 9] = seq_lh_trans
        seq_body[:, 10] = seq_rh_trans

        # ic(seq_bow.shape)
        # ic(seq_lh_trans.shape)
        # ic(seq_lh_pose.shape)
        # ic(seq_rotation_6d.shape)
        # ic(seq_rotation_withinpiece_6d.shape)
        # ic(seq_contactpoint.shape)
        # ic(seq_cp_info.shape)
        # ic(seq_instrument.shape)

        # audio_path = f'wavs_cello/cello{piece_idx_list[piece_i - 1]}.wav'
        # audio_feats = extract_dac_feats(audio_path, (SEQ_LEN / FRAME_RATE), (HOP_LEN / FRAME_RATE))
        # audio_feats = extract_encodec_feats(audio_path, (SEQ_LEN / FRAME_RATE), (HOP_LEN / FRAME_RATE))
        # audio_feats = extract_jukebox_feats(audio_path, piece_len)

        # assert len(audio_feats) == len(seq_lh_pose), \
        #             f'audio: {len(audio_feats)} != pose: {len(seq_lh_pose)}'

        data_all = {
            'lh_pose': seq_lh_pose.tolist(),
            'lh_trans': seq_lh_trans.tolist(),
            'rh_pose': seq_rh_pose.tolist(),
            'rh_trans': seq_rh_trans.tolist(),
            # 'audio': audio_feats.tolist(),
            'length': piece_len,
            'bow': seq_bow_vec.tolist(),
            # 'bow': seq_bow.tolist(),
            'body': seq_body.tolist(),
            # 'instrument': seq_instrument,
            'instrument': average_cello.tolist(),
            'cp_info': seq_cp_info.tolist(),
            # 'cp_pos' : seq_cp_pos.tolist(),
            # 'used_finger_tip_idx': seq_used_finger_tip_idx.tolist(),
            'used_finger_idx': seq_used_finger_idx.tolist()
            # 'rotation': seq_rotation_withinpiece_6d.tolist()
        }

        save_sub_dir = os.path.join(save_dir, 'wholebody_normalized')
        os.makedirs(save_sub_dir, exist_ok=True)
        save_path = os.path.join(save_sub_dir, f'cello{piece_idx_list[piece_i - 1]}.json')
        with open(save_path, 'w') as file:
            json.dump(data_all, file, indent=4)
