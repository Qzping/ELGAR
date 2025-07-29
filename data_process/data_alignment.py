import numpy as np
import json
import os
from data_process.data_manipulation import DataManipulatorCello
from scipy.spatial.transform import Rotation as R
from icecream import ic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def angle_between_vectors_angle(v1_start, v1_end, v2_start, v2_end):

    v1 = v1_end - v1_start
    v2 = v2_end - v2_start

    dot_product = np.dot(v1, v2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (norm_v1 * norm_v2)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_radians = np.arccos(cos_theta)

    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def visualize_cello(points, rotated_points, instrument_links):
    """
    This function visualizes the original and rotated positions of the cello,
    as well as the coordinate axes.

    Args:
    - points (numpy array): The original 7 key points of the cello.
    - rotated_points (numpy array): The rotated 7 key points of the cello.
    - instrument_links (list): List of pairs of indices indicating which points are connected.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X axis')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z axis')

    for link in instrument_links:
        ax.plot([points[link[0], 0], points[link[1], 0]],
                [points[link[0], 1], points[link[1], 1]],
                [points[link[0], 2], points[link[1], 2]], 'r')

    for link in instrument_links:
        ax.plot([rotated_points[link[0], 0], rotated_points[link[1], 0]],
                [rotated_points[link[0], 1], rotated_points[link[1], 1]],
                [rotated_points[link[0], 2], rotated_points[link[1], 2]], 'b')

    for i, point in enumerate(rotated_points):
        ax.text(point[0], point[1], point[2], f'P{i}', color='black')

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()


def visualize_cello_and_hand(cello_points, rotated_cello_points, cello_links, hand_points, rotated_hand_points,
                             hand_links):
    """
    This function visualizes the original and rotated positions of both the cello and the hand,
    as well as the coordinate axes.

    Args:
    - cello_points (numpy array): The original 7 key points of the cello.
    - rotated_cello_points (numpy array): The rotated 7 key points of the cello.
    - cello_links (list): List of pairs of indices indicating which points are connected for the cello.
    - hand_points (numpy array): The original 21 key points of the hand.
    - rotated_hand_points (numpy array): The rotated 21 key points of the hand.
    - hand_links (list): List of pairs of indices indicating which points are connected for the hand.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X axis')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z axis')

    for link in cello_links:
        ax.plot([cello_points[link[0], 0], cello_points[link[1], 0]],
                [cello_points[link[0], 1], cello_points[link[1], 1]],
                [cello_points[link[0], 2], cello_points[link[1], 2]], 'r')

    for link in cello_links:
        ax.plot([rotated_cello_points[link[0], 0], rotated_cello_points[link[1], 0]],
                [rotated_cello_points[link[0], 1], rotated_cello_points[link[1], 1]],
                [rotated_cello_points[link[0], 2], rotated_cello_points[link[1], 2]], 'b')

    for link in hand_links:
        ax.plot([hand_points[link[0], 0], hand_points[link[1], 0]],
                [hand_points[link[0], 1], hand_points[link[1], 1]],
                [hand_points[link[0], 2], hand_points[link[1], 2]], 'r')

    for link in hand_links:
        ax.plot([rotated_hand_points[link[0], 0], rotated_hand_points[link[1], 0]],
                [rotated_hand_points[link[0], 1], rotated_hand_points[link[1], 1]],
                [rotated_hand_points[link[0], 2], rotated_hand_points[link[1], 2]], 'b')

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()



# R1, rotate around point G, adjust the FG direction so that parallel to the yz plane
def rotate_FG_to_yz_plane(points):
    """
    This function rotates the points of the cello about the fixed point G (origin) such that
    the tailpiece (FG) aligns with the yz-plane.

    Args:
    - points (numpy array): The 7 key points of the cello. G is assumed to be the last point (fixed at the origin).

    Returns:
    - numpy array: The rotated points of the cello.
    """

    F = points[5]  #  tailgut
    G = points[6]  #  endpin
    FG = F - G

    # Calculate the angle between FG and the yz plane, with the goal of making FG parallel to the yz plane
    # Calculate the rotation angle so that the FG along x axis is 0.
    angle = np.arctan2(FG[0], FG[2])  # Calculate the angle between FG and the yz plane, and rotate around the z axis
    rotation_matrix = rotation_matrix_around_z(angle)

    # Rotate FG around the z-axis so that it is parallel to the yz plane
    points_rotated = points - G  # Translate G to the origin
    points_rotated = np.dot(points_rotated, rotation_matrix.T)  # rotate
    points_rotated += G  # Translate back to G
    return points_rotated, rotation_matrix

# R2, rotate around the x-axis so that FG forms a specific angle with the y-axis.
def rotate_about_x_to_targe_tangle(points):
    """
    rotates the cello points such that the tailpiece FG makes a target angle with the y-axis
    by performing a rigid body rotation around the x-axis.

    Args:
    - points (numpy array): The 7 key points of the cello. G is assumed to be the last point (fixed at the origin).

    Returns:
    - numpy array: The rotated points of the cello.
    """
    # G point is the origin, FG as the direction vector.
    F = points[5]  # F: tailgut
    G = points[6]  # G: endpin
    FG = F - G

    current_angle = np.arctan2(FG[0], FG[2])  # Angle between FG and the y-axis (angle of rotation around the x-axis)

    # Target angle: 40 degrees
    target_angle = 40

    angle_to_rotate = target_angle - np.rad2deg(current_angle)

    rotation_matrix = rotation_matrix_around_x(angle_to_rotate)

    # Rotate all points around the x-axis
    points_rotated = points - G  # Translate G to the origin
    points_rotated = np.dot(points_rotated, rotation_matrix.T)  # apply rotations
    points_rotated += G  # Translate back to G

    return points_rotated, rotation_matrix

# R3, rotate cello around FG
def rotate_about_FG_to_target_orientation(points):
    """
    Rotate the cello around the FG axis so that the plane containing B, C, and D is perpendicular to the yz plane.

    params:
    - points: (7, 3), representing 7 key points of cello, order in A, B, C, D, E, F, G

    return:
    - rotated_points: positions of the seven key points on the cello after rotation。
    """
    G = points[6]  # G as origin
    F = points[5]
    B = points[1]
    C = points[2]
    D = points[3]

    FG = F - G

    BC = C - B
    BD = D - B

    # Calculate the normal vector of vector BC and BD.
    normal = np.cross(BC, BD)

    yz_normal = np.array([0, 1, 0])  # Normal vector of yz plane is [0, 1, 0]

    # Calculate the angle between the normal vector and the normal vector of the yz plane.
    angle = angle_between_vectors(normal, yz_normal)

    axis = FG / np.linalg.norm(FG)  # normalize FG vector

    # Calculating rotation matrices using the Rodrigues formula
    rotation = R.from_rotvec(axis * angle)  # Rotation axis and rotation angle

    # Rotate all points
    rotated_points = points - G
    rotated_points = rotation.apply(rotated_points)
    rotated_points += G

    return rotated_points, rotation.as_matrix()

def rotation_matrix_around_z(angle):
    """
    This function returns the rotation matrix for a given angle around the z-axis.

    Args:
    - angle (float): The angle by which to rotate.

    Returns:
    - numpy array: The rotation matrix.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return rotation_matrix

def rotation_matrix_around_x(angle):
    """
    returns the rotation matrix for a given angle around the x-axis.

    Args:
    - angle (float): The angle by which to rotate (in degrees).

    Returns:
    - numpy array: The rotation matrix.
    """
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])
    return rotation_matrix


def angle_between_vectors(v1, v2):

    dot_product = np.dot(v1, v2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (norm_v1 * norm_v2)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_radians = np.arccos(cos_theta)

    return angle_radians


def apply_rotations_to_points(points, rotations):

    rotated_points = points.copy()

    for rotation in rotations:
        rotated_points = np.dot(rotated_points, rotation.T)

    return rotated_points


def R_of_orientating(instrument_frame1):

    # R1
    instrument_frame1_rotated, R1 = rotate_FG_to_yz_plane(instrument_frame1)
    # R2
    instrument_frame1_rotated, R2 = rotate_about_x_to_targe_tangle(instrument_frame1_rotated)
    # R3
    instrument_frame1_rotated, R3 = rotate_about_FG_to_target_orientation(instrument_frame1_rotated)

    R_s = [R1, R2, R3]

    return R_s, instrument_frame1_rotated
