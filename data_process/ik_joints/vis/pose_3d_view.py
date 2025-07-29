import json
import os
import cv2
import numpy as np
from icecream import ic
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


HUMAN_LINKS_SMPL = [
                    #body
                    [0,1],
                    [0,2],
                    [0,3],
                    [1,4],
                    [2,5],
                    [3,6],
                    [4,7],
                    [5,8],
                    [6,9],
                    [7,10],
                    [8,11],
                    [9,12],
                    [9,13],
                    [9,14],
                    [12,15],
                    [13,16],
                    [14,17],
                    [16,18],
                    [17,19],
                    [18,20],
                    [19,21],
                    

                    #left hand
                    #left index
                    [20,25],
                    [25,26],
                    [26,27],
                    [27,67],

                    #left middle
                    [20,28],
                    [28,29],
                    [29,30],
                    [30,68],

                    #left pinky
                    [20,31],
                    [31,32],
                    [32,33],
                    [33,70],

                    #left ring
                    [20,34],
                    [34,35],
                    [35,36],
                    [36,69],
                    
                    #left thumb
                    [20,37],
                    [37,38],
                    [38,39],
                    [39,66],


                    #right hand
                    #right index
                    [21,40],
                    [40,41],
                    [41,42],
                    [42,72],

                    #right middle
                    [21,43],
                    [43,44],
                    [44,45],
                    [45,73],

                    #right pinky
                    [21,46],
                    [46,47],
                    [47,48],
                    [48,75],

                    #right ring
                    [21,49],
                    [49,50],
                    [50,51],
                    [51,74],
                    
                    #right thumb
                    [21,52],
                    [52,53],
                    [53,54],
                    [54,71],  
                    ]

HUMAN_LINKS_COCOwholebody =[[15, 13],
                            [13, 11],
                            [16, 14],
                            [14, 12],
                            [11, 12],
                            [5, 11],
                            [6, 12],
                            [5, 6],
                            [5, 7],
                            [6, 8],
                            [7, 9],
                            [8, 10],
                            [1, 2],
                            [0, 1],
                            [0, 2],
                            [1, 3],
                            [2, 4],
                            [3, 5],
                            [4, 6],
                            [15, 17],
                            [15, 18],
                            [15, 19],
                            [16, 20],
                            [16, 21],
                            [16, 22],

                            # use the index of coco but the mano link
                            [ 91,  92],
                            [ 92,  93],
                            [ 93,  94],
                            [ 91,  95],
                            [ 95,  96],
                            [ 96,  97],
                            [ 91,  98],
                            [ 98,  99],
                            [ 99, 100],
                            [ 91, 101],
                            [101, 102],
                            [102, 103],
                            [ 91, 104],
                            [104, 105],
                            [105, 106],
                            [ 94, 107],
                            [ 97, 108],
                            [100, 109],
                            [103, 110],
                            [106, 111],
                            
                            [112, 113],
                            [113, 114],
                            [114, 115],
                            [112, 116],
                            [116, 117],
                            [117, 118],
                            [112, 119],
                            [119, 120],
                            [120, 121],
                            [112, 122],
                            [122, 123],
                            [123, 124],
                            [112, 125],
                            [125, 126],
                            [126, 127],
                            [115, 128],
                            [118, 129],
                            [121, 130],
                            [124, 131],
                            [127, 132]]


def compute_axis_lim(triangulated_points, scale_factor=1):
    # triangulated_points in shape [num_frame, num_keypoint, 3 axis]
    xlim, ylim, zlim = None, None, None
    minmax = np.nanpercentile(triangulated_points, q=[0, 100], axis=0).T
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


def add_comment2imagearray(imagearray,dataset_name,frames,color=(255,255,255),position = 'top'):
    if position == 'top':
        imagearray = cv2.putText(imagearray, dataset_name, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
        imagearray = cv2.putText(imagearray, str(frames), (imagearray.shape[1]-150*len(str(frames+1)), 150), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 5)
    else:
        imagearray = cv2.putText(imagearray, dataset_name, (150, imagearray.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
        imagearray = cv2.putText(imagearray, str(frames), (imagearray.shape[1]-150*len(str(frames+1)), imagearray.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 5)
    return imagearray


def get_plot_from_pose(plot_info):
    if plot_info['xyz']['xlim'] is None:
        plot_info['xyz']['xlim'],plot_info['xyz']['ylim'],plot_info['xyz']['zlim'] = compute_axis_lim(
                                                                                                    np.concatenate(list(plot_info['kp_3d'].values())
                                                                                                    , axis= 0)
                                                                                                    , scale_factor=1.3)
    
    plot_facecolor = 'white'
    fig = plt.figure(facecolor=plot_facecolor, figsize=[23.00, 26.56])
    
    axes3 = fig.add_subplot(projection="3d", computed_zorder=False)
    axes3.set_box_aspect((0.1, 0.1, 3))
    axes3.set_facecolor(plot_facecolor)
    axes3.set_xlim3d(plot_info['xyz']['xlim'])
    axes3.set_ylim3d(plot_info['xyz']['ylim'])
    axes3.set_zlim3d(plot_info['xyz']['zlim'])
    axes3.set_box_aspect((1, 1, 1))
    # axes3.view_init(azim=110, elev=0, roll=-43) #Every frames
    axes3.view_init(azim=-68, elev=5, roll=3)  # Every frames

    axes3.axis('off')

    human_line_width = 5.5
    human_point_size = 90
    HUMAN_LINKS_Threshold = 133

    for name, kp_3d in plot_info['kp_3d'].items():
        if name == 'smpl':
            if kp_3d.shape[0] > 22:
                HUMAN_LINKS_Threshold = 133
            else:
                HUMAN_LINKS_Threshold = 22
            break
    
    for name, kp_3d in plot_info['kp_3d'].items(): 
        if name == 'smpl':
            HUMAN_LINKS = [link for link in HUMAN_LINKS_SMPL if all(l <= HUMAN_LINKS_Threshold for l in link)]
            plotcolor = 'palevioletred'
            if HUMAN_LINKS_Threshold > 22:
                hand_joints = np.concatenate((kp_3d[20:22],kp_3d[25:55],kp_3d[66:76]),axis = 0)
                axes3.scatter(hand_joints[:, 0],
                              hand_joints[:, 1],
                              hand_joints[:, 2], s=human_point_size, zorder=90, color=plotcolor)
        else:
            HUMAN_LINKS = [link for link in HUMAN_LINKS_COCOwholebody if all(l <= HUMAN_LINKS_Threshold for l in link)]
            plotcolor = 'lightgreen'
            if HUMAN_LINKS_Threshold > 22:
                hand_joints = kp_3d[91:]
                axes3.scatter(hand_joints[:, 0],
                              hand_joints[:, 1],
                              hand_joints[:, 2], s=human_point_size, zorder=90, color=plotcolor)
        
        human_segs3d = kp_3d[tuple([HUMAN_LINKS])]
        human_coll_3d = Line3DCollection(human_segs3d, linewidths=human_line_width, zorder=15, edgecolors=plotcolor)
        axes3.add_collection(human_coll_3d)

        body_joints = kp_3d[0:22]
        axes3.scatter(body_joints[:, 0],
                      body_joints[:, 1],
                      body_joints[:, 2], s=human_point_size, zorder=90, color=plotcolor)
        
        
        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()

    # plt.show()
    plt.close()

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(height, width, 3)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) #image_array[:, :, ::-1]  # rgb to bgr

    for name, kp_3d in plot_info['kp_3d'].items():
        if name == 'smpl':
            dataset = 'smpl'
            textpos = 'bottom'
            plotcolor_value = (147, 112, 229)            
        else:
            dataset = 'coco wholebody'
            textpos = 'top'
            plotcolor_value = (144, 238, 144)
        image_array = add_comment2imagearray(image_array, dataset, plot_info['frame_number'], plotcolor_value, textpos)
    
    # Visualization each frame by CV2
    # vis = cv2.resize(image_array, (1150, 1328))
    # cv2.imshow('img', vis)
    # cv2.waitKey(0)
    return image_array


def visualize_3d_whole_combination(data, proj_path, view_angle='whole'):
    plot_info = {
        'kp_3d': {key:None for key in data.keys()},
        'xyz': {
            'xlim': None,
            'ylim': None,
            'zlim': None
        },
        'frame_number': None
    }
    
    os.makedirs(f'./ik_results/{proj_path}', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'./ik_results/{proj_path}/output_{view_angle}.avi', fourcc, fps=30,
                          frameSize=[2300, 2656])


    framenum = list(data.values())[-1].shape[0]

    for f in range(framenum):
        plot_info['frame_number'] = f
        for k,v in data.items():
            plot_info['kp_3d'][k] = v[f]
        image_array = get_plot_from_pose(plot_info)
        out.write(image_array)
        print(f'{proj_path} frame {f} graph generated.')
    
