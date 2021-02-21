#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Scientific libraries
import numpy as np
import pandas as pd
import os
path = os.getcwd()
path
# Helper libraries
from tqdm.notebook import tqdm, trange
from colorama import Fore, Back, Style
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import ffmpeg
skeleton = [[15, 13],
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
   [0, 5],
   [0, 6]]

def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
    }
    return joints


def draw_points(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    """
    Draws `points` on `image`.
    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5
    Returns:
        A new image with overlaid points
    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))

    for i, pt in enumerate(points):
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)

    return image


def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0,
                  confidence_threshold=0.5):
    """
    Draws a `skeleton` on `image`.
    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5
    Returns:
        A new image with overlaid joints
    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            image = cv2.line(
                image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                tuple(colors[person_index % len(colors)]), 2
            )

    return image


def draw_points_and_skeleton(image, points, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5):
    """
    Draws `points` and `skeleton` on `image`.
    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5
    Returns:
        A new image with overlaid joints
    """
    image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                          palette_samples=skeleton_palette_samples, person_index=person_index,
                          confidence_threshold=confidence_threshold)
    image = draw_points(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold)
    return image


# In[13]:


from SimpleHRNet import SimpleHRNet

def extracthumanposeINFERENCE(nombrevideo, pintaresqueleto = False, device= 'cuda'):
    modelhuman = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", multiperson = False, return_bounding_boxes = False, device = device)
    nombrearchivocsv = 'DSMaskImagenesBBoxesTiemposPuntuacion' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv) 
    df = pd.DataFrame(data)
    puntos0 = []
    for x in tqdm(df.index):
        image = cv2.imread(path + '/imagenesjugadoresPARTIDO/' + df['p0filename'][x], cv2.IMREAD_COLOR)
        joints = modelhuman.predict(image)
        puntos0.append(joints)
        if pintaresqueleto:
            draw_points_and_skeleton(image, joints[0], skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5)
            plt.imshow(image)
            cv2.imwrite(path + '/esqueletojugador0/' +'esq' + df.p0filename[x],image)
    dataframe = pd.DataFrame(list(map(np.ravel, puntos0)))
    headers = ["p0_nose_x","p0_nose_y","p0_nose_sc"
                , "p0_left_eye_x","p0_left_eye_y","p0_left_eye_sc"
                , "p0_right_eye_x","p0_right_eye_y","p0_right_eye_sc"
                , "p0_left_ear_x","p0_left_ear_y","p0_left_ear_sc"
                , "p0_right_ear_x","p0_right_ear_y","p0_right_ear_sc"
                , "p0_left_shoulder_x", "p0_left_shoulder_y","p0_left_shoulder_sc"
                , "p0_right_shoulder_x","p0_right_shoulder_y","p0_right_shoulder_sc" 
                , "p0_left_elbow_x","p0_left_elbow_y","p0_left_elbow_sc"
                , "p0_right_elbow_x","p0_right_elbow_y","p0_right_elbow_sc"
                , "p0_left_wrist_x","p0_left_wrist_y","p0_left_wrist_sc"
                , "p0_right_wrist_x","p0_right_wrist_y","p0_right_wrist_sc"
                , "p0_left_hip_x","p0_left_hip_y", "p0_left_hip_sc"
                , "p0_right_hip_x","p0_right_hip_y", "p0_right_hip_sc"
                , "p0_left_knee_x","p0_left_knee_y","p0_left_knee_sc"
                , "p0right_knee_x", "p0_right_knee_y","p0_right_knee_sc"
                , "p0_left_ankle_x","p0_left_ankle_y","p0_left_ankle_sc"
                , "p0right_ankle_x", "p0_right_ankle_y","p0_right_ankle_sc"]
    dataframe.columns = headers
    dataframe.to_csv('humanposep0' + nombrevideo + '.csv',  encoding='utf-8')
    puntos1 = []
    for x in tqdm(df.index):
        image = cv2.imread(path + '/imagenesjugadoresPARTIDO/' + df['p1filename'][x], cv2.IMREAD_COLOR)
        joints = modelhuman.predict(image)
        puntos1.append(joints)
        if pintaresqueleto:
            draw_points_and_skeleton(image, joints[0], skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5)
            plt.imshow(image)
            cv2.imwrite(path + '/esqueletojugador1/' +'esq' + df.p1filename[x],image)
    dataframe1 = pd.DataFrame(list(map(np.ravel, puntos1)))
    headers1 = ["p1_nose_x","p1_nose_y","p1_nose_sc"
                , "p1_left_eye_x","p1_left_eye_y","p1_left_eye_sc"
                , "p1_right_eye_x","p1_right_eye_y","p1_right_eye_sc"
                , "p1_left_ear_x","p1_left_ear_y","p1_left_ear_sc"
                , "p1_right_ear_x","p1_right_ear_y","p1_right_ear_sc"
                , "p1_left_shoulder_x", "p1_left_shoulder_y","p1_left_shoulder_sc"
                , "p1_right_shoulder_x","p1_right_shoulder_y","p1_right_shoulder_sc" 
                , "p1_left_elbow_x","p1_left_elbow_y","p1_left_elbow_sc"
                , "p1_right_elbow_x","p1_right_elbow_y","p1_right_elbow_sc"
                , "p1_left_wrist_x","p1_left_wrist_y","p1_left_wrist_sc"
                , "p1_right_wrist_x","p1_right_wrist_y","p1_right_wrist_sc"
                , "p1_left_hip_x","p1_left_hip_y", "p1_left_hip_sc"
                , "p1_right_hip_x","p1_right_hip_y", "p1_right_hip_sc"
                , "p1_left_knee_x","p1_left_knee_y","p1_left_knee_sc"
                , "p1right_knee_x", "p1right_knee_y","p1right_knee_sc"
                , "p1_left_ankle_x","p1_left_ankle_y","p1_left_ankle_sc"
                , "p1right_ankle_x", "p1right_ankle_y","p1right_ankle_sc"]
    dataframe1.columns = headers1
    dataframe1.to_csv('humanposep1' + nombrevideo + '.csv',  encoding='utf-8')


# In[48]:


from SimpleHRNet import SimpleHRNet

def extracthumanposewithoutimagesINFERENCE(nombrevideo, pintaresqueleto = False, device= 'cuda'):
    modelhuman = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", multiperson = False, return_bounding_boxes = False, device = device)
    nombrearchivocsv = 'DSMaskImagenesBBoxesTiemposPuntuacion' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv) 
    df = pd.DataFrame(data)
    puntos0 = []
    for x in tqdm(df.index):
        photofilename = path + '/imagenesjuegoono/imagenesjuegoPARTIDO/' + df['Filename'][x]
        img = cv2.imread(photofilename, cv2.IMREAD_COLOR)
        if np.isnan(df.p0score[x]):
            puntos0.append(np.nan)
        else:
            image = img[int(df.p0ymin[x]):int(df.p0ymax[x]),int(df.p0xmin[x]):int(df.p0xmax[x])]
            joints = modelhuman.predict(image)
            puntos0.append(joints)
            if pintaresqueleto:
                draw_points_and_skeleton(image, joints[0], skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5)
                plt.imshow(image)
                cv2.imwrite(path + '/esqueletojugador0/' +'esq' + df.p0filename[x],image)
    dataframe = pd.DataFrame(list(map(np.ravel, puntos0)))
    headers = ["p0_nose_x","p0_nose_y","p0_nose_sc"
                , "p0_left_eye_x","p0_left_eye_y","p0_left_eye_sc"
                , "p0_right_eye_x","p0_right_eye_y","p0_right_eye_sc"
                , "p0_left_ear_x","p0_left_ear_y","p0_left_ear_sc"
                , "p0_right_ear_x","p0_right_ear_y","p0_right_ear_sc"
                , "p0_left_shoulder_x", "p0_left_shoulder_y","p0_left_shoulder_sc"
                , "p0_right_shoulder_x","p0_right_shoulder_y","p0_right_shoulder_sc" 
                , "p0_left_elbow_x","p0_left_elbow_y","p0_left_elbow_sc"
                , "p0_right_elbow_x","p0_right_elbow_y","p0_right_elbow_sc"
                , "p0_left_wrist_x","p0_left_wrist_y","p0_left_wrist_sc"
                , "p0_right_wrist_x","p0_right_wrist_y","p0_right_wrist_sc"
                , "p0_left_hip_x","p0_left_hip_y", "p0_left_hip_sc"
                , "p0_right_hip_x","p0_right_hip_y", "p0_right_hip_sc"
                , "p0_left_knee_x","p0_left_knee_y","p0_left_knee_sc"
                , "p0right_knee_x", "p0_right_knee_y","p0_right_knee_sc"
                , "p0_left_ankle_x","p0_left_ankle_y","p0_left_ankle_sc"
                , "p0right_ankle_x", "p0_right_ankle_y","p0_right_ankle_sc"]
    dataframe.columns = headers
    dataframe.to_csv('humanposep0' + nombrevideo + '.csv',  encoding='utf-8')
    puntos1 = []
    for x in tqdm(df.index):
        photofilename = path + '/imagenesjuegoono/imagenesjuegoPARTIDO/' + df['Filename'][x]
        img = cv2.imread(photofilename, cv2.IMREAD_COLOR)
        if np.isnan(df.p1score[x]):
            puntos1.append(np.nan)
        else:
            image = img[int(df.p1ymin[x]):int(df.p1ymax[x]),int(df.p1xmin[x]):int(df.p1xmax[x])]
            joints = modelhuman.predict(image)
            puntos1.append(joints)
            if pintaresqueleto:
                draw_points_and_skeleton(image, joints[0], skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5)
                plt.imshow(image)
                cv2.imwrite(path + '/esqueletojugador1/' +'esq' + df.p1filename[x],image)
    dataframe1 = pd.DataFrame(list(map(np.ravel, puntos1)))
    headers1 = ["p1_nose_x","p1_nose_y","p1_nose_sc"
                , "p1_left_eye_x","p1_left_eye_y","p1_left_eye_sc"
                , "p1_right_eye_x","p1_right_eye_y","p1_right_eye_sc"
                , "p1_left_ear_x","p1_left_ear_y","p1_left_ear_sc"
                , "p1_right_ear_x","p1_right_ear_y","p1_right_ear_sc"
                , "p1_left_shoulder_x", "p1_left_shoulder_y","p1_left_shoulder_sc"
                , "p1_right_shoulder_x","p1_right_shoulder_y","p1_right_shoulder_sc" 
                , "p1_left_elbow_x","p1_left_elbow_y","p1_left_elbow_sc"
                , "p1_right_elbow_x","p1_right_elbow_y","p1_right_elbow_sc"
                , "p1_left_wrist_x","p1_left_wrist_y","p1_left_wrist_sc"
                , "p1_right_wrist_x","p1_right_wrist_y","p1_right_wrist_sc"
                , "p1_left_hip_x","p1_left_hip_y", "p1_left_hip_sc"
                , "p1_right_hip_x","p1_right_hip_y", "p1_right_hip_sc"
                , "p1_left_knee_x","p1_left_knee_y","p1_left_knee_sc"
                , "p1right_knee_x", "p1right_knee_y","p1right_knee_sc"
                , "p1_left_ankle_x","p1_left_ankle_y","p1_left_ankle_sc"
                , "p1right_ankle_x", "p1right_ankle_y","p1right_ankle_sc"]
    dataframe1.columns = headers1
    dataframe1.to_csv('humanposep1' + nombrevideo + '.csv',  encoding='utf-8')


# In[ ]:




