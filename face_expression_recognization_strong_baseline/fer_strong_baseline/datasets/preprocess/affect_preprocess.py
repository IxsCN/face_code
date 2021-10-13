# encoding:utf-8

import numpy as np
import cv2
import pandas as pd
import pdb
import torch
from tqdm import tqdm
import os
from skimage import io

from face_align import FaceAlign

# anno_list
img_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated/Manually_Annotated_Images"
save_img_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated/Face_Images_Small"
train_lst_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated_file_lists/training.csv"
val_lst_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated_file_lists/validation.csv"

affect_cls_num = ['normal', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'anger', 'contempt']

base_coords = [0.31556875, 0.46157422, 0.68262305, 0.45983398, 0.5002625,
                   0.64050547, 0.34947187, 0.8246918, 0.65343633, 0.82325078]

with open(train_lst_file, 'r') as f:
    samples = f.readlines()

head = samples[0]
print("read label...")
print(head)

samples = samples[1:]

for sample in tqdm(samples):
    subDirectory_filePath, face_x,face_y,face_width,face_height,facial_landmarks,expression,valence,arousal = sample.split(',')  
    try:
        subDirectory_filePath, face_x,face_y,face_width,face_height,facial_landmarks,expression,valence,arousal = \
            subDirectory_filePath, int(face_x),int(face_y),int(face_width),int(face_height),facial_landmarks.split(';'),int(expression),float(valence),float(arousal.split("\n")[0]) 
    except Exception as e:
        print("label of {} is wrong!".format(subDirectory_filePath))
        continue
    facial_landmarks = [int(float(x)) for x in facial_landmarks]

    img_path = os.path.join(img_file, subDirectory_filePath)
    save_img_path = os.path.join(save_img_file, subDirectory_filePath)

    if not os.path.exists(os.path.dirname(save_img_path)):
        os.makedirs(os.path.dirname(save_img_path))

    # pdb.set_trace()
    if os.path.exists(save_img_path):
        continue

    img = cv2.imread(img_path)

    # face_img = img[face_y:face_y + face_height, face_x:face_x + face_width, :]

    def get_five_landmark(landmarks):
        """
        代码中的索引只适用于AffectNet，其它数据集的landmark标注可能有问题
        """
        # landmark_idx = [30, 37, 38, 40, 41, 43, 44, 46, 47, 48, 54]
        
        # for p_idx in range(landmark_idx):
        #     px = landmarks[p_idx * 2]
        #     py = landmarks[p_idx * 2 + 1]

        nose = landmarks[30 * 2], landmarks[30 * 2 + 1]
        mouth_left = landmarks[48 * 2], landmarks[48 * 2 + 1]
        mouth_right = landmarks[54 * 2], landmarks[54 * 2 + 1]

        eye_lefts = [[landmarks[p_idx * 2], landmarks[p_idx * 2 + 1]] for p_idx in [37, 38, 40, 41]]
        eye_rights = [[landmarks[p_idx * 2], landmarks[p_idx * 2 + 1]] for p_idx in [43, 44, 46, 47]]

        eye_center_left = eye_lefts[0][0] + eye_lefts[1][0] + eye_lefts[2][0] + eye_lefts[3][0], \
                                eye_lefts[0][1] + eye_lefts[1][1] + eye_lefts[2][1] + eye_lefts[3][1]
        eye_center_left = eye_center_left[0]/4, eye_center_left[1]/4

        eye_center_right = eye_rights[0][0] + eye_rights[1][0] + eye_rights[2][0] + eye_rights[3][0], \
                                eye_rights[0][1] + eye_rights[1][1] + eye_rights[2][1] + eye_rights[3][1]
        eye_center_right = eye_center_right[0]/4, eye_center_right[1]/4

        return eye_center_left, eye_center_right, nose, mouth_left, mouth_right

    # for p_idx in range(68):
    #     px = facial_landmarks[p_idx * 2]
    #     py = facial_landmarks[p_idx * 2 + 1]
    #     # px, py = int(float(px)), int(float(py))   #  181, 530
    #     # px, py = int(float(px)) - face_x, int(float(py)) - face_y   #  181, 530
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     guiimg = cv2.putText(img.copy(), str(p_idx), (50, 300), font, 1.2, (255, 255, 255), 2)

    #     cv2.circle(guiimg, (px, py), 2, (255, 0, 0), 2)
    #     cv2.namedWindow("gui", 0)
    #     cv2.imshow("gui", guiimg)
    #     cv2.waitKey(0)

    five_landmarks = get_five_landmark(facial_landmarks)
    # for p in five_landmarks:
    #     px, py = int(p[0]), int(p[1])
    #     cv2.circle(img, (px, py), 2, (255, 0, 0), 2)

    # cv2.namedWindow("gui", 0)
    # cv2.imshow("gui", img)
    # cv2.waitKey(0)


    # cv2.estimateAffine2D()
    try:
        face_img = FaceAlign(img, five_landmarks, (100, 100))
    except Exception as e:
        print("img {} is wrong!".format(img_path))
        continue
    # cv2.namedWindow("gui", 0)
    # cv2.imshow("gui", face_img)
    # cv2.waitKey(0)
    # pdb.set_trace()

    cv2.imwrite(save_img_path, face_img)


