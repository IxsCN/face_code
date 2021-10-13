# encoding:utf-8

import numpy as np
import cv2
import pandas as pd
import pdb
import torch
from torch.utils.data import Dataset
from fer_strong_baseline.datasets.aug import fer_test_aug,fer_train_aug
from  fer_strong_baseline.config.default_cfg import  get_fer_cfg_defaults
from  fer_strong_baseline.utils.common import setup_seed
from tqdm import tqdm
import scipy.misc as sm
from torchvision import transforms
import os
import random
from fer_strong_baseline.datasets.aug import add_gaussian_noise, flip_image
from fer_strong_baseline.utils.common import get_center_point, get_most_left_and_most_right_point, get_nearest_landmark_idx

from fer_strong_baseline.datasets.gaussian import GaussianTransformer

import face_alignment
from skimage import io
import pdb
import cv2
from random import randint


class MakeRafGaussianMask(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.img_dir = cfg.DATA.img_dir

        if self.is_train:
            self.data_lst = cfg.DATA.train_label_path
          
        else:
            self.data_lst = cfg.DATA.val_label_path

        self.landmark_dir = os.path.join(os.path.dirname(os.path.dirname(self.data_lst)), 'Annotation', 'manual')

        f = open(self.data_lst, 'r')
        label_lst = f.readlines()
        f.close()

        samples = []
        labels = []
        landmarks = []
        boxeses = []
        genders = []
        races = []
        ages = []

        instances = label_lst[0].split('.jpg')
        head = instances[0]
        for i in tqdm(range(1, len(instances))):
            instance = instances[i]
            new_head = instance[-11:]
            if 'test' in new_head:
                new_head = instance[-9:]

            if i != len(instances) - 1:
                if 'test' in new_head:
                    instance = head + '.jpg' + instance[:-9]
                else:
                    instance = head + '.jpg' + instance[:-11]
            else:
                instance = head + '.jpg' + instance

            
            img_fn, boxes, preds, gt = instance.split('=')
            samples.append(img_fn)
            labels.append(gt)

            boxes = boxes.split(';')
            preds = preds.split(';')
            boxes = [[float(p) for p in box.split(' ')] for box in boxes]
            preds = [[[int(float(p)) for p in landp.split(' ')] for landp in pred.split(',')] for pred in preds]
            head = new_head

            annotation_file = open(os.path.join(self.landmark_dir, img_fn.replace('.jpg', '_manu_attri.txt')), 'r')
            annotations = annotation_file.readlines()
            annotations = [anno.replace('\n', '').replace('\t', ' ') for anno in annotations]
            gt_landmarks = annotations[:5]
            gt_gender = annotations[5]
            gt_race = annotations[6]
            gt_age = annotations[7]
            gt_landmarks = [[int(float(x)) for x in landp.split(' ')] for landp in gt_landmarks]
            genders.append(gt_gender)
            races.append(gt_race)
            ages.append(gt_age)

            assert len(boxes) == len(preds), print("num of boxes not equal to num of landmarks!")

            five_landmark = []
            for i in range(len(preds)):
                left_corner_of_mouth, right_corner_of_mouth = get_most_left_and_most_right_point(preds[i][48:])
                left_eye_candidate = []
                # left_eye_candidate.extend(preds[i][22:27])
                left_eye_candidate.extend(preds[i][42:48])
                left_eye = get_center_point(left_eye_candidate)
                right_eye_candidate = []
                # right_eye_candidate.extend(preds[i][17:22])
                right_eye_candidate.extend(preds[i][36:42])
                right_eye = get_center_point(right_eye_candidate)

                nose = get_center_point(preds[i][27:36])

                five_landmark.append([left_corner_of_mouth, right_corner_of_mouth, left_eye, right_eye, nose])

            idx = get_nearest_landmark_idx(gt_landmarks, five_landmark)
            landmarks.append(preds[idx])
            # boxeses.append(boxes[idx])

            nptmp = np.array(preds[idx])
            bbot = cv2.boundingRect(nptmp)
            boxeses.append(list(bbot))


        self.rafdb_cls_num = ['surprised', 'fear', 'disgust', 'happy',
                   'sad', 'anger', 'normal']


        train_cnt = 0
        test_cnt = 0

        train_samples = []
        train_labels = []
        train_landmarks = []
        train_boxes = []
        train_genders = []
        train_races = []
        train_ages = []

        test_samples = []
        test_labels = []
        test_landmarks = []
        test_boxes = []
        test_genders = []
        test_races = []
        test_ages = []

        for i in range(len(samples)):
            img_fn = samples[i]
            if 'train' in img_fn:
                train_samples.append(samples[i])
                train_labels.append(labels[i])
                train_landmarks.append(landmarks[i])
                train_boxes.append(boxeses[i])
                train_genders.append(genders[i])
                train_races.append(races[i])
                train_ages.append(ages[i])
                train_cnt += 1
            if 'test' in img_fn:
                test_samples.append(samples[i])
                test_labels.append(labels[i])
                test_landmarks.append(landmarks[i])
                test_boxes.append(boxeses[i])
                test_genders.append(genders[i])
                test_races.append(races[i])
                test_ages.append(ages[i])
                test_cnt += 1

            if self.is_train:
                self.samples = train_samples
                self.labels = train_labels
                self.landmarks = train_landmarks
                self.boxes = train_boxes
                self.genders = train_genders
                self.races = train_races
                self.ages = train_ages
            else:
                self.samples = test_samples
                self.labels = test_labels
                self.landmarks = test_landmarks
                self.boxes = test_boxes
                self.genders = test_genders
                self.races = test_races
                self.ages = test_ages

        print("dataset has {} samples".format(len(self.samples)))

    def __len__(self):
        assert len(self.samples) == len(self.labels)
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        save_face_crop_dir = os.path.join(self.img_dir, path).replace('original', 'face_crop')
        save_gaussian_mask_dir = os.path.join(self.img_dir, path).replace('original', 'face_mask')
        
        if os.path.exists(save_gaussian_mask_dir):
            return idx
        
        try:
            image = cv2.imread(os.path.join(self.img_dir, path))
            h, w, _ = image.shape
            label = self.labels[idx]
            face_box = self.boxes[idx]
        except Exception as e:
            pdb.set_trace()

        

        '''
        landmark crop and augmentation
        '''
        pred = self.landmarks[idx]

        ## extract part of landmark
        '''
        # 0-16是下颌线(红），
        # 17-21是右眼眉（橙），
        # 22-26是左眼眉（黄），
        # 27-35是鼻子（浅绿），
        # 36-41是右眼（深绿），
        # 42-47是左眼（浅蓝），
        # 48-60是嘴外轮廓（深蓝），
        # 61-67是嘴内轮廓（紫）
        '''
        chin = pred[:17]
        r_eyebrow = pred[17:22]
        l_eyebrow = pred[22:27]
        nose = pred[27:36]
        r_eye = pred[37:42]
        l_eye = pred[42:48]
        ex_mouth_outline = pred[48:61]
        in_mouth_outline = pred[61:68]

        def eyeboxmerge(box1, box2):
            b1x1, b1y1, b1w, b1h = box1
            b2x1, b2y1, b2w, b2h = box2
            b1x2 = b1x1 + b1w
            b1y2 = b1y1 + b1h
            b2x2 = b2x1 + b2w
            b2y2 = b2y1 + b2h

            b1cx = (b1x1 + b1x2) / 2
            b2cx = (b2x1 + b2x2) / 2
            ytop = min(b1y1, b2y1)
            ydown = max(b1y2, b2y2)
            xleft = b1cx
            xright = b2cx

            return [int(xleft), int(ytop), int(xright - xleft), int(ydown - ytop)]


        ## calc bounding box of landmark part
        box_1 = cv2.boundingRect(np.concatenate((r_eyebrow, r_eye)))  # right eye
        box_2 = cv2.boundingRect(np.concatenate((l_eyebrow, l_eye)))  # left eye
        box_3 = cv2.boundingRect(np.concatenate((ex_mouth_outline, in_mouth_outline)))  # mouth
        box_4 = cv2.boundingRect(np.array(nose))  # nose
        box_5 = eyeboxmerge(box_1, box_2)
        boxes = [box_1, box_2, box_3, box_4, box_5]

        # 四边形，顺时针，左上起
        boxes = [[[x[0], x[1]], [x[0] + x[2], x[1]], [x[0] + x[2],  x[1] + x[3]], [x[0], x[1] + x[3]]] for x in boxes]
        boxes = [np.array(box) for box in boxes]

        

        gaussian = GaussianTransformer(512, 0.4, 0.2)
        gaussian_mask = gaussian.generate_region((image.shape[0], image.shape[1], 1), boxes)

        try:
            x,y,w,h = face_box
            x = max(0, x)
            y = max(0, y)
            w = min(image.shape[1] - 1, w)
            h = min(image.shape[0] - 1, h)
            face_box = [x,y,w,h]

            face_crop = image[face_box[1]: face_box[1] + face_box[3], face_box[0]: face_box[0] + face_box[2], :]
            face_mask = gaussian_mask[face_box[1]: face_box[1] + face_box[3], face_box[0]: face_box[0] + face_box[2]]
        except Exception as e:
            print("bad crop!")
            pdb.set_trace()

        try:
            face_crop = cv2.resize(face_crop, (224, 224))
            face_mask = cv2.resize(face_mask, (224, 224))
        except Exception as e:
            print("croped img is empty!")
            pdb.set_trace()


        cv2.imwrite(save_face_crop_dir, face_crop)
        cv2.imwrite(save_gaussian_mask_dir, face_mask)

        # try:
        #     ## visualize
        #     color_vec = [[randint(155, 255), randint(155, 255), randint(155, 255)] for _ in range(9)]
        #     cv2.rectangle(gui, (box_1[0], box_1[1]), (box_1[0] + box_1[2], box_1[1] + box_1[3]), color_vec[0], 2)
        #     cv2.rectangle(gui, (box_2[0], box_2[1]), (box_2[0] + box_2[2], box_2[1] + box_2[3]), color_vec[1], 2)
        #     cv2.rectangle(gui, (box_3[0], box_3[1]), (box_3[0] + box_3[2], box_3[1] + box_3[3]), color_vec[2], 2)
        #     cv2.rectangle(gui, (box_4[0], box_4[1]), (box_4[0] + box_4[2], box_4[1] + box_4[3]), color_vec[3], 2)
        #     cv2.rectangle(gui, (box_5[0], box_5[1]), (box_5[0] + box_5[2], box_5[1] + box_5[3]), color_vec[4], 2)
        # except Exception as e:
        #     pdb.set_trace()
        # cv2.namedWindow('landmark', 0)
        # cv2.imshow('landmark', gui)
        # cv2.waitKey(0)
        # pdb.set_trace()

        # cv2.namedWindow("g", 0)
        # cv2.imshow("g", gaussian_mask)
        # cv2.waitKey(0)

     
        return idx




if __name__ == "__main__":
    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    config_path = '../../configs/rafdb_scnres18_gaussian_ce.yml'
    cfg.merge_from_file(config_path)

    cfg.DATA.img_dir = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Image/original'

    cfg.DATA.train_label_path='/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/EmoLabel/new_list_patition_label.txt'
    cfg.DATA.val_label_path='/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/EmoLabel/new_list_patition_label.txt'
    #label_path = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/EmoLabel/new_list_patition_label.txt'
    #img_path = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Image/original'
    
    mask_generator = MakeRafGaussianMask(cfg, False)
    for idx in tqdm(mask_generator):
        continue
