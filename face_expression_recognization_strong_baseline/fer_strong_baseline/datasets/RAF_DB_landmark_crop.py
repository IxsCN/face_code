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


import face_alignment
from skimage import io
import pdb
import cv2
from random import randint


landmark_group_idx =  [[0 for _ in range(0,17)],
                 [1 for _ in range(17,22)],
                 [2 for _ in range(22,27)],
                 [3 for _ in range(27,36)],
                 [4 for _ in range(36,42)],
                 [5 for _ in range(42,48)],
                 [6 for _ in range(48,61)],
                 [7 for _ in range(61,68)],
                  ]

def expand_box2square(box, img_shape, ratio):
    h, w, _ = img_shape
    x_extand = ratio * box[2]  # box_w
    y_extand = ratio * box[3]  # box_h
    center_x, center_y = int(box[0] + box[2]/2), int(box[1] + box[3]/2)
    left = center_x - x_extand / 2
    top = center_y - y_extand / 2
    right = center_x + x_extand / 2
    down = center_y + y_extand / 2

    left = int(max(left, 0))
    top = int(max(top, 0))
    right = int(min(w - 1, right))
    down = int(min(h - 1, down))

    return [left, top, right - left, down - top]



class LandmarkCropRafDataSet(Dataset):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.img_dir = cfg.DATA.img_dir

        if self.is_train:
            self.data_lst = cfg.DATA.train_label_path
            self.aug = fer_train_aug(cfg.DATA.input_size,
                                     cfg.DATA.crop_residual_pix)
            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                            transforms.RandomErasing(scale=(0.02, 0.25))])
        else:
            self.data_lst = cfg.DATA.val_label_path
            self.aug = fer_test_aug(cfg.DATA.input_size)
            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
        self.landmark_dir = os.path.join(os.path.dirname(os.path.dirname(self.data_lst)), 'Annotation', 'manual')

        if self.is_train:
            self.random_erasing = transforms.RandomErasing(scale=(0.02,0.25))
        else:
            self.random_erasing = None

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

        self.aug_func = [flip_image, add_gaussian_noise]

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

        # pdb.set_trace()
        print("dataset has {} samples".format(len(self.samples)))

    def __len__(self):
        assert len(self.samples) == len(self.labels)
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            path = self.samples[idx]
            image = cv2.imread(os.path.join(self.img_dir, path))
            gui = image.copy()
            h, w, _ = image.shape
            # print(path)
            # image = image[:, :, ::-1]  # BGR to RGB
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
        box_5 = cv2.boundingRect(np.array(chin[:9]))  # left chin
        box_6 = cv2.boundingRect(np.array(chin[9:]))  # right chin
        box_8 = [int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])]   # center crop
        box_9 = [0, 0, w - 1, h -1]  # raw

        ## expand box from rectangle to square
        box_1 = expand_box2square(box_1, image.shape, 1.3)
        box_2 = expand_box2square(box_2, image.shape, 1.3)
        box_3 = expand_box2square(box_3, image.shape, 1.3)
        box_4 = expand_box2square(box_4, image.shape, 1.3)
        box_5 = expand_box2square(box_5, image.shape, 1.3)
        box_6 = expand_box2square(box_6, image.shape, 1.3)
        # box_7 = expand_box2square(box_7, image.shape, 1.3)
        # box_8 = expand_box2square(box_8, image.shape, 1)
        # box_9 = expand_box2square(box_9, image.shape, 1)
        box_7 = eyeboxmerge(box_1, box_2)

        # try:
        #     ## visualize
        #     color_vec = [[randint(155, 255), randint(155, 255), randint(155, 255)] for _ in range(9)]
        #     cv2.rectangle(gui, (box_1[0], box_1[1]), (box_1[0] + box_1[2], box_1[1] + box_1[3]), color_vec[0], 2)
        #     cv2.rectangle(gui, (box_2[0], box_2[1]), (box_2[0] + box_2[2], box_2[1] + box_2[3]), color_vec[1], 2)
        #     cv2.rectangle(gui, (box_3[0], box_3[1]), (box_3[0] + box_3[2], box_3[1] + box_3[3]), color_vec[2], 2)
        #     cv2.rectangle(gui, (box_4[0], box_4[1]), (box_4[0] + box_4[2], box_4[1] + box_4[3]), color_vec[3], 2)
        #     cv2.rectangle(gui, (box_5[0], box_5[1]), (box_5[0] + box_5[2], box_5[1] + box_5[3]), color_vec[4], 2)
        #     cv2.rectangle(gui, (box_6[0], box_6[1]), (box_6[0] + box_6[2], box_6[1] + box_6[3]), color_vec[5], 2)
        #     cv2.rectangle(gui, (box_7[0], box_7[1]), (box_7[0] + box_7[2], box_7[1] + box_7[3]), color_vec[6], 2)
        #     cv2.rectangle(gui, (box_8[0], box_8[1]), (box_8[0] + box_8[2], box_8[1] + box_8[3]), color_vec[7], 2)
        #     cv2.rectangle(gui, (box_9[0], box_9[1]), (box_9[0] + box_9[2], box_9[1] + box_9[3]), color_vec[8], 2)
        # except Exception as e:
        #     pdb.set_trace()
        # cv2.namedWindow('landmark', 0)
        # cv2.imshow('landmark', gui)
        # cv2.waitKey(0)
        # pdb.set_trace()

        input_image = []
        boxes = [box_1, box_2, box_3, box_4, box_5, box_6, box_7, box_8, box_9]
        for i in range(9):
            try:
                # image_i = image[boxes[i]]
                image_i = image
            except Exception as e:
                pdb.set_trace()
            if self.is_train:
                if random.uniform(0, 1) > 0.5:
                    index = random.randint(0, 1)
                    image_i = self.aug_func[index](image_i)
            image_i = self.transform(image_i)
            input_image.append(image_i)

        label = torch.from_numpy(np.array([int(x) - 1 for x in label]))

        return input_image[0], label, input_image[1], label,   \
                 input_image[2], label, input_image[3], label,  \
                input_image[4], label, input_image[5], label,  \
                input_image[6], label, input_image[7], label, \
                input_image[8], label

                