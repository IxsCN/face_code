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


class RafGaussianOfflineDataSet(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
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
                            transforms.RandomErasing(scale=(0.02, 0.25))
                            ])
        else:
            self.data_lst = cfg.DATA.val_label_path
            self.aug = fer_test_aug(cfg.DATA.input_size)
            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                            transforms.ToTensor(),   # will divide 255
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

        

        self.gaussian_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0],
                            #                      std=[255])
                            ])


        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(self.data_lst, sep=' ', header=None)

        if self.is_train:
            # dataset = df[df[NAME_COLUMN].str.startswith('test')]
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
            # dataset = df[df[NAME_COLUMN].str.startswith('train')]

        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.labels = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.labels = np.array(self.labels)

        # use raf aligned images for training/testing
        self.aligned_samples = [os.path.join(cfg.DATA.img_dir,
                                     x.split('\n')[0].split(' ')[0].replace(".jpg", "_aligned.jpg"))
                        for x in file_names]

        self.face_crop_samples = [os.path.join(cfg.DATA.img_dir.replace("aligned", "face_crop"), 
                                     x.split('\n')[0].split(' ')[0])
                        for x in file_names]

        # 有些图没有检测到landmark，此时只能用aligned image替代
        self.samples_and_labels = [{x1:y}  if os.path.exists(x1) else {x2:y} for (x1, x2, y) in zip(self.face_crop_samples, self.aligned_samples, self.labels)]
        # self.samples_and_labels = [{x:y} for (x, y) in zip(self.samples, self.labels)]

        # TODO. 写的不合理,self.face_crop_samples需要用 os.path.exist过滤一下
        print("The number of failed images:", len(self.aligned_samples) - len(self.face_crop_samples))
        print("The num of missed images:", len(self.aligned_samples) - len(self.samples_and_labels))

        self.samples = [list(x.keys())[0] for x in self.samples_and_labels]
        self.labels = [list(x.values())[0] for x in self.samples_and_labels]


        self.gaussianes = [x.replace("face_crop", "face_mask") for x in self.samples]

        self.rafdb_cls_num = ['surprised', 'fear', 'disgust', 'happy',
                   'sad', 'anger', 'normal']

        self.aug_func = [flip_image, add_gaussian_noise]

       
        print("dataset has {} samples".format(len(self.samples)))

    def __len__(self):
        assert len(self.samples) == len(self.labels)
        return len(self.samples)

    def __getitem__(self, idx):

        try:
            # /media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Image/face_crop/train_11176.jpg
            
            path = self.samples[idx]
            # rgb_dir = os.path.dirname(path)
            # rgb_fn = os.path.basename(path)
            # rgb_dir = rgb_dir.replace("face_crop", "aligned")
            # rgb_fn = rgb_fn.replace(".jpg", "_aligned.jpg")
            # path = os.path.join(rgb_dir, rgb_fn)
            gaussian_path = self.gaussianes[idx]
            # import pdb; pdb.set_trace()
            image = cv2.imread(path)
            # img_b, img_g, img_r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            

            # 如果没有landmark， gaussian 图也不存在，用全1代替
            # TODO. 确定gaussian image的像素值范围
            if os.path.exists(gaussian_path):
                gaussian_img = cv2.imread(gaussian_path)
                gaussian_img = cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2GRAY)
            else:
                gaussian_img = np.ones(image.shape, np.uint8) * 255

            h, w, _ = image.shape
            # print(path)
            # image = image[:, :, ::-1]  # BGR to RGB
            label = self.labels[idx]
            # print("label:", label)
        except Exception as e:
            pdb.set_trace()

        # cv2.namedWindow('image', 0)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        
        # cv2.namedWindow('gaussian_img', 0)
        # cv2.imshow('gaussian_img', gaussian_img)
        # cv2.waitKey(0)

        if self.is_train:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        image = self.transform(image)

        gaussian_mask = self.gaussian_transform(gaussian_img)

        input_image = torch.cat((image, gaussian_mask), dim = 0)

        # input_image = image

        # try:
        #     label = torch.from_numpy(np.array([int(x) - 1 for x in label]))
        # except Exception as e:
        #     label = torch.from_numpy(np.array([int(label) - 1])).long()
        label = int(label)
        # label = label.squeeze()
        # print("get {}th img in dataset".format(idx))
        return input_image, label, idx


class RafGaussianDataSet(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
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
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])


        self.gaussian_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            ])

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
        box_5 = eyeboxmerge(box_1, box_2)
        boxes = [box_1, box_2, box_3, box_4, box_5]

        # pdb.set_trace()
        boxes = [[[x[0], x[1]], [x[0] + x[2], x[1]], [x[0] + x[2],  x[1] + x[3]], [x[0], x[1] + x[3]]] for x in boxes]
        boxes = [np.array(box) for box in boxes]

        # bbox4 = np.array([[[96, 0], [151, 9], [139, 64], [83, 58]]])
        

        gaussian = GaussianTransformer(512, 0.4, 0.2)
        gaussian_mask = gaussian.generate_region((self.cfg.DATA.input_size, self.cfg.DATA.input_size, 1), boxes)

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

        if self.is_train:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        image = self.transform(image)
        gaussian_mask = self.gaussian_transform(gaussian_mask)

        input_image = torch.cat((image, gaussian_mask), dim = 0)

        label = torch.from_numpy(np.array([int(x) - 1 for x in label]))
        # print("get {}th img in dataset".format(idx))
        return input_image, label, idx