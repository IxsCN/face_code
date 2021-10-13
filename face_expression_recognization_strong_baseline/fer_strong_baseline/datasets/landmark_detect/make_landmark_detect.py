import face_alignment
from skimage import io
import pdb
import cv2
from random import randint
import os
from tqdm import tqdm
import numpy as np

from fer_strong_baseline.utils.common import get_center_point, get_most_left_and_most_right_point

# distribution of landmark  https://blog.csdn.net/keyanxiaocaicai/article/details/52150322
#
# 0-16是下颌线（红），
# 17-21是右眼眉（橙），
# 22-26是左眼眉（黄），
# 27-35是鼻子（浅绿），
# 36-41是右眼（深绿），
# 42-47是左眼（浅蓝），
# 48-60是嘴外轮廓（深蓝），
# 61-67是嘴内轮廓（紫）

landmark_idx_tmp =  [[0 for _ in range(0,17)],
                 [1 for _ in range(17,22)],
                 [2 for _ in range(22,27)],
                 [3 for _ in range(27,36)],
                 [4 for _ in range(36,42)],
                 [5 for _ in range(42,48)],
                 [6 for _ in range(48,61)],
                 [7 for _ in range(61,68)]]

landmark_idx = []
for group in landmark_idx_tmp:
    landmark_idx.extend(group)

color_vec = [[randint(155, 255), randint(155, 255), randint(155, 255)] for _ in range(8)]


img_dir = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Image/original'
bbox_dir = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Annotation/boundingbox'
landmark_dir = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Annotation/manual'
label_dir = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/EmoLabel/list_patition_label.txt'

save_landmark_dir = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/EmoLabel/new_list_patition_label.txt'

def detect_landmark():
    f = open(label_dir, 'r')
    label_lst = f.readlines()
    f.close()
    img_names = []
    gts = []
    for sample in label_lst:
        sample = sample.replace('\n', '')
        img_name, gt = sample.replace('\t', ' ').split(' ')
        img_names.append(img_name)
        gts.append(gt)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    nf = open(save_landmark_dir, 'w')

    def lst2str(lst):
        '''
        :param lst: person_num * 68 * 2
        :return: string
        '''
        slst = ';'.join([','.join([' '.join([str(x) for x in l]) for l in p]) for p in lst])
        return slst

    for idx, (img_name, gt) in tqdm(enumerate(zip(img_names, gts))):
        try:
            full_img_name = os.path.join(img_dir, img_name)
            input = io.imread(full_img_name)
            boxes, preds = fa.get_boxes_and_landmarks(input)

            # n * 4
            lst_boxes = [box.tolist() for box in boxes]
            all_boxes_str = ';'.join([' '.join([str(p) for p in box]) for box in lst_boxes])

            lst_preds = [pred.tolist() for pred in preds]
            all_preds_str = lst2str(lst_preds)
            nf.write('{}={}={}={}'.format(img_name, all_boxes_str, all_preds_str, gt))


        except Exception as e:
            print(e)
            print("bad img:{}".format(img_name))
            pass

    nf.close()
    exit(0)


def data_correct():
    f = open(save_landmark_dir, 'r')
    label_lst = f.readlines()
    f.close()
    img_names = []
    gts = []
    samples = label_lst[0].split('.jpg')
    head = samples[0]
    for i in range(1, len(samples)):
        sample = samples[i]
        new_head = sample[-11:]
        if 'test' in new_head:
            new_head = sample[-9:]

        if i != len(samples) - 1:
            if 'test' in new_head:
                sample = head + '.jpg' + sample[:-9]
            else:
                sample = head + '.jpg' + sample[:-11]
        else:
            sample = head + '.jpg' + sample

        img_fn, boxes, preds, gt = sample.split('=')

        boxes = boxes.split(';')
        preds = preds.split(';')
        boxes = [[float(p) for p in box.split(' ')] for box in boxes]
        preds = [[ [int(float(p)) for p in landp.split(' ')] for landp in pred.split(',')] for pred in preds]
        head = new_head

        annotation_file = open(os.path.join(landmark_dir, img_fn.replace('.jpg', '_manu_attri.txt')), 'r')
        annotations = annotation_file.readlines()
        annotations = [anno.replace('\n', '').replace('\t', ' ') for anno in annotations]
        gt_landmarks = annotations[:5]
        gt_gender = annotations[5]
        gt_race = annotations[6]
        gt_age = annotations[7]
        gt_landmarks = [[int(float(x)) for x in landp.split(' ')] for landp in gt_landmarks]

        assert len(boxes) == len(preds), print("num of boxes not equal to num of landmarks!")

        show_lst = []
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

            show_lst.append([left_corner_of_mouth, right_corner_of_mouth, left_eye, right_eye, nose])

        # visualize
        full_img_name = os.path.join(img_dir, img_fn)

        # pred gui
        # 0-16是下颌线（红），
        # 17-21是右眼眉（橙），
        # 22-26是左眼眉（黄），
        # 27-35是鼻子（浅绿），
        # 36-41是右眼（深绿），
        # 42-47是左眼（浅蓝），
        # 48-60是嘴外轮廓（深蓝），
        # 61-67是嘴内轮廓（紫）
        # pred_gui = cv2.imread(full_img_name)
        # for pred in preds:
        #     for i, color_i in enumerate(landmark_idx):
        #         if i in range(48, 61):
        #             r, g, b = color_vec[color_i]
        #             cv2.circle(pred_gui, (pred[i][0], pred[i][1]), 3, (b,g,r), 3)
        # cv2.namedWindow('pred_landmark', 0)
        # cv2.imshow('pred_landmark', pred_gui)

        pred_gui = cv2.imread(full_img_name)
        for slst in show_lst:
            for color_i, pred in enumerate(slst):
                r, g, b = color_vec[color_i]
                cv2.circle(pred_gui, (int(pred[0]), int(pred[1])), 3, (b,g,r), 3)
            cv2.namedWindow('pred_landmark', 0)
            cv2.imshow('pred_landmark', pred_gui)

        # gt
        gt_gui = cv2.imread(full_img_name)
        for pred in preds:
            for i, gt_land_p in enumerate(gt_landmarks):
                r, g, b = color_vec[i]
                cv2.circle(gt_gui, (gt_land_p[0], gt_land_p[1]), 3, (b, g, r), 3)
        cv2.namedWindow('gt_landmark', 0)
        cv2.imshow('gt_landmark', gt_gui)
        cv2.waitKey(0)
        pdb.set_trace()


data_correct()




