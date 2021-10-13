# -*- coding: utf-8 -*-
import requests
import base64
import json
import pdb
import array
import os
import cv2
import shutil

class UpdateDatasetToEasyDLCompress:
    def __init__(self, cls_names):
        self.cls_names = cls_names

    def create_dataset(self, img_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.img_path = img_path
        return True

    def upload_sample(self, img_fn, label):
        try:
            label_name = self.cls_names[label]
        except Exception as e:
            return 
        sub_dir = os.path.basename(os.path.dirname(img_fn))
        img_name = os.path.basename(img_fn)
        save_img_file_path = os.path.join(self.save_path, sub_dir, img_name)
        save_json_file_path = os.path.join(self.save_path, sub_dir, img_name.split('.')[0] + ".json")
        

        if not os.path.exists(os.path.dirname(save_img_file_path)):
            os.makedirs(os.path.dirname(save_img_file_path))

        jdict = {"labels": [{"name": label_name}]}
        jstr = json.dumps(jdict)
        with open(save_json_file_path, "w", encoding='utf-8') as f:
            f.write(jstr)
        shutil.copy(img_fn, save_img_file_path)
        
        return
        


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    img_dir = '/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated/Face_Images_Small'
    save_dir = '/media/yz/62C9BA4E5D826344/data/AffectNetCompress'
    train_lst_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated_file_lists/training.csv"
    val_lst_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated_file_lists/validation.csv"
    affect_cls_num = ['normal', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'anger', 'contempt']

    error_img_lst = [
            "2/9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg",
            "994/8fb7a123f36ae89be4bf72e9fa77d1434a29135f3342f62fb55c7aee.jpg",
        ]

    affect_cls_num = ['normal', 'happy', 'sad', 'surprised',
                                 'fear', 'disgust', 'anger', 'contempt']

    datauploader = UpdateDatasetToEasyDLCompress(affect_cls_num)
   
    if not datauploader.create_dataset(img_dir, save_dir):
        exit(-1)
    # datauploader.dataset_id = 136812

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

        if subDirectory_filePath in error_img_lst:
            continue

        img_path = os.path.join(img_dir, subDirectory_filePath)
        datauploader.upload_sample(img_path, expression)
