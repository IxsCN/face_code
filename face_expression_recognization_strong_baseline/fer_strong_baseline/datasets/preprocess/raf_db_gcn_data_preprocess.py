import cv2
import numpy as np
import os
import shutil
import pdb
from tqdm import tqdm

img_path = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Image/aligned'
label_path = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/EmoLabel/list_patition_label.txt'
save_img_path = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/ForTorchImageFolder'

cls_num = ['surprised', 'fear', 'disgust', 'happy', 'sad', 'anger', 'normal']

with open(label_path, 'r') as f:
    img_and_label_lst = f.readlines()

img_lst = [x.replace('\n', '').replace("\t", ' ').split(' ')[0] for x in img_and_label_lst]
label_lst = [x.replace('\n', '').replace("\t", ' ').split(' ')[1] for x in img_and_label_lst]


for img_fn, label in tqdm(zip(img_lst, label_lst)):
    img_fn = img_fn.replace(".jpg", "_aligned.jpg")
    full_img_fn = os.path.join(img_path, img_fn)
    
    if 'train' in img_fn:
        save_folder = os.path.join(save_img_path, 'train', cls_num[int(label) - 1])
    else:
        save_folder = os.path.join(save_img_path, 'test', cls_num[int(label) - 1])
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_img_fn = os.path.join(save_folder, img_fn)
    shutil.copy(full_img_fn, save_img_fn)
    
    





