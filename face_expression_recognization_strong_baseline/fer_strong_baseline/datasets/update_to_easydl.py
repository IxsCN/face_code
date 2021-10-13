# -*- coding: utf-8 -*-
import requests
import base64
import json
import pdb
import array


class UpdateDatasetToEasyDL:
    def __init__(self):
        pass

    def get_access_token(self):

        ###########################
        # post请求，获取AccessToken #
        ###########################
        """
        grant_type： 必须参数，固定为client_credentials；
        client_id： 必须参数，应用的API Key；
        client_secret： 必须参数，应用的Secret Key；

        例子:
            https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=Va5yQRHlA4Fq5eR3LT0vuXV4&client_secret=0rDSjzQ20XUj5itV6WRtznPQSzr5pVw2&
        """
        grant_type = 'client_credentials'
        client_id = 'AGumYKu7hSHyeABCae0KVySp'
        client_secret = 'D7vqCnr6q42ER2Vy4F5LybQ5RNrQ3Pp3'
        get_token_url = 'https://aip.baidubce.com/oauth/2.0/token'+'?grant_type=' + grant_type + \
        '&client_id=' + client_id + '&client_secret=' + client_secret
        access_token_response = requests.get(get_token_url)
        # print(access_token_response.json())
        try:
            self.access_token = access_token_response.json()['access_token']
            return True
        except Exception as e:
            return False

    def create_dataset(self):
        #####################
        # post请求,数据集创建 #
        #####################

        """
        成功:
        {
            "dataset_id": 14611,
            "log_id": 1928365800
        }
        失败（数据集已存在）:
        {'error_code': 406003, 'error_msg': 'dataset already exists', 'log_id': 2113921487}

        """
        url = 'https://aip.baidubce.com/rpc/2.0/easydl/pro/dataset/create' + '?access_token='+ self.access_token  #这里也需要改动
        # 请求头
        headers = {
            'Content-Type' : 'application/json'
        }

        # 请求内容
        body = {
        "type":"IMAGE_CLASSIFICATION",
        "template_type":"IMAGE_CLASSIFICATION_ONE_LABEL",
        "dataset_name":"ferdataset_exp"
        }

        get_dataset_id_and_log_id = requests.post(url,headers= headers, data=json.dumps(body))
        print(get_dataset_id_and_log_id.json())
        try:
            self.dataset_id = get_dataset_id_and_log_id.json()['dataset_id']
            return True
        except Exception as e:
            return False

    def upload_sample(self, img_fn, label, label_name):

        ###########
        # 上传图片 #
        ###########
        url = 'https://aip.baidubce.com/rpc/2.0/easydl/pro/dataset/addentity' + '?access_token='+ self.access_token  #这里也需要改动

        #打开图片文件
        with open(img_fn , 'rb') as file:
            pic = base64.b64encode(file.read()).decode()

        label = 0

        # 请求头
        headers = {
            'Content-Type' : 'application/json'
        }


        # 请求内容
        body = {
        "dataset_id":self.dataset_id,  # 官网能查到
        "type":"IMAGE_CLASSIFICATION",
        "template_type":"IMAGE_CLASSIFICATION_ONE_LABEL",
        "entity_content":pic,
        "labels": [{"name": "Action"}],
        "label_name":label_name,
        # "append_label":True
        }

        get_dataset_id_and_log_id = requests.post(url,headers= headers, data=json.dumps(body))
        print(get_dataset_id_and_log_id.json())



if __name__ == '__main__':
    import os
    from tqdm import tqdm
    img_dir = '/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated/Face_Images_Small'
    train_lst_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated_file_lists/training.csv"
    val_lst_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated_file_lists/validation.csv"
    affect_cls_num = ['normal', 'happy', 'sad', 'surprised', 'fear', 'disgust', 'anger', 'contempt']

    error_img_lst = [
            "2/9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg",
            "994/8fb7a123f36ae89be4bf72e9fa77d1434a29135f3342f62fb55c7aee.jpg",
        ]

    affect_cls_num = ['normal', 'happy', 'sad', 'surprised',
                                 'fear', 'disgust', 'anger', 'contempt']

    datauploader = UpdateDatasetToEasyDL()
    if not datauploader.get_access_token():
        exit(-1)
    if not datauploader.create_dataset():
        exit(-1)
    # datauploader.dataset_id = 136812

    with open(val_lst_file, 'r') as f:
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
        try:
            datauploader.upload_sample(img_path, expression, affect_cls_num[expression])
        except Exception as e:
            continue
