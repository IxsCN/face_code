from torch.utils.data import Dataset
from fer_strong_baseline.datasets.aug import fer_test_aug,fer_train_aug
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from fer_strong_baseline.datasets.aug import flip_image, add_gaussian_noise, CropByScale, CaffeCrop
import random
from PIL import Image
from fer_strong_baseline.utils.common import load_imgs
import torchvision.transforms.functional as F
import cv2

class ImbalancedRAF_DB_Dataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        if self.is_train:
            self.data_lst = cfg.DATA.train_label_path
            self.aug = fer_train_aug(cfg.DATA.input_size,
                                     cfg.DATA.crop_residual_pix)
            caffe_crop = CaffeCrop('train')
            self.transform = [
                # center down
                # box = (int(0.125*width), int(0.25*height), int(0.875*width), int(1*height))
                transforms.Compose([
                    transforms.ToPILImage(),
                    CropByScale(0.125, 0.25, 0.875, 1),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),
                # top left
                # box = (int(0*width), int(0*height), int(0.75*width), int(0.75*height))
                transforms.Compose([
                    transforms.ToPILImage(),
                    CropByScale(0, 0, 0.75, 0.75),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),
                # top right
                # box = (int(0.25*width), int(0*height), int(1*width), int(0.75*height))
                transforms.Compose([
                    transforms.ToPILImage(),
                    CropByScale(0.25, 0, 1, 0.75),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),
                # first center crop
                # box = (int(0.05*width), int(0.05*height), int(0.95*width), int(0.95*height))
                transforms.Compose([
                    transforms.ToPILImage(),
                    CropByScale(0.05, 0.05, 0.95, 0.95),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),
                # second center crop
                # box = (int(0.1*width), int(0.1*height), int(0.9*width), int(0.9*height))
                transforms.Compose([
                    transforms.ToPILImage(),
                    CropByScale(0.1, 0.1, 0.9, 0.9),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),

                # third center crop
                # box = (int(0.125*width), int(0.125*height), int(0.875*width), int(0.875*height))
                transforms.Compose([
                    transforms.ToPILImage(),
                    CropByScale(0.125, 0.125, 0.875, 0.875),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),
                # fouth center top
                # box = (int(0.125*width), int(0*height), int(0.875*width), int(0.75*height))
                transforms.Compose([
                    transforms.ToPILImage(),
                    CropByScale(0.125, 0, 0.875, 0.75),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),

                # img original
                transforms.Compose([
                    transforms.ToPILImage(),
                    caffe_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(scale=(0.02, 0.25))]),
            ]
        else:
            caffe_crop = CaffeCrop('test')
            self.data_lst = cfg.DATA.val_label_path
            self.aug = fer_test_aug(cfg.DATA.input_size)
            self.transform = [transforms.Compose([
                transforms.ToPILImage(),
                caffe_crop,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])]

        if self.is_train:
            self.random_erasing = transforms.RandomErasing(scale=(0.02, 0.25))
        else:
            self.random_erasing = None

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        IMAGEPATHES_COLUMN = 0
        df = pd.read_csv(self.data_lst, sep=' ', header=None)

        if self.is_train:
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]

        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.labels = dataset.iloc[:,
                      LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.labels = np.array(self.labels)

        # use raf aligned images for training/testing
        self.samples = [os.path.join(cfg.DATA.img_dir, x.split('\n')[0].split(' ')[0].replace(".jpg", "_aligned.jpg"))
                        for x in file_names]

        self.rafdb_cls_num = ['surprised', 'fear', 'disgust', 'happy',
                              'sad', 'anger', 'normal']

        self.aug_func = [flip_image, add_gaussian_noise]

    def __len__(self):
        assert len(self.samples) == len(self.labels)
        return len(self.samples)

    def __getitem__(self, idx):

        path = self.samples[idx]
        image = cv2.imread(path)
        h, w, _ = image.shape
        # gui = image.copy()
        # print(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.labels[idx]
        landmark_path = os.path.join(os.path.dirname(os.path.dirname(self.data_lst)), 'Annotation', 'manual',
                                     path.replace('_aligned.jpg', '_manu_attri.txt'))

        with open(landmark_path, 'r') as f:
            sample = f.readlines()
        sample = [[float(x) for x in p.split('\n')[0].replace(' ', '\t').split('\t')] for p in sample]
        landmark = sample[:5]

        gender = sample[5]
        race = sample[6]
        age = sample[7]

        # for p in landmark:
        #     cv2.circle(gui, (int(p[0]), int(p[1])), 3, (0,255,0), 3)
        # cv2.namedWindow('gui', 0)
        # cv2.imshow('gui', gui)
        # cv2.waitKey(0)
        # pdb.set_trace()
        # landmark =[[p[0]/h*224, p[1]/w*224] for p in landmark]

        # augmentation
        if self.is_train:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)
        # image = self.aug(image=image)['image']

        # if self.is_train:
        image1, image2, image3, image4, image5, image6, image7, image8 = [self.transform[i](image) for i in range(8)]

        return image1, label, image2, label, image3, label, image4, label, \
               image5, label, image6, label, image7, label, image8, label,


class MsCelebDataset(Dataset):
    def __init__(self, img_dir, image_list_file, label_file, transform=None):
        self.imgs_first, self.imgs_second, self.imgs_third, self.imgs_forth,\
        self.imgs_fifth, self.imgs_sixth, self.imgs_seventh, self.imgs_eigth = \
            load_imgs(img_dir, image_list_file, label_file)
        self.transform = transform

    def __getitem__(self, index):
        # pdb.set_trace()
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        # img_first = cv2.imread(path_first)
        # center down
        height = img_first.size[0]
        width = img_first.size[1]
        box = (int(0.125*width), int(0.25*height), int(0.875*width), int(1*height))
        img_first = img_first.crop(box)
        # img_first = img_first[int(0.25*height):int(1*height),int(0.125*width):int(0.875*width)]
        if self.transform is not None:
            img_first = self.transform(img_first)
        # top left
        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        box = (int(0*width), int(0*height), int(0.75*width), int(0.75*height))
        # img_second = img_second[int(0*height):int(0.75*height),int(0*width):int(0.75*width)]
        img_second = img_second.crop(box)
        if self.transform is not None:
            img_second = self.transform(img_second)

        # top right

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        box = (int(0.25*width), int(0*height), int(1*width), int(0.75*height))
        # img_third = img_third[int(0*height):int(0.75*height),int(0.25*width):int(1*width)]
        img_third = img_third.crop(box)
        if self.transform is not None:
            img_third = self.transform(img_third)

        # first center crop

        path_forth, target_forth = self.imgs_forth[index]
        img_forth = Image.open(path_forth).convert("RGB")
        box = (int(0.05*width), int(0.05*height), int(0.95*width), int(0.95*height))
        # img_forth = img_forth[int(0.05*height):int(0.95*height),int(0.05*width):int(0.95*width)]
        img_forth = img_forth.crop(box)
        if self.transform is not None:
            img_forth = self.transform(img_forth)





        # second center crop
        path_fifth, target_fifth = self.imgs_fifth[index]
        img_fifth = Image.open(path_fifth).convert("RGB")
        box = (int(0.1*width), int(0.1*height), int(0.9*width), int(0.9*height))
        # img_fifth = img_fifth[int(0.1*height):int(0.9*height),int(0.1*width):int(0.9*width)]
        img_fifth = img_fifth.crop(box)
        if self.transform is not None:
            img_fifth = self.transform(img_fifth)



       # img original
        path_sixth, target_sixth = self.imgs_sixth[index]
        img_sixth = Image.open(path_sixth).convert("RGB")
        box = (int(0.125*width), int(0.125*height), int(0.875*width), int(0.875*height))
        img_sixth = img_sixth.crop(box)
        if self.transform is not None:
            img_sixth = self.transform(img_sixth)
        # pdb.set_trace()


        # center top

        path_seventh, target_seventh = self.imgs_seventh[index]
        img_seventh = Image.open(path_seventh).convert("RGB")
        box = (int(0.125*width), int(0*height), int(0.875*width), int(0.75*height))
        img_seventh = img_seventh.crop(box)
        if self.transform is not None:
            img_seventh = self.transform(img_seventh)

        # third center crop

        path_eigth, target_eigth = self.imgs_eigth[index]
        img_eigth = Image.open(path_eigth).convert("RGB")
        #box = (int(0.125*width), int(0.125*height), int(0.875*width), int(0.875*height))
        #img_eigth = img_eigth.crop(box)
        if self.transform is not None:
            img_eigth = self.transform(img_eigth)
        #pdb.set_trace()

        return img_first, target_first ,img_second,target_second,img_third,target_third,img_forth,target_forth, img_fifth, target_fifth, img_sixth, target_sixth, img_seventh, target_seventh, img_eigth, target_eigth
    def __len__(self):
        return len(self.imgs_first)
