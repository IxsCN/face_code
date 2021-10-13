class MyLOGGER:
    def __init__(self, log_path):
        self.log_path = log_path
        print(self.log_path)
        with open(self.log_path, 'w') as f:
            f.write("trial log create!")
            f.write('\n')
    
    def write(self, *x):
        with open(self.log_path, 'a') as f:
            f.write("write:\n")
            for xx in x:
                if not isinstance(xx, str):
                    xx = str(xx)
                f.write(xx)
            f.write('\n')

    def info(self, *x):
        with open(self.log_path, 'a') as f:
            f.write("info:\n")
            for xx in x:
                if not isinstance(xx, str):
                    xx = str(xx)
                f.write(xx)
            f.write('\n')


    def debug(self, *x):
        with open(self.log_path, 'a') as f:
            f.write("debug:\n")
            for xx in x:
                if not isinstance(xx, str):
                    xx = str(xx)
                f.write(xx)
            f.write('\n')


mylogger = MyLOGGER('nni_trial_log.txt')  # my logger
err_f = open('nni_trial_error.txt', 'a') 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, '..')

import traceback
import logging
import nni
from nni.utils import merge_parameter
import shutil

import torch
from torchvision import datasets, transforms
import torch.utils.data as Data

import numpy as np

from fer_strong_baseline.loss import get_loss

from  fer_strong_baseline.utils.logger import TxtLogger
from  fer_strong_baseline.config.default_cfg import  get_fer_cfg_defaults

from  fer_strong_baseline.datasets import  get_fer_train_dataloader,get_fer_val_dataloader

from  fer_strong_baseline.models.build_model import  build_model

# from  fer_strong_baseline.loss.softmaxloss import  CrossEntropyLabelSmooth, WassersteinCrossEntropyLabelSmooth
from  fer_strong_baseline.optim.optimizer.nadam import Nadam
from  fer_strong_baseline.optim.lr_scheduler import WarmupMultiStepLR

from tools.simple_learner import  SimpleLearner

from  fer_strong_baseline.utils.common import setup_seed

from tools.build_learner import build_learner

from tqdm import tqdm

nnilogger = logging.getLogger('mnist_AutoML')   # nni logger

from PIL import Image

import  argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        # default='../configs/fer2013_res50_ce.yml',
                        # default='../configs/affectnet_res50_ce.yml',
                        default='../configs/rafdb_gcn_ce.yml',
                        type=str,
                        # required=True,
                        help="")
    parser.add_argument("--resume",
                        # default='/media/seven/7EF450DF596F8A46/weights/fer/affectnet/ce_wls/best.pth',
                        default='/media/seven/7EF450DF596F8A46/weights/fer/raf-db/gcn/2020_12_28_20_59_45/single_ce/best.pth',
                        type=str)
    return parser.parse_args()

def infer(cfg, resume, logger):
    assert resume, "not specify resume model path!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # val_loader = get_fer_val_dataloader(cfg)
    transform = {
    'train': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.RandomCrop(90),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.ToTensor(),          
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
    ]),

    'test': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
    ]),
    }

    data_dir = '/media/seven/7EF450DF596F8A46/data/fer_data/RAF-DB/basic/FrontForTorchImageFolder/'
    # data_dir = '/media/seven/7EF450DF596F8A46/data/fer_data/RAF-DB/basic/ForTorchImageFolder/'

    # pdb.set_trace()

    # train_data_set = datasets.ImageFolder(data_dir + 'train', transform['train'])
    # test_data_set = datasets.ImageFolder(data_dir + 'test', transform['test'])

    # image_datasets = {x: datasets.ImageFolder(
    #                 data_dir + x,
    #                 transform['test'])
    #                 for x in ['train', 'test']}

    class RAFDB_GCN_DATASET_FOR_INFER:
        def __init__(self, datadir, transform, split='train'):
            self.cls_num = ['surprised', 'fear', 'disgust', 'happy', 'sad', 'anger', 'normal']
            self.dataset = datasets.ImageFolder(data_dir + split, transform['test'])
            self.transform = transform['test']
            self.label_map = dict()

        def __len__(self):
            return len(self.dataset.imgs)

        def __getitem__(self, index):
            imgfn, folderlabel = self.dataset.imgs[index]  # folderlabel is defined by datasets.ImageFolder
            label_str = os.path.basename(os.path.dirname(imgfn))
            try:
                label = self.cls_num.index(label_str)
            except:
                print("wrong img:{}".format(imgfn))
                pdb.set_trace()
            
            if folderlabel not in self.label_map.keys():
                self.label_map.update({folderlabel:label})
            else:
                try:
                    assert self.label_map[folderlabel] == label
                except:
                    pdb.set_trace()

            src = Image.open(imgfn)
            img = self.transform(src)
            
            return img, folderlabel, index, imgfn

    trainset = RAFDB_GCN_DATASET_FOR_INFER(data_dir, transform, 'train')
    testset = RAFDB_GCN_DATASET_FOR_INFER(data_dir, transform, 'test')

    train_loader = Data.DataLoader(dataset = trainset, batch_size = cfg.TRAIN.batch_size, shuffle=False)
    test_loader = Data.DataLoader(dataset = testset, batch_size = 32, shuffle=False)

    model = build_model(cfg)

    state_dict = torch.load(resume)
    model.load_state_dict(state_dict)
    # model = model.cuda()

    model = model.to(device)
    # model = model.cuda()
    params = model.parameters()

    if cfg.TRAIN.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.learning_rate)
    elif cfg.TRAIN.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, cfg.TRAIN.learning_rate,
                                    momentum=cfg.TRAIN.momentum,
                                    weight_decay=cfg.TRAIN.weight_decay)
    elif cfg.TRAIN.optimizer == 'nadam':
        optimizer = Nadam(params=model.parameters(), lr=cfg.TRAIN.learning_rate)
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optimizer = RMSprop(params, cfg.TRAIN.learning_rate, alpha=0.99, eps=1e-8,
                                        weight_decay=cfg.TRAIN.weight_decay, momentum=0,centered=False)
    else:
        raise ValueError("Optimizer not supported.")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = get_loss(cfg, args, device, mylogger)

    # loss_fn = WassersteinCrossEntropyLabelSmooth(num_classes = cfg.MODEL.num_classes)
    # optimizer = Nadam(params=model.parameters(), lr=cfg.TRAIN.learning_rate)
  
    if cfg.LOSS.type == 'CE':
        model_save_dir = os.path.join(cfg.TRAIN.save_dir, 'single_ce')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        # loss_fn = CrossEntropyLabelSmooth(num_classes=cfg.MODEL.num_classes,
        #                                   use_label_smooth=cfg.LOSS.use_label_smooth,
        #                                   logger=logger)

        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = CrossEntropy()
    elif cfg.LOSS.type == 'CE_WLS':
        model_save_dir = os.path.join(cfg.TRAIN.save_dir, 'alpha_' + str(cfg.LOSS.alpha))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

    learner = build_learner(cfg, 
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        logger=mylogger,
        nnilogger=nnilogger,
        device=device,

        # scn param
        margin_1=cfg.TRAIN.margin_1,
        margin_2=cfg.TRAIN.margin_2,
        beta=cfg.TRAIN.beta,
        relabel_epoch=cfg.TRAIN.relabel_epoch,
        use_update_w=cfg.LOSS.use_update_w,
        update_w_start_epoch=cfg.LOSS.update_w_start_epoch,

        # kwarg param
        save_dir=model_save_dir,
        log_steps=cfg.TRAIN.log_steps,
        gradient_accum_steps=1,
        max_grad_norm=1.0,
        batch_to_model_inputs_fn=None,
        early_stop_n=cfg.TRAIN.early_stop_n)

    # learner.training(train_loader, val_loader, epoches=cfg.TRAIN.num_train_epochs)

    # acc = learner.validation(val_loader)

    split = "test"

    acc, all_preds, all_labels, all_img_fns = learner.validation(eval(split + "_loader"), need_result_list=True)
    # acc, all_preds, all_labels, all_img_fns = learner.validation(test_loader, need_result_list=True)

    print("test acc:", acc)

    cls_num = ['surprised', 'fear', 'disgust', 'happy', 'sad', 'anger', 'normal']
    save_path = os.path.join('/media/seven/7EF450DF596F8A46/data/fer_data/RAF-DB/basic/result_analysis', split)
    origin_path = "/media/seven/7EF450DF596F8A46/data/fer_data/RAF-DB/basic/Image/original"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_f_path = os.path.join(save_path, "result.txt")
    save_f = open(save_f_path, 'w')

    def rename_img_fn_to_origin_img_fn(img_fn):
        origin_base_img_fn = os.path.basename(img_fn).replace("yaw_0.0_"+split, split).replace("_vertical.jpg", ".jpg")
        origin_img_fn = os.path.join(origin_path, origin_base_img_fn)
        return origin_img_fn

    all_origin_img_fns = np.array([rename_img_fn_to_origin_img_fn(x) for x in all_img_fns.tolist()])
    sort_idx = np.argsort(all_origin_img_fns)

    all_origin_img_fns = all_origin_img_fns[sort_idx]
    all_img_fns = all_img_fns[sort_idx]
    all_preds = all_preds[sort_idx]
    all_labels = all_labels[sort_idx]

    for folderpred, folderlabel, img_fn in tqdm(zip(all_preds, all_labels, all_img_fns)):
        
        dataset = eval(split + "set")
        pred = dataset.label_map[folderpred]
        label = dataset.label_map[folderlabel]
        if pred != label:
            # img_fn.replace("_aligned.jpg", ".jpg")
            # print(img_fn)
            origin_img_fn = rename_img_fn_to_origin_img_fn(img_fn)

            if not os.path.exists(origin_img_fn):
                import pdb; pdb.set_trace()
            
            origin_base_img_fn = os.path.basename(origin_img_fn)
            origin_base_img_fn_without_subfix = origin_base_img_fn.replace(".jpg", "")
            shutil.copy(img_fn, save_path + "/front_{}_pred-{}_gt-{}.jpg".format(origin_base_img_fn_without_subfix, cls_num[pred], cls_num[label]))
            shutil.copy(origin_img_fn, save_path + "/origin_{}_pred-{}_gt-{}.jpg".format(origin_base_img_fn_without_subfix, cls_num[pred], cls_num[label]))
            save_f.write("img_fn:{},pred:{},label:{}\n".format(img_fn, pred, label))

    save_f.close()

if __name__ == '__main__':
    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    cfg.merge_from_file(args.config)
    logger = TxtLogger(cfg.TRAIN.save_dir + "/val_logger.txt")
    logger.write('using config: ', args.config.strip())
    logger.write(cfg)
    infer(cfg, args.resume, logger)
