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

import torch
from fer_strong_baseline.utils.logger import TxtLogger
from fer_strong_baseline.utils.common import dict_to_object

from fer_strong_baseline.config.default_cfg import get_fer_cfg_defaults
from fer_strong_baseline.datasets import get_fer_train_dataloader,get_fer_val_dataloader
from fer_strong_baseline.loss import get_loss

from fer_strong_baseline.models.build_model import build_model

from fer_strong_baseline.optim.optimizer.nadam import Nadam
from torch.optim.rmsprop import RMSprop
from fer_strong_baseline.optim.lr_scheduler import WarmupMultiStepLR

# from tools.simple_learner import SimpleLearner
# from tools.scn_learner import ScnLearner
from fer_strong_baseline.utils.common import setup_seed

from tools.build_learner import build_learner

from torchvision import datasets, transforms
import torch.utils.data as Data
import datetime
import pdb
import argparse

nnilogger = logging.getLogger('mnist_AutoML')   # nni logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        #default='../configs/fer2013_res50_ce.yml',
                        # default='../configs/fer2013_res_ce_wls.yml',
                        #  default='../configs/affectnet_scnres18_ce.yml',
                        #  default='../configs/rafdb_scnres18_gaussian_ce.yml',
                          # default='../configs/rafdb_scnres18_ce_wls.yml',
                         # default='../configs/rafdb_res18_ce.yml',
                         # default='../configs/rafdb_scnres18_ce_wls.yml',
                        # default='../configs/rafdb_scnres18_ce.yml',
                        default='../configs/rafdb_gcn_ce.yml',
                        #  default='../configs/rafdb_landmarkcropres18_ce.yml',
                         # default='../configs/affectnet_scnres18_ce_wls.yml',
                        type=str,
                        # required=True,
                        help="")
    parser.add_argument("--resume",
                        default="",
                        # default="/home/yz/workspace/project/face_expression_recognization_strong_baseline/best_test_para.pth.tar",
                        # default=r"/media/yz/62C9BA4E5D826344/weights/fer/raf_db/scn/ce/commit 540a9f7模型1的实验，用混淆矩阵更新w，行归一化，只调wloss_0.8673/alpha_0.0125/epoch13_acc0.8673.pth",
                        # default='/media/yz/62C9BA4E5D826344/weights/fer/raf_db/scn/ce/isRecord/commit b0fba58e_ce_without_weight_decay_without_branchx2_0.867/single_ce/epoch44_acc0.867.pth',
                        # default='/media/yz/62C9BA4E5D826344/weights/fer/raf_db/scn/ce/isRecord/commit cfcd5592_ce_with_weight_decay_no_branchx2_0.8698/single_ce/epoch49_acc0.8699.pth',
                        # default='/media/yz/62C9BA4E5D826344/weights/fer/raf_db/scn/ce/gaussian_trainingset_100_val_0.8501_default_lr/single_ce/epoch17_acc0.8501.pth',
                        # default='/media/yz/62C9BA4E5D826344/weights/fer/raf_db/scn/ce/2020_10_29_01_05_32/alpha_0.0125/epoch8_acc0.869.pth',
                        #default='/media/seven/7EF450DF596F8A46/weights/fer/fer_2013/ce_wls/best.pth',
                        # default='/media/seven/7EF450DF596F8A46/weights/fer/affectnet/ce_wls/best.pth',
                        type=str)

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.00035, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--alpha', type=float, default=0.9, metavar='N',
                        help='alpha * wloss + celoss')

    parser.add_argument('--w00', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w01', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w02', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w03', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w04', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w05', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w06', type=float, default=1.0, metavar='N',
                        help='element of matrix W')

    parser.add_argument('--w12', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w13', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w14', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w15', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w16', type=float, default=1.0, metavar='N',
                        help='element of matrix W')

    parser.add_argument('--w23', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w24', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w25', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w26', type=float, default=1.0, metavar='N',
                        help='element of matrix W')

    parser.add_argument('--w34', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w35', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    parser.add_argument('--w36', type=float, default=1.0, metavar='N',
                        help='element of matrix W')
    
    parser.add_argument('--w45', type=float, default=1.0, metavar='N',
                        help='element of matrix W')                    
    parser.add_argument('--w46', type=float, default=1.0, metavar='N',
                        help='element of matrix W')

    parser.add_argument('--w56', type=float, default=1.0, metavar='N',
                        help='element of matrix W')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    return parser.parse_args()

def train(cfg, args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    

    # train_loader = get_fer_train_dataloader(cfg)
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

    # pdb.set_trace()

    # train_data_set = datasets.ImageFolder(data_dir + 'train', transform['train'])
    # test_data_set = datasets.ImageFolder(data_dir + 'test', transform['test'])

    image_datasets = {x: datasets.ImageFolder(
                    data_dir + x,
                    transform[x])
                    for x in ['train', 'test']}

    train_loader = Data.DataLoader(dataset = image_datasets['train'], batch_size = cfg.TRAIN.batch_size, shuffle=True)
    val_loader = Data.DataLoader(dataset = image_datasets['test'], batch_size = 32)

    model = build_model(cfg)

    if args.resume:
        mylogger.info("use_cuda:", use_cuda)
        if use_cuda:
            state_dict = torch.load(args.resume)['state_dict']
            # state_dict = torch.load(args.resume)
            # state_dict.pop("fc.weight")
            # state_dict.pop("fc.bias")
            model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(args.resume, map_location='cpu')['state_dict']
            # state_dict = torch.load(args.resume, map_location='cpu')
            # state_dict.pop("fc.weight")
            # state_dict.pop("fc.bias")
            model.load_state_dict(state_dict)
    
    model = model.to(device)
    # model = model.cuda()
    params = model.parameters()

    # for para in model.parameters():
    #     para.requires_grad = False

    # for para in model.fc.parameters():
    #     para.requires_grad = True

    # for para in model.alpha.parameters():
    #     para.requires_grad = True

    # def set_bn_eval(m):
    #     classname = m.__class__.__name__
    #     if classname.find('BatchNorm') != -1:
    #         m.eval()
    #         m.momentum = 0

    # model.apply(set_bn_eval)

    


    # for para in model.conv1.parameters():
    #     para.requires_grad = False

    # for para in model.parameters():
    #     para.requires_grad = True

    if cfg.TRAIN.optimizer == 'adam':
        # optimizer = torch.optim.Adam(params, lr=1e-11, betas=(0.9, 0.999), eps=1e-8,
        #                              weight_decay=cfg.TRAIN.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        
        optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.learning_rate)


        # optimizer = torch.optim.Adam(params, weight_decay=cfg.TRAIN.weight_decay)
        # optimizer = torch.optim.Adam(params, lr = cfg.TRAIN.learning_rate, weight_decay=cfg.TRAIN.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
        #                                         weight_decay=cfg.TRAIN.weight_decay)
 

        # '''
        # self.features_cbr = nn.Sequential(*list(resnet.children())[:4])  # after avgpool 512x1
        # self.features_stage1 = nn.Sequential(*list(resnet.children())[4])
        # self.features_stage2 = nn.Sequential(*list(resnet.children())[5])
        # self.features_stage3 = nn.Sequential(*list(resnet.children())[6])
        # self.features_stage4 = nn.Sequential(*list(resnet.children())[7])
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # '''

        # optimizer = torch.optim.Adam([
        #     # {'params':model.parameters(())}
        #         {'params': model.features_cbr.parameters(), 'lr':1e-5},
        #         {'params': model.features_stage1.parameters(), 'lr':1e-5},
        #         {'params': model.features_stage2.parameters(), 'lr':1e-5},
        #         {'params': model.features_stage3.parameters(), 'lr':1e-5},
        #         {'params': model.features_stage4.parameters(), 'lr':1e-5},
        #         {'params': model.fc.parameters(), 'lr': 1e-3},
        #         {'params': model.alpha.parameters(), 'lr': 1e-5}
        #     ], weight_decay=cfg.TRAIN.weight_decay)


        # # optimizer = torch.optim.Adam([
        # #     # {'params':model.parameters(())}
        # #     ], weight_decay=cfg.TRAIN.weight_decay)


    elif cfg.TRAIN.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, cfg.TRAIN.learning_rate,
                                    momentum=cfg.TRAIN.momentum,
                                    weight_decay=cfg.TRAIN.weight_decay)

    elif cfg.TRAIN.optimizer == 'nadam':
        optimizer = Nadam(params=model.parameters(), lr=cfg.TRAIN.learning_rate)

# self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optimizer = RMSprop(params, cfg.TRAIN.learning_rate, alpha=0.99, eps=1e-8,
                                        weight_decay=cfg.TRAIN.weight_decay, momentum=0,centered=False)

    else:
        raise ValueError("Optimizer not supported.")

    # lr_scheduler = WarmupMultiStepLR(
    #     optimizer,
    #     milestones=cfg.TRAIN.milestones,
    #     gamma=cfg.TRAIN.lr_decay_gamma
    # )

    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = get_loss(cfg, args, device, mylogger)

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

    learner.training(train_loader, val_loader, epoches=cfg.TRAIN.num_train_epochs)


if __name__ == '__main__':
    try:
        cfg = get_fer_cfg_defaults()
        args = get_args()

        setup_seed(args.seed)

        if args.config.endswith('\r'):
            args.config = args.config[:-1]
        cfg.merge_from_file(args.config)

        #########################################
        # note: use below code only use nni!!!! #
        #########################################
        tuner_params = nni.get_next_parameter()
        nnilogger.debug(tuner_params)
        mylogger.debug("tuner_params:",tuner_params)
        args_dict = vars(args)
        mylogger.debug("vars(args):", args_dict)
        # args = dict_to_object(merge_parameter(vars(args), tuner_params))
        for k,v in tuner_params.items():
            if k in args_dict:
                args_dict[k] = v
                # mylogger.debug("update {}:{}}".format(str(k),str(v)))
        args = dict_to_object(args_dict)
        mylogger.debug("vars(args):", args_dict)
        mylogger.write("这是一个trial的print！")

        # pdb.set_trace()
        cfg.TRAIN.learning_rate = args.lr
        cfg.TRAIN.batch_size = args.batch_size
        cfg.TRAIN.momentum = args.momentum
        cfg.LOSS.alpha = args.alpha
        #########################################
        # end                                   #
        #########################################

        systime_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        cfg.TRAIN.save_dir = os.path.join(cfg.TRAIN.save_dir, systime_str)

        os.makedirs(cfg.TRAIN.save_dir)

        mylogger.write('using config: ',args.config.strip())
        mylogger.write(cfg)
        train(cfg, args)

    # except Exception as exception:
    except:
        error_type, error_value, error_trace = sys.exc_info()  # 错误类型、错误内容、traceback对象

        print(error_type)
        print(error_value)

        traceback.print_tb(error_trace)
        traceback.print_tb(error_trace, file=err_f)  # 将信息存入文件
        err_f.close()
        # nnilogger.exception(exception)
        raise
