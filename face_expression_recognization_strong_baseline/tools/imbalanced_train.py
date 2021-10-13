import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, '..')
import torch
from fer_strong_baseline.utils.logger import TxtLogger
from fer_strong_baseline.config.default_cfg import get_fer_cfg_defaults
from fer_strong_baseline.datasets import get_fer_train_dataloader,get_fer_val_dataloader
from fer_strong_baseline.loss import get_loss

from fer_strong_baseline.models.build_model import build_model

from fer_strong_baseline.loss.softmaxloss import (
    CrossEntropy,
    CrossEntropyLabelSmooth,
    WassersteinCrossEntropyLabelSmooth)

from fer_strong_baseline.optim.optimizer.nadam import Nadam
from fer_strong_baseline.optim.lr_scheduler import WarmupMultiStepLR

from tools.simple_learner import SimpleLearner
from tools.scn_learner import ScnLearner
from fer_strong_baseline.utils.common import setup_seed

import datetime
import pdb
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        #default='../configs/fer2013_res50_ce.yml',
                        # default='../configs/fer2013_res_ce_wls.yml',
                        #  default='../configs/rafdb_scnres18_ce.yml',
                          default='../configs/rafdb_scnres18_ce_wls.yml',
                         # default='../configs/rafdb_res18_ce.yml',
                         # default='../configs/rafdb_scnres18_ce_wls.yml',
                         # default='../configs/affectnet_scnres18_ce_wls.yml',
                        type=str,
                        # required=True,
                        help="")
    parser.add_argument("--resume",
                        # default='',
                        default='/media/yz/62C9BA4E5D826344/weights/fer/raf_db/scn/ce/2020_10_29_01_05_32/alpha_0.0125/epoch8_acc0.869.pth',
                        #default='/media/seven/7EF450DF596F8A46/weights/fer/fer_2013/ce_wls/best.pth',
                        # default='/media/seven/7EF450DF596F8A46/weights/fer/affectnet/ce_wls/best.pth',
                        type=str)
    return parser.parse_args()

def train(cfg, resume, logger):
    train_loader = get_fer_train_dataloader(cfg)
    val_loader = get_fer_val_dataloader(cfg)

    model = build_model(cfg)

    if resume:
        state_dict = torch.load(resume)['model_state_dict']
        model.load_state_dict(state_dict)
    model = model.cuda()
    params = model.parameters()

    if cfg.TRAIN.optimizer == 'adam':
        # optimizer = torch.optim.Adam(params, weight_decay=cfg.TRAIN.weight_decay)
        optimizer = torch.optim.Adam([
                {'params': model.features.parameters(), 'lr':1e-4},
                # {'params': model.fc.parameters(), 'lr': 1e-3},
                # {'params': model.alpha.parameters(), 'lr': 1e-5}
            ], weight_decay=cfg.TRAIN.weight_decay)
    elif cfg.TRAIN.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, cfg.TRAIN.lr,
                                    momentum=cfg.TRAIN.momentum,
                                    weight_decay=cfg.TRAIN.weight_decay)
    elif cfg.TRAIN.optimizer == 'nadam':
        optimizer = Nadam(params=model.parameters(), lr=cfg.TRAIN.learning_rate)

    else:
        raise ValueError("Optimizer not supported.")

    # lr_scheduler = WarmupMultiStepLR(
    #     optimizer,
    #     milestones=cfg.TRAIN.milestones,
    #     gamma=cfg.TRAIN.lr_decay_gamma
    # )

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = get_loss(cfg, logger)

    if cfg.LOSS.type == 'CE':
        model_save_dir = os.path.join(cfg.TRAIN.save_dir, 'single_ce')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        # loss_fn = CrossEntropyLabelSmooth(num_classes=cfg.MODEL.num_classes,
        #                                   use_label_smooth=cfg.LOSS.use_label_smooth,
        #                                   logger=logger)

        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = CrossEntropy()

        if not cfg.TRAIN.use_scn_group:
            learner = SimpleLearner(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                logger=logger,
                save_dir=model_save_dir,
                log_steps=cfg.TRAIN.log_steps,
                device_ids=cfg.TRAIN.device_ids,
                gradient_accum_steps=1,
                max_grad_norm=1.0,
                batch_to_model_inputs_fn=None,
                early_stop_n=cfg.TRAIN.early_stop_n)
            learner.train(train_loader, val_loader, epoches=cfg.TRAIN.num_train_epochs)
        else:
            learner = ScnLearner(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                margin_1=cfg.TRAIN.margin_1,
                margin_2=cfg.TRAIN.margin_2,
                beta=cfg.TRAIN.beta,
                relabel_epoch=cfg.TRAIN.relabel_epoch,
                use_update_w=cfg.LOSS.use_update_w,
                update_w_start_epoch=cfg.LOSS.update_w_start_epoch,
                logger=logger,
                save_dir=model_save_dir,
                log_steps=cfg.TRAIN.log_steps,
                device_ids=cfg.TRAIN.device_ids,
                gradient_accum_steps=1,
                max_grad_norm=1.0,
                batch_to_model_inputs_fn=None,
                early_stop_n=cfg.TRAIN.early_stop_n)
            learner.train(train_loader, val_loader, epoches=cfg.TRAIN.num_train_epochs)

    elif cfg.LOSS.type == 'CE_WLS':
        if isinstance(cfg.LOSS.alpha, list):
            for alpha in cfg.LOSS.alpha:
                model_save_dir = os.path.join(cfg.TRAIN.save_dir, 'alpha_' + str(alpha))
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                if not cfg.TRAIN.use_scn_group:
                    learner = SimpleLearner(
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        logger=logger,
                        save_dir=model_save_dir,
                        log_steps=cfg.TRAIN.log_steps,
                        device_ids=cfg.TRAIN.device_ids,
                        gradient_accum_steps=1,
                        max_grad_norm=1.0,
                        batch_to_model_inputs_fn=None,
                        early_stop_n=cfg.TRAIN.early_stop_n)

                    learner.train(train_loader, val_loader, epoches=cfg.TRAIN.num_train_epochs, alpha=alpha)
                else:
                    learner = ScnLearner(
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        margin_1=cfg.TRAIN.margin_1,
                        margin_2=cfg.TRAIN.margin_2,
                        beta=cfg.TRAIN.beta,
                        relabel_epoch=cfg.TRAIN.relabel_epoch,
                        use_update_w=cfg.LOSS.use_update_w,
                        update_w_start_epoch=cfg.LOSS.update_w_start_epoch,
                        logger=logger,
                        save_dir=model_save_dir,
                        log_steps=cfg.TRAIN.log_steps,
                        device_ids=cfg.TRAIN.device_ids,
                        gradient_accum_steps=1,
                        max_grad_norm=1.0,
                        batch_to_model_inputs_fn=None,
                        early_stop_n=cfg.TRAIN.early_stop_n)

                    learner.train(train_loader, val_loader, epoches=cfg.TRAIN.num_train_epochs, alpha=alpha)


if __name__ == '__main__':
    #setup_seed(1)
    cfg = get_fer_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    cfg.merge_from_file(args.config)

    systime_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    cfg.TRAIN.save_dir = os.path.join(cfg.TRAIN.save_dir, systime_str)

    # def myMkdirs(path):
    #     dirname = os.path.dirname(path)
    #     if os.path.exists(dirname):
    #         os.mkdir(path, 755)
    #     else:
    #         myMkdirs(dirname)
    #         os.mkdir(path, 755)

    os.makedirs(cfg.TRAIN.save_dir)
    # myMkdirs(cfg.TRAIN.save_dir)
    logger = TxtLogger(cfg.TRAIN.save_dir + "/logger.txt")
    logger.write('using config: ',args.config.strip())
    logger.write(cfg)
    train(cfg, args.resume, logger)
