from  torch.utils.data import  Dataset,DataLoader
from fer_strong_baseline.datasets.ExpW import  ExpW_Dataset
from  fer_strong_baseline.datasets.AffectNet import  AffectNet_Dataset
from fer_strong_baseline.datasets.FER2013 import FER2013Dataset, FER2013FolderDataset
from fer_strong_baseline.datasets.RAF_DB import RafDataSet

from fer_strong_baseline.datasets.ImbalancedDatasetSampling.ImbalancedRAF_DB import ImbalancedRAF_DB_Dataset
from fer_strong_baseline.datasets.RAF_DB_gaussian import RafGaussianDataSet, RafGaussianOfflineDataSet

from fer_strong_baseline.datasets.ImbalancedDatasetSampling.ImbalancedDatasetSampler import ImbalancedDatasetSampler

from fer_strong_baseline.datasets.RAF_DB_landmark_crop import LandmarkCropRafDataSet

from fer_strong_baseline.datasets.RAF_DB_copy import RafDataSet_white_gaussian


__mapping_dataset = {
    'ExpW': ExpW_Dataset,
    'FER2013': FER2013Dataset, # FER2013Dataset,
    'AffectNet':AffectNet_Dataset,
    'RAF_DB':RafDataSet,
    'RafDataSet_white_gaussian':RafDataSet_white_gaussian,

    'RAF_DB_gaussian':RafGaussianDataSet,
    'RAF_DB_gaussian_offline':RafGaussianOfflineDataSet,

    'LandmarkCrop_RAF_DB': LandmarkCropRafDataSet,

    'Imbalanced_RAF_DB':ImbalancedRAF_DB_Dataset,
}

def get_dataset(cfg, is_train = True):
    if cfg.DATA.dataset_type not in __mapping_dataset.keys():
        raise  NotImplementedError('Dataset Type not supported!')
    return  __mapping_dataset[cfg.DATA.dataset_type](
        cfg,
        is_train = is_train
    )

def get_fer_train_dataloader(cfg):
    # if cfg.DATA.dataset_type == 'RAF_DB':
    #     from torchvision import transforms
    #     data_transforms = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225]),
    #         transforms.RandomErasing(scale=(0.02, 0.25))])
    #
    #     train_dataset = RafDataSet(cfg.DATA.train_label_path, phase='train', transform=data_transforms, basic_aug=True)
    #     print('Train set size:', train_dataset.__len__())
    #     import torch
    #     train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                                batch_size = cfg.TRAIN.batch_size,
    #                                                num_workers = cfg.SYSTEM.NUM_WORKERS,
    #                                                shuffle = True,
    #                                                pin_memory = True,
    #                                                drop_last=True)
    #     return train_loader

    ds = get_dataset(cfg, True)
    if 'Imbalanced' in cfg.DATA.dataset_type:
        dloader = DataLoader(
            ds,
            # sampler=ImbalancedDatasetSampler(ds),
            batch_size = cfg.TRAIN.batch_size,
            num_workers = cfg.SYSTEM.NUM_WORKERS,
            shuffle=True,
            drop_last=True
        )
    else:
        dloader = DataLoader(
            ds,
            # sampler=ImbalancedDatasetSampler(ds),
            batch_size=cfg.TRAIN.batch_size,
            num_workers=cfg.SYSTEM.NUM_WORKERS,
            shuffle=True,
            drop_last=True
        )
    return dloader

def get_fer_val_dataloader(cfg):
    # if cfg.DATA.dataset_type == 'RAF_DB':
    #     from torchvision import transforms
    #     data_transforms_val = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])])
    #     val_dataset = RafDataSet(cfg.DATA.val_label_path, phase='test', transform=data_transforms_val)
    #     print('Validation set size:', val_dataset.__len__())
    #     import torch
    #     val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                              batch_size=cfg.TRAIN.batch_size,
    #                                              num_workers=cfg.SYSTEM.NUM_WORKERS,
    #                                              shuffle=False,
    #                                              pin_memory=True,
    #                                              drop_last=True)
    #     return val_loader

    ds = get_dataset(cfg, False)
    if 'Imbalanced' in cfg.DATA.dataset_type:
        dloader = DataLoader(
            ds,
            batch_size = int(cfg.TRAIN.batch_size/1.5),
            shuffle = False,
            num_workers = cfg.SYSTEM.NUM_WORKERS,
            drop_last=False
        )
    else:
        dloader = DataLoader(
            ds,
            batch_size=int(cfg.TRAIN.batch_size / 1.5),
            shuffle = False,
            num_workers=cfg.SYSTEM.NUM_WORKERS,
            drop_last=False
        )
    return  dloader

def get_fer_test_dataloader(cfg):
    ds = get_dataset(cfg, False)
    dloader = DataLoader(
        ds,
        batch_size = cfg.TEST.batch_size,
        shuffle = False,
        num_workers = cfg.SYSTEM.NUM_WORKERS
    )
    return dloader


