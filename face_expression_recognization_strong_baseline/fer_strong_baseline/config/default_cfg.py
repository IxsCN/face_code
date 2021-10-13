from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 7

_C.DATA = CN()
_C.DATA.input_size = 224
_C.DATA.input_channel = 3
_C.DATA.crop_residual_pix = 16
_C.DATA.dataset_type = 'ExpW'
_C.DATA.img_dir = '/home/lhw/yangzhi/FaceExpRecog/2.data/ExpW/image/origin/'
_C.DATA.label_dir = '/home/lhw/yangzhi/FaceExpRecog/2.data/ExpW/label/'
_C.DATA.train_label_path = _C.DATA.label_dir+'/label_filtered_conf_thresh_0_no_0_2_part_0_train.lst'
_C.DATA.val_label_path =  _C.DATA.label_dir+'/label_filtered_conf_thresh_0_no_0_2_part_0_val.lst'
_C.DATA.need_crop_xyxy = False
_C.DATA.wanted_catogories = [ ['neutral'],['happy'], ['sad'],['angry']]

_C.TRAIN = CN()
_C.TRAIN.device_ids_str = "0"
_C.TRAIN.device_ids = [0]

_C.TRAIN.do_train = True
_C.TRAIN.batch_size = 72
_C.TRAIN.save_dir = '/home/lhw/data_disk_fast/comp_workspace/saved_model/fer_seres50/'

_C.TRAIN.gradient_accumulation_steps = 1
_C.TRAIN.num_train_epochs = 30

_C.TRAIN.optimizer = 'adam' # 'adam' or 'nadam'
_C.TRAIN.warmup_steps = 300 # 500
_C.TRAIN.learning_rate = 4*1e-6  # 1e-4  initial lr
_C.TRAIN.momentum = 0.9
_C.TRAIN.lr_decay_gamma = 0.2         # lr = lr * lr_decay_gamma
_C.TRAIN.weight_decay = 0.9          # lr = lr - weight_decay
_C.TRAIN.milestones = [2000, 4000]
_C.TRAIN.adam_epsilon = 1e-6
_C.TRAIN.drop_rate = 0

_C.TRAIN.early_stop_n = 3

_C.TRAIN.log_steps = 5


_C.TRAIN.learner = 'scn' # 'simple' or 'scn' or 'ran'
_C.TRAIN.beta = 0.7
_C.TRAIN.relabel_epoch = 10
_C.TRAIN.margin_1 = 0.15
_C.TRAIN.margin_2 = 0.2






_C.TEST = CN()
_C.TEST.do_train = False
_C.TEST.batch_size = 256
_C.TEST.model_load_path = '/home/lhw/data_disk_fast/comp_workspace/saved_model/fer_mobile_v2_data_part0/'
_C.TEST.save_dir = '/home/lhw/data_disk_fast/comp_workspace/saved_model/test_log/'
_C.TEST.device_ids_str = "0"
_C.TEST.device_ids = [0]

_C.MODEL = CN()
_C.MODEL.model_name = 'se_resnext50_32x4d'
_C.MODEL.num_classes = 4
_C.MODEL.pretrained = ''
_C.MODEL.checkpoint = ''


_C.LOSS = CN()
_C.LOSS.type = 'CE'  # or 'CE_WLS'
_C.LOSS.alpha = 0.0125
_C.LOSS.use_update_w = False
_C.LOSS.update_w_start_epoch = 20
_C.LOSS.use_label_smooth = False

def get_fer_cfg_defaults(merge_from = None):
  cfg =  _C.clone()
  if merge_from is not None:
      cfg.merge_from_other_cfg(merge_from)
  return cfg