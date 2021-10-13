#coding=utf-8
import os, sys
import numpy as np
import torch
from fer_strong_baseline.config.default_cfg import  get_fer_cfg_defaults
from fer_strong_baseline.face_detect import MTCNN
from fer_strong_baseline.face_align.face_align import FaceAlign
from IPython import embed
import shutil
import cv2
from  fer_strong_baseline.models.build_model import build_model
import argparse
import random
import time
from pathlib import Path
from fer_strong_baseline.utils.common import setup_seed
from fer_strong_baseline.datasets.aug import fer_test_aug

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='./configs/mobilev2_4_cls.yml',
                        type=str,
                        help="模型配置文件路径")
    parser.add_argument('--images', type=str, default='./examples/data/', help='需要进行检测的图片文件夹')
    parser.add_argument('--img-size', type=int, default=416, help='输入分辨率大小')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='检测后的图片或视频保存的路径')
    parser.add_argument('--half', default=False, help='是否采用半精度FP16进行推理')
    parser.add_argument('--webcam', default=True, help='是否使用摄像头进行检测')
    return parser.parse_args()


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  #
    clb = c1[0], c2[1]
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        #c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(img, c1, c2, color, -1)  # filled
        #cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        clb2 = clb[0] + t_size[0], clb[1] + t_size[1] + 3
        #c1 = c2[0] - t_size[0], c2[1] + t_size[1] + 3
        cv2.rectangle(img, clb, clb2, color, -1)  # filled
        cv2.putText(img, label, (clb[0], clb2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class LoadWebcam:  # for inference
    def __init__(self, img_size=416, half=False, det_type='ctdet'):
        self.img_size = img_size
        self.half = half  # half precision fp16 images
        self.det_type = det_type
        pipe = 0  # local camera

        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.mode = 'webcam'

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cap.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right
        print('webcam %g: ' % self.count, end='')

        return img_path, img0, None


    def __len__(self):
        return 0


class LoadImages:
    def __init__(self, path, img_size=416, half=False):
        raise NotImplementedError


def detect(cfg,
           images=None,  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           save_txt=False,
           save_images=True):
   
    # Initialize
    device = 'cpu'  # cpu or gpu
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # remove previous result
    os.makedirs(output)

    mtcnn = MTCNN(
        image_size=224,
        min_face_size=40,
        #         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device=torch.device('cpu'))

    model = build_model(cfg)
    # USE CPU
    if device == 'cpu':
        model.load_state_dict(torch.load(cfg.TEST.model_load_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(cfg.TEST.model_load_path))


    # Eval mode
    model.to(device).eval()

    # Half precision
    args.half = args.half and device.type != 'cpu'  # half precision only supported on CUDA
    if args.half:
        model.half()


    # Set Dataloader
    vid_path, vid_writer = None, None
    if args.webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size, half=args.half)
    else:
        dataloader = LoadImages(images, img_size=img_size, half=args.half)

    classes = ['happy', 'anger', 'sad', 'neutral', 'disgust', 'surprised'] # class list
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))] # random color for each class

    aug = fer_test_aug(cfg.DATA.input_size)

    # Run inference
    t0 = time.time()
    for i, (path, img, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name) 
        # Get detections and align

        if img.dtype != 'uint8': # check whether image or not
            raise RuntimeError('dtype of numpy array is not uint8!!! check it !!!')

        bboxs, scores, landmarks = mtcnn.detect(img, landmarks=True)

        cls_faces = []
        if bboxs is not None:
            landmarks = landmarks.tolist()
            for face_id, bbox in enumerate(bboxs):
                ori_landmark = landmarks[face_id]
                # embed()
                ori_landmark.append([bbox[0], bbox[1]])
                ori_landmark.append([bbox[2], bbox[3]])

                alignedImg = FaceAlign(img, ori_landmark, 255, use_bbox=True)

                alignedImg = cv2.cvtColor(alignedImg, cv2.COLOR_BGR2RGB)
                alignedImg = aug(image=alignedImg)['image']
                alignedImg = alignedImg.transpose((2, 0, 1))
                alignedImg = torch.from_numpy(alignedImg)
                alignedImg = alignedImg.unsqueeze(0)

                pred_loggits = model(alignedImg)
                pred_loggits = pred_loggits.softmax(dim=-1)
                cls = np.argmax(pred_loggits)
                cls_faces.append(cls)
                print(classes[int(cls)])

            for face_id, bbox in enumerate(bboxs):
                plot_one_box(bbox, img, label=classes[cls_faces[face_id]], color=colors[int(cls_faces[face_id])])

            if args.webcam:  # Show live webcam
                cv2.imshow("fer", img)
        

if __name__ == "__main__":

    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    args = get_args()
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    print(cfg)

    with torch.no_grad():
        detect(cfg,
               images=args.images,
               img_size=args.img_size,
               fourcc=args.fourcc,
               output=args.output)

