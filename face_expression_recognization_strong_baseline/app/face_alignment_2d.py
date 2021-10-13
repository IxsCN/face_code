import face_alignment
from skimage import io
import pdb
import cv2
from random import randint

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

img_fn = '/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Image/aligned/test_0088_aligned.jpg'
input = io.imread(img_fn)
boxes, preds = fa.get_boxes_and_landmarks(input)

gui = cv2.imread(img_fn)

# distribution of landmark  https://blog.csdn.net/keyanxiaocaicai/article/details/52150322

# 0-16是下颌线(红），
# 17-21是右眼眉（橙），
# 22-26是左眼眉（黄），
# 27-35是鼻子（浅绿），
# 36-41是右眼（深绿），
# 42-47是左眼（浅蓝），
# 48-60是嘴外轮廓（深蓝），
# 61-67是嘴内轮廓（紫）
#

landmark_idx_tmp =  [[0 for _ in range(0,17)],
                 [1 for _ in range(17,22)],
                 [2 for _ in range(22,27)],
                 [3 for _ in range(27,36)],
                 [4 for _ in range(36,42)],
                 [5 for _ in range(42,48)],
                 [6 for _ in range(48,61)],
                 [7 for _ in range(61,68)],
                  ]
landmark_idx = []
for group in landmark_idx_tmp:
    landmark_idx.extend(group)

color_vec = [[randint(155, 255), randint(155, 255), randint(155, 255)] for _ in range(8)]

for pred in preds:
    for i, color_i in enumerate(landmark_idx):
        r, g, b = color_vec[color_i]
        cv2.circle(gui, (pred[i][0], pred[i][1]), 3, (b,g,r), 3)

for box in boxes:
    cv2.rectangle(gui, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
cv2.namedWindow('landmark', 0)
cv2.imshow('landmark', gui)
cv2.waitKey(0)


pdb.set_trace()


