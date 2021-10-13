import face_alignment
from skimage import io
import os
from tqdm import tqdm

def get_img_lst(img_path):
    img_lst = []
    for _root, _dir, _file in os.walk(img_path):
        for file_name in _file:
            if '.jpg' in file_name or '.png' in file_name:
                img_lst.append(os.path.join(_root, file_name))
    return img_lst

img_lst = get_img_lst('/media/yz/62C9BA4E5D826344/data/RAF-DB/basic/Image/original')

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

for img_fn in tqdm(img_lst):
    input = io.imread(img_fn)
    preds = fa.get_landmarks(input)
    
    with open(img_fn.replace(".jpg", "_auto_attri.txt"), 'w') as f:
        try:
            f.write(img_fn)
            f.write('\t')
            for i in range(len(preds)):
                one_face_landmark = preds[i]
                one_face_landmark = one_face_landmark.tolist()
                one_face_landmark_str = ';'.join(','.join(str(s) for s in p) for p in one_face_landmark)
                f.write(one_face_landmark_str)
                f.write('\t')
            f.write('\n')
        except Exception as e:
            # import pdb
            # pdb.set_trace()
            print(img_fn)

