import  numpy as np
import os
from os import listdir
import errno
import  random
import  torch

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def create_dir_maybe(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def load_imgs(img_dir, image_list_file, label_file):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    imgs_forth = list()
    imgs_fifth = list()
    imgs_sixth = list()
    imgs_seventh = list()
    imgs_eigth = list()
    max_label = 0
    count = 0
    with open(image_list_file, 'r') as imf:
        with open(label_file, 'r') as laf:
            for line in imf:
                # count += 1
                # print('count',count)
                space_index = line.find(' ')

                video_name = line[0:space_index]  # name of video
                img_count = line[space_index + 1:]  # number of frames in video

                video_path = os.path.join(img_dir, video_name)  # video_path is the path of each video
                ###  for sampling triple imgs in the single video_path  ####

                img_lists = listdir(video_path)
                # pdb.set_trace()
                record = laf.readline().strip().split()
                img_lists.sort()  # sort files by ascending
                # pdb.set_trace()
                img_path_first = video_path + '/' + img_lists[0]
                img_path_second = video_path + '/' + img_lists[0]
                img_path_third = video_path + '/' + img_lists[0]
                img_path_forth = video_path + '/' + img_lists[0]
                img_path_fifth = video_path + '/' + img_lists[0]
                img_path_sixth = video_path + '/' + img_lists[0]
                img_path_seventh = video_path + '/' + img_lists[0]
                img_path_eigth = video_path + '/' + img_lists[0]
                # pdb.set_trace()
                label = int(record[0])
                imgs_first.append((img_path_first, label))
                imgs_second.append((img_path_second, label))
                imgs_third.append((img_path_third, label))
                imgs_forth.append((img_path_forth, label))
                imgs_fifth.append((img_path_fifth, label))
                imgs_sixth.append((img_path_sixth, label))
                # pdb.set_trace()
                imgs_seventh.append((img_path_seventh, label))
                imgs_eigth.append((img_path_eigth, label))

                ###  return multi paths in a single video  #####

                # print 'record[0],record[1],record[2]',record[0],record[1],record[2]

    return imgs_first, imgs_second, imgs_third, imgs_forth, imgs_fifth, imgs_sixth, imgs_seventh, imgs_eigth


def get_most_left_and_most_right_point(lst):
    left_idx = -1
    left_x = np.inf
    right_idx = -1
    right_x = 0

    for i, (x, y) in enumerate(lst):
        if x < left_x:
            left_x = x
            left_idx = i
        if x > right_x:
            right_x = x
            right_idx = i

    return lst[left_idx], lst[right_idx]

def get_center_point(lst):
    cx, cy = 0, 0
    for x,y in lst:
        cx += x
        cy += y
    return [cx/len(lst), cy/len(lst)]


def get_nearest_landmark_idx(gt_landmarks, five_landmark):
    idx = -1
    dist = np.inf
    import pdb
    # pdb.set_trace()
    for i, pred_landmark in enumerate(five_landmark):
        curr_dist = 0
        for p, q in zip(gt_landmarks, pred_landmark):
            curr_dist += abs(p[0] - q[0]) + abs(p[1] - q[1])
        if curr_dist < dist:
            dist = curr_dist
            idx = i
    return idx


class Dict(dict):
    # # self.属性写入 等价于调用dict.__setitem__
    __setattr__ = dict.__setitem__
    # # self.属性读取 等价于调用dict.__setitem__
    __getattribute__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst

