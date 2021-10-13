import numpy as np
import cv2
from parse_align_json import AlignConfig
import os
## Referenced From Matlab cp2tform.m, 2015a
import argparse


def get_channel(img):
    if len(img.shape) == 2:
        return 1
    else:
        return img.shape[2]


def getMatRank(matrix, thres=0.05):
    s, u, vt = cv2.SVDecomp(matrix)
    return np.sum(s > thres)


def getShift(points):  ## n x 2
    tol = 1000
    minPoints = np.min(points, 0)
    maxPoints = np.max(points, 0)
    center = (minPoints + maxPoints) / 2
    span = maxPoints - minPoints
    if (span[0] > 0 and np.abs(center[0]) * 1.0 / span[0] > tol) or \
            (span[1] > 0 and np.abs(center[1]) * 1.0 / span[1] > tol):
        shift = center
    else:
        shift = np.zeros((2))
    return shift


def findNonReflectiveSimilarity(uv, xy):
    x = xy[:, 0]
    y = xy[:, 1]
    M = xy.shape[0]
    # X = [x   y  ones(M,1)   zeros(M,1);
    #     y  -x  zeros(M,1)  ones(M,1)  ];
    X = np.zeros((2 * M, 4)).astype(np.float32)
    X[0:M, 0] = x
    X[0:M, 1] = y
    X[M:2 * M, 0] = y
    X[M:2 * M, 1] = -1 * x
    X[0:M, 2] = 1
    X[M:2 * M, 3] = 1
    # u,v
    u = uv[:, 0]
    v = uv[:, 1]
    U = np.zeros((2 * M)).astype(np.float32)
    U[0:M] = u
    U[M:2 * M] = v
    # Least Squares Solution
    X_trans = cv2.transpose(X)
    tmp_matrix = np.dot(X_trans, X)
    flag, inv_mat = cv2.invert(tmp_matrix)
    if flag == 1.0:
        # inv(X' * X) * X' * b        
        r = np.dot(np.dot(inv_mat, X_trans), U)
        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]
        Tinv = np.float32([[sc, -1 * ss, 0],
                           [ss, sc, 0],
                           [tx, ty, 1]])
        flag, T = cv2.invert(Tinv)
        assert (flag == 1.0)
        T[0, 2] = 0
        T[1, 2] = 0.0
        T[2, 2] = 1
        return T
    else:
        return None


def GetAffinePoints(pts_in, trans):
    pts_out = pts_in.copy()
    assert (pts_in.shape[1] == 2)

    for k in range(pts_in.shape[0]):
        pts_out[k, 0] = pts_in[k, 0] * trans[0, 0] \
                        + pts_in[k, 1] * trans[0, 1] \
                        + trans[0, 2]
        pts_out[k, 1] = pts_in[k, 0] * trans[1, 0] \
                        + pts_in[k, 1] * trans[1, 1] \
                        + trans[1, 2]
    return pts_out


def cp2tform(uv, xy):  ## src and dst
    assert (type(uv) == type(np.array([])) and uv.shape[0] >= 3)
    assert (type(xy) == type(np.array([])) and xy.shape[0] >= 3)
    uvShift = getShift(uv)
    xyShift = getShift(xy)
    assert (uv.shape == xy.shape)

    needToShift = 0
    # if np.sum(xyShift != 0) + np.sum(uvShift != 0) > 0:
    #    needToShift = 1
    if needToShift == 0:
        trans = findNonReflectiveSimilarity(uv, xy)
    else:
        uv_shift = uv - uvShift
        xy_shift = xy - xyShift
        trans = findNonReflectiveSimilarity(uv, xy)
    if trans is not None:
        t_trans = cv2.transpose(trans[:, 0:2])
    return t_trans


def align_face(lmks_pts, img, align_param):
    coord5points = [30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366,
                    33.5493, 92.3655, 62.7299, 92.2041]
    dst = np.array(coord5points).reshape((5, 2)).astype(np.float32)
    assert (len(lmks_pts) == 10)
    src = np.array(lmks_pts).reshape((5, 2)).astype(np.float32)
    t = cp2tform(src, dst)

    if t is not None:
        dst_w = align_param.width
        dst_h = align_param.height
        if align_param.fill_with_value == 1:
            value = align_param.fill_value
        else:
            value = 255
        channel = get_channel(img)
        dst_img = cv2.warpAffine(img, t, (dst_w, dst_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(value,) * channel)
        dst_pts = GetAffinePoints(src, t)
        return dst_img, dst_pts
    else:
        return None, None


"""
if __name__ == '__main__':
    coord5points = [30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366,  33.5493, 92.3655, 62.7299, 92.2041]
    dst = np.array(coord5points).reshape((5,2)).astype(np.float32)
    #points = [64.86, 96.21, 104.66, 104.17, 82.94, 119.73, 59.01, 135.91, 97.25, 143.71]
    points = [24.92626953,  20.69677734, 52.41796875,  21.35546875 ,  39.52148438,  37.61474609 , 26.52099609,  49.40185547 ,49.19384766,  49.60986328]
    src = np.array(points).reshape((5,2)).astype(np.float32)
    t = cp2tform(src, dst)    
    #ground_truth = np.array(
    #    [[  0.90980589 , -0.17128041,0],
    #    [  0.17128041 ,  0.90980589 , 0   ],
    #    [-46.88935089, -23.22565651  , 1   ]])
    image = cv2.imread('000023.jpg', 1)
    dst_image = cv2.warpAffine(image.copy(), t, (96,112))
    cv2.imwrite('test_1.jpg', dst_image)
    print(GetAffinePoints(src, t))
    
    align_param = AlignConfig('align.json')
    dst_img, pts = align_face(points, image, align_param)
    print(pts)
    cv2.imwrite('test_2.jpg', dst_img)
"""


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmks_file", dest="lmks_file", type=str,
                        default="lmks.txt")
    parser.add_argument("--output_dir", dest="output_dir", type=str,
                        default="./output")
    parser.add_argument("--img_size", dest="img_size", type=int, default=256)
    parser.add_argument("--img_dir", dest="img_dir", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    # 5 (x,y) standard landmarks on 1x1 image
    base_coords = [0.31556875, 0.46157422, 0.68262305, 0.45983398, 0.5002625,
                   0.64050547, 0.34947187, 0.8246918,
                   0.65343633, 0.82325078]

    # scalar base coords
    # coord5points = base_coords * args.img_size
    dst = np.array(base_coords).reshape((5, 2)).astype(np.float32) * (
        args.img_size, args.img_size)

    # get landmarks
    f = open(args.lmks_file, "r")
    # lmks.txt file write "img_path 10 coords\n" each line
    lines = f.readlines()
    f.close()
    paths = []
    dict = {}
    for line in lines:
        path = line.split()[0]
        line = line.split()[1:]
        line = np.reshape(line, (5, 2)).astype(np.float32)
        paths.append(path)
        dict[path] = line
        # lmks.append(line)
    # lmks = np.asarray(lmks).astype(np.float32)

    # align
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    M = []
    for i in paths:
        src = dict[i]
        t = cp2tform(src, dst)
        image = cv2.imread(os.path.join(args.img_dir, i), 1)
        dst_image = cv2.warpAffine(image.copy(), t,
                                   (args.img_size, args.img_size))
        cv2.imwrite("{}/{}".format(args.output_dir, os.path.basename(i)),
                    dst_image)
        # print("[*] Finished {}".format(i))
        flatten = np.reshape(t, (-1,))
        concat = np.concatenate([[os.path.basename(i)], flatten])
        M.append(concat)
        if paths.index(i) % 500 == 0:
            print("Processed {}/{}".format(paths.index(i), len(cd )))
    m = np.asarray(M)
    np.savetxt("m.txt", m, "%s")
"""
        align_param = AlignConfig("align.json")
        dst_img, pts =align_face(lmks[i],image,align_param)
        print(pts)
        cv2.imwrite("{}".format(os.path.basename(path[i])), dst_img)
"""
