import os, sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy
from IPython import embed
import time

# expect 5 landmark coord on imgSize 
# ImgSize = [112, 96]           
# Coord7Point = [[30.2946, 51.6963],
#                [65.5318, 51.6963],
#                [48.0252, 71.7366],
#                [33.5493, 92.3655],
#                [62.7299, 92.3655],
#                [      0,       0],
#                [    112,      96]]

base_coord = [[0.31556875, 0.46157422], [0.68262305, 0.45983398], [0.5002625, 0.64050547], 
                [0.34947187, 0.8246918], [0.65343633, 0.82325078]]

def OutSizeRescale(outSize):
    assert len(outSize) == 2
    # global ImgSize
    # global Coord7Point
    global base_coord
    outSize_target_coord = [[outSize[0] * x[0], outSize[1] *x[1]] for x in base_coord]
    # Coord7Point = [ [sh * x[0],sw * x[1]] for x in Coord7Point]
    # ImgSize = [OutSize, OutSize]
    return outSize_target_coord


def TransformationFromPoints(p, q):

    pad = numpy.ones(p.shape[0])
    p = numpy.insert(p, 2, values=pad, axis=1)
    q = numpy.insert(q, 2, values=pad, axis=1)

    # 最小二乘
    # M1 = numpy.linalg.inv(p.T*p)
    M1 = numpy.linalg.pinv(p.T*p)  # pseudo inverse 
    M2 = p.T*q
    M = M1*M2
    return M.T


def WarpIm(img_im, orgi_landmarks, tar_landmarks, outimgsize):
    # embed()
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = TransformationFromPoints(pts1, pts2)
    try:
        dst = cv2.warpAffine(img_im, M[:2], (outimgsize[0], outimgsize[1]))
    except Exception as e:
        raise
    return dst


def FaceAlign(img_im, face_landmarks, outimgsize):
    # import pdb
    # pdb.set_trace()
    assert len(face_landmarks) == 5
    target_coord = OutSizeRescale(outimgsize)
    try:
        dst = WarpIm(img_im, face_landmarks, target_coord, outimgsize)
    except Exception as e:
        raise
    # crop_im = dst[0:outimgsize[1], 0:outimgsize[0]]
    # return crop_im
    return dst


