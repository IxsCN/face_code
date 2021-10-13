import cv2
import numpy as np
import argparse
import os


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=5, help="radios of circle")
    parser.add_argument("--dir_name", type=str, default="")
    parser.add_argument("--lmk_file", type=str, default="")
    parser.add_argument("--draw_type", type=str, default="draw_line",
                        help="draw_line or draw_points")
    return parser.parse_args()


def img_creater(img, lmks, r):
    img = cv2.circle(img, lmks, r, (255, 255), -1)
    return img


def draw_line(img, pt1, pt2):
    img = cv2.line(img, tuple(pt1), tuple(pt2), (255, 255))
    return img


if __name__ == "__main__":
    args = arg_parser()
    if not os.path.exists(args.dir_name):
        os.mkdir(args.dir_name)
    f = np.loadtxt(args.lmk_file, str)
    img_name = []
    dict = {}
    for i in f:
        img_name.append(i[0])
        dict[i[0]] = i[1:].astype(np.float32).reshape(
            (68, 2))

    for name in img_name:
        shape = (256, 256)
        img = np.zeros(shape, np.uint8)
        if args.draw_type == "draw_points":
            for pt in dict[name]:
                img = img_creater(img, tuple(pt), args.r)
        if args.draw_type == "draw_line":
            pts = dict[name]
            pts1 = pts[:17]
            for i in range(len(pts1) - 1):
                img = draw_line(img, pts1[i], pts1[i + 1])
            pts2 = pts[17:22]
            for i in range(len(pts2) - 1):
                img = draw_line(img, pts2[i], pts2[i + 1])
            pts3 = pts[22:27]
            for i in range(len(pts3) - 1):
                img = draw_line(img, pts3[i], pts3[i + 1])
            pts4 = pts[27:31]
            for i in range(len(pts4) - 1):
                img = draw_line(img, pts4[i], pts4[i + 1])
            pts5 = pts[31:36]
            for i in range(len(pts5) - 1):
                img = draw_line(img, pts5[i], pts5[i + 1])
            pts6 = pts[36:42]
            for i in range(len(pts6) - 1):
                img = draw_line(img, pts6[i], pts6[i + 1])
                img = draw_line(img, pts6[0], pts6[-1])
            pts7 = pts[42:48]
            for i in range(len(pts7) - 1):
                img = draw_line(img, pts7[i], pts7[i + 1])
                img = draw_line(img, pts7[0], pts7[-1])
            pts8 = pts[48:61]
            for i in range(len(pts8) - 1):
                img = draw_line(img, pts8[i], pts8[i + 1])
                img = draw_line(img, pts8[0], pts8[-1])
            pts9 = pts[61:]
            for i in range(len(pts9) - 1):
                img = draw_line(img, pts9[i], pts9[i + 1])
                img = draw_line(img, pts9[0], pts9[-1])
        if args.draw_type == "fill":
            pts = np.concatenate([dict[name][:17],dict[name][26:23:-1],dict[name][21:18:-1]])
            cv2.fillConvexPoly(img,np.array(pts, dtype=np.int32), [255,255])  # an error will raise if not convert pts to np.int32
        cv2.imwrite(os.path.join(args.dir_name, name), img)
        if img_name.index(name) % 500 == 0:
            print(
                "Processed {}/{}".format(img_name.index(name), len(img_name)))
    print("Done")
