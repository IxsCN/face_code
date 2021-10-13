import numpy as np
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_file", dest="m_file", type=str,
                        default="m.txt")
    parser.add_argument("--lmk_68_file", dest="lmk_68_file", type=str,
                        default="bai_life_tmp.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    m = open(args.m_file).readlines()
    x = open(args.lmk_68_file).readlines()
    m_name = []
    m_dict = {}
    for i in m:
        name = i.split()[0]
        mat = i.split()[1:]
        mat = np.reshape(np.asarray(mat, np.float32), (2, 3))
        m_dict[name] = mat
        m_name.append(name)
    x_dict = {}
    x_name = []
    for i in x:
        name = i.split()[0]
        dots = i.split()[1:]
        dots = np.reshape(np.asarray(dots, np.float32), (68, 2))
        x_dict[name] = dots
        x_name.append(name)

    b = []
    for i in m_name:
        if i in x_name:
            a = []
            for j in range(68):
                res = np.matmul(m_dict[i], np.concatenate(
                    [x_dict[i][j], np.array([1])])).flatten().tolist()
                res=np.clip(res,0.,None)
                a.append(res)
            a = np.concatenate([[i],np.reshape(a,(-1,))])
        b.append(a)
    b = np.asarray(b, str)
    np.savetxt("lmks_68_tmp.txt", b, "%s")
