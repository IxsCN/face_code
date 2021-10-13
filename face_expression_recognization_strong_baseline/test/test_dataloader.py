import os
from tqdm import tqdm
from fer_strong_baseline.config.default_cfg import get_fer_cfg_defaults


from fer_strong_baseline.datasets.RAF_DB import RafDataSet
from fer_strong_baseline.datasets.RAF_DB_gaussian import RafGaussianOfflineDataSet


def test_rafdb(cfg):
    print("test Train dataset...")
    raf_g = RafGaussianOfflineDataSet(cfg, is_train=True)
    raf = RafDataSet(cfg, is_train=True)

    cnt = 0
    for r, l in tqdm(zip(raf.samples, raf.labels)):
        rb = os.path.basename(r).replace("_aligned.jpg", "")
        for rg, gg, lg in zip(raf_g.samples, raf_g.gaussianes, raf_g.labels):
            rgb = os.path.basename(rg).replace(".jpg", "")
            ggb = os.path.basename(gg).replace(".jpg", "")
            
            if rgb != ggb:
                print("something wrong with rgb and ggb...")
                import pdb; pdb.set_trace()
            if rb == rgb:
                if l != lg:
                    print("wrong label, l:{}, lg:{}".format(l,lg))
                    cnt += 1

    print("wrong number of label:{}".format(cnt))

    print("test Test dataset...")
    raf_g = RafGaussianOfflineDataSet(cfg, is_train=False)
    raf = RafDataSet(cfg, is_train=False)

    cnt = 0
    for r, l in tqdm(zip(raf.samples, raf.labels)):
        rb = os.path.basename(r).replace("_aligned.jpg", "")
        for rg, gg, lg in zip(raf_g.samples, raf_g.gaussianes, raf_g.labels):
            rgb = os.path.basename(rg).replace(".jpg", "")
            ggb = os.path.basename(gg).replace(".jpg", "")
            
            if rgb != ggb:
                print("something wrong with rgb and ggb...")
                import pdb; pdb.set_trace()
            if rb == rgb:
                if l != lg:
                    print("wrong label, l:{}, lg:{}".format(l,lg))
                    cnt += 1

    print("wrong number of label:{}".format(cnt))




def test_dataloader():
    # init cfg for test
    config_path = "../configs/rafdb_scnres18_gaussian_ce.yml"
    cfg = get_fer_cfg_defaults()
    cfg.merge_from_file(config_path)

    print("test_rafdb...")
    test_rafdb(cfg)






if __name__ == "__main__":
    test_dataloader()