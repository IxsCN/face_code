from fer_strong_baseline.loss.softmaxloss import (
    CrossEntropy,
    WassersteinCrossEntropy,
)


__mapping_loss = {
    'CE': CrossEntropy,
    'CE_WLS': WassersteinCrossEntropy,
}

def get_loss(cfg, args, device, logger):
    if cfg.LOSS.type not in __mapping_loss.keys():
        raise NotImplementedError('LOSS Type not supported!')
    if cfg.LOSS.type == "CE":
        return CrossEntropy()
    elif cfg.LOSS.type == "CE_WLS":
        return WassersteinCrossEntropy(args, cfg.MODEL.num_classes, device=device, logger=logger)

    # return __mapping_loss[cfg.LOSS.type]()