import torch
import torch.nn as nn
import numpy as np
import math

def norm_array_by_row(data):
    for i in range(data.shape[0]):
        _range = torch.sum(data[i])
        data[i] = data[i] / (_range + 1e-9)
    return data

def l2_norm_array_by_matrix_norm(data):
    tmp = data * data
    return data / (np.sqrt(sum(sum(tmp)))+ 1e-9)




class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward_all(self, pred, label):
        loss = self.criterion(pred, label)
        return {"loss": loss, "ce": loss}

    def forward_all_condition(self, pred, label, epoch):
        loss = self.criterion(pred, label)
        return {"loss": loss, "ce": loss}

    def update_w_line_norm(self, weight):
        pass

    def update_w_matrix_norm(self, weight):
        pass


    def init_w_line_norm(self, weight):
        pass

    def init_w_matrix_norm(self, weight):
        pass


class WassersteinCrossEntropy(nn.Module):
    """
    only support raf-db yet!
    """
    def __init__(self, args, num_classes,
                 device='cpu', logger=None):

        super(WassersteinCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.alpha = args['alpha']
        self.device = device
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        # self.datatype = datatype
        self.logger = logger
        self.base_criterion = nn.CrossEntropyLoss()

        # if self.datatype == 'AffectNet':
        #     self.cls_num = ['normal', 'happy', 'sad', 'surprised',
        #                          'fear', 'disgust', 'anger', 'contempt']
        #     # self.dist = [[0.   , 0.733, 0.621, 0.749, 0.788, 0.791, 0.704, 0.753],
        #     #         [0.733, 0.   , 1.342, 0.786, 1.053, 1.412, 1.222, 1.286],
        #     #         [0.621, 1.342, 0.   , 1.251, 1.144, 0.717, 0.844, 0.849],
        #     #         [0.749, 0.786, 1.251, 0.   , 0.316, 0.904, 0.645, 0.703],
        #     #         [0.788, 1.053, 1.144, 0.316, 0.   , 0.647, 0.383, 0.43 ],
        #     #         [0.791, 1.412, 0.717, 0.904, 0.647, 0.   , 0.265, 0.219],
        #     #         [0.704, 1.222, 0.844, 0.645, 0.383, 0.265, 0.   , 0.064],
        #     #         [0.753, 1.286, 0.849, 0.703, 0.43 , 0.219, 0.064, 0.   ]]

        # elif self.datatype == 'FER2013':
        #     self.cls_num = ['anger', 'disgust', 'fear', 'happy',
        #                        'sad', 'surprised', 'normal']

        #     # self.dist = [[0.   , 0.265, 0.383, 1.222, 0.844, 0.645, 0.704],
        #     #         [0.265, 0.   , 0.647, 1.412, 0.717, 0.904, 0.791],
        #     #         [0.383, 0.647, 0.   , 1.053, 1.144, 0.316, 0.788],
        #     #         [1.222, 1.412, 1.053, 0.   , 1.342, 0.786, 0.733],
        #     #         [0.844, 0.717, 1.144, 1.342, 0.   , 1.251, 0.621],
        #     #         [0.645, 0.904, 0.316, 0.786, 1.251, 0.   , 0.749],
        #     #         [0.704, 0.791, 0.788, 0.733, 0.621, 0.749, 0.   ]]
        # elif self.datatype == 'RAF_DB':
        #     self.cls_num = ['surprised', 'fear', 'disgust', 'happy',
        #                'sad', 'anger', 'normal']

        #     # self.dist = [[0.   , 0.316, 0.904, 0.786, 1.251, 0.645, 0.749],
        #     #      [0.316, 0.   , 0.647, 1.053, 1.144, 0.383, 0.788],
        #     #      [0.904, 0.647, 0.   , 1.412, 0.717, 0.265, 0.791],
        #     #      [0.786, 1.053, 1.412, 0.   , 1.342, 1.222, 0.733],
        #     #      [1.251, 1.144, 0.717, 1.342, 0.   , 0.844, 0.621],
        #     #      [0.645, 0.383, 0.265, 1.222, 0.844, 0.   , 0.704],
        #     #      [0.749, 0.788, 0.791, 0.733, 0.621, 0.704, 0.   ],]
        # else:
        #     print('unsupport dataset')
        #     raise NotImplementedError

        # self.dist = [[0.         , args['w01'], args['w02'], args['w03'], args['w04'], args['w05'], args['w06'], args['w07'], args['w08'], args['w09']],
        #             [args['w01'], 0.         , args['w12'], args['w13'], args['w14'], args['w15'], args['w16'], args['w17'], args['w18'], args['w19']],
        #             [args['w02'], args['w12'], 0.         , args['w23'], args['w24'], args['w25'], args['w26'], args['w27'], args['w28'], args['w29']],
        #             [args['w03'], args['w13'], args['w23'], 0.         , args['w34'], args['w35'], args['w36'], args['w37'], args['w38'], args['w39']],
        #             [args['w04'], args['w14'], args['w24'], args['w34'], 0.         , args['w45'], args['w46'], args['w47'], args['w48'], args['w49']],
        #             [args['w05'], args['w15'], args['w25'], args['w35'], args['w45'], 0.         , args['w56'], args['w57'], args['w58'], args['w59']],
        #             [args['w06'], args['w16'], args['w26'], args['w36'], args['w46'], args['w56'], 0.         , args['w67'], args['w68'], args['w69']],
        #             [args['w07'], args['w17'], args['w27'], args['w37'], args['w47'], args['w57'], args['w67'], 0.         , args['w78'], args['w79']],
        #             [args['w08'], args['w18'], args['w28'], args['w38'], args['w48'], args['w58'], args['w68'], args['w78'], 0.         , args['w89']],
        #             [args['w09'], args['w19'], args['w29'], args['w39'], args['w49'], args['w59'], args['w69'], args['w79'], args['w89'], 0.         ]]

        self.dist = [[0.        , args['w01'], args['w02'], args['w03'], args['w04'], args['w05'], args['w06']],
                    [args['w01'], 0.         , args['w12'], args['w13'], args['w14'], args['w15'], args['w16']],
                    [args['w02'], args['w12'], 0.         , args['w23'], args['w24'], args['w25'], args['w26']],
                    [args['w03'], args['w13'], args['w23'], 0.         , args['w34'], args['w35'], args['w36']],
                    [args['w04'], args['w14'], args['w24'], args['w34'], 0.         , args['w45'], args['w46']],
                    [args['w05'], args['w15'], args['w25'], args['w35'], args['w45'], 0.         , args['w56']],
                    [args['w06'], args['w16'], args['w26'], args['w36'], args['w46'], args['w56'], 0.         ]]


        self.dist = torch.from_numpy(np.array(self.dist, np.float))
        # self.dist = pow(self.dist, 2)
        self.dist = l2_norm_array_by_matrix_norm(self.dist)  # norm after pow
        if self.logger:
            self.logger.write('alpha:', self.alpha)
            self.logger.write(self.dist)

    # def update_w_line_norm(self, weight):
    #     self.dist += 0.1 * norm_array_by_row(torch.from_numpy(weight).float())
    #     self.logger.write("delta w:", norm_array_by_row(torch.from_numpy(weight).float()))
    #     self.dist = norm_array_by_row(self.dist)
    #     if self.logger:
    #         self.logger.write("update wasser weight by line norm", self.dist)

    # def update_w_matrix_norm(self, weight):
    #     self.dist += 0.1 * l2_norm_array_by_matrix_norm(torch.from_numpy(weight).float())
    #     self.logger.write("delta w:", l2_norm_array_by_matrix_norm(torch.from_numpy(weight).float()))
    #     self.dist = l2_norm_array_by_matrix_norm(self.dist)
    #     if self.logger:
    #         self.logger.write("update wasser weight by matrix norm", self.dist)

    # def init_w_line_norm(self, weight):
    #     self.dist = norm_array_by_row(torch.from_numpy(weight).float())
    #     if self.logger:
    #         self.logger.write("init wasser weight by line norm", self.dist)

    # def init_w_matrix_norm(self, weight):
    #     self.dist = l2_norm_array_by_matrix_norm(torch.from_numpy(weight).float())
    #     if self.logger:
    #         self.logger.write("init wasser weight by matrix norm", self.dist)

    def wasser_criterion(self, pred, gt_cls):
        '''
        gt_cls   not one hot, not cuda
        '''
        gt_cls = gt_cls.detach().cpu().long()  # index_select need int64
        import pdb
        gt_cls = gt_cls.squeeze()  # b,1 -> b
        self.wasserstein_weight = self.dist.index_select(dim=0, index=gt_cls)  # (num_bg/fg) * 5
        
        self.wasserstein_weight.requires_grad = False
        # print('wasserstein_weight:\n', wasserstein_weight, '\n')

        # pred = torch.mul(wasserstein_weight, pred).reshape(-1)
        if self.device != 'cpu':
            self.wasserstein_weight = self.wasserstein_weight.to(self.device)

        loss = torch.mul(self.wasserstein_weight, pred)
        loss = torch.sum(loss)
        return loss

    def forward_all(self, inputs, targets):
        """
                Args:
                    inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
                    targets: ground truth labels with shape (num_classes)
                """
        probs = self.softmax(inputs)
        wloss = self.wasser_criterion(probs, targets)
        ce_loss = self.base_criterion(inputs, targets)

        loss = ce_loss + float(self.alpha) * wloss
        # loss = wloss
        # self.logger.write("loss:",loss)
        # self.logger.write("ce_loss:",ce_loss)
        # self.logger.write("wloss:",wloss)
        return {"loss": loss, "ce": ce_loss, "wloss": wloss}


    # def forward_all_condition(self, inputs, targets,epoch):
    #     """
    #         Args:
    #             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
    #             targets: ground truth labels with shape (num_classes)
    #         """
    #     probs = self.softmax(inputs)
    #     wloss = self.wasser_criterion(probs, targets)
    #     ce_loss = self.base_criterion(inputs, targets)

    #     # loss = ce_loss + float(self.alpha) * wloss
    #     if epoch < 5:
    #         loss = ce_loss
    #     elif epoch < 200:
    #         loss = ce_loss + float(self.alpha) * wloss
    #     else:
    #         loss = float(self.alpha) * wloss

    #     # print("loss:",loss)
    #     # print("ce_loss:",ce_loss)
    #     # print("wloss:",wloss)
    #     return {"loss": loss, "ce": ce_loss, "wloss": wloss}


# class WassersteinCrossEntropyLabelSmooth(nn.Module):
#     def __init__(self, num_classes,
#                  alpha = 0.3,
#                  use_label_smooth=False,
#                  epsilon=0.1,
#                  datatype = 'FER2013',
#                  use_gpu=True,
#                  logger=None):
#         super(WassersteinCrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.use_label_smooth = use_label_smooth
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#         self.softmax = nn.Softmax(dim=1)
#         self.datatype = datatype
#         self.logger = logger

#         # affectnet_cls_num_and_center = {'normal': (-0.019574175715601782, -0.0627188977615683),
#         #                                 'happy': (0.07028056723210271, 0.6643268644924916),
#         #                                 'sad': (-0.2570043582489509, -0.6370063573392657),
#         #                                 'surprised': (0.6893279744258441, 0.1804439995542943),
#         #                                 'fear': (0.7661802158670421, -0.12574468561147706),
#         #                                 'disgust': (0.4573438234998662, -0.693718996983959),
#         #                                 'anger': (0.5667776418527468, -0.45273569245960493)}
#         #
#         # def CalcEulerDistance(p1, p2):
#         #     return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

#         # distance = np.zeros(shape=(7, 7))
#         # for i in range(len(fer2013_cls_num)):
#         #     for j in range(len(fer2013_cls_num)):
#         #         p1 = affectnet_cls_num_and_center[fer2013_cls_num[i]]
#         #         p2 = affectnet_cls_num_and_center[fer2013_cls_num[j]]
#         #         distance[i, j] = CalcEulerDistance(p1, p2)

#         # add wasserstein
#         # TODO.
#         # 0 anger 生气； 1 disgust 厌恶； 2 fear 恐惧； 3 happy 开心；
#         # 4 sad 伤心；5 surprised 惊讶； 6 normal 中性
#         # dist = [[0, 1, 1, 1, 1, 1, 1],
#         #         [1, 0, 1, 1, 1, 1, 1],
#         #         [1, 1, 0, 1, 1, 1, 1],
#         #         [1, 1, 1, 0, 1, 1, 1],
#         #         [1, 1, 1, 1, 0, 1, 1],
#         #         [1, 1, 1, 1, 1, 0, 1],
#         #         [1, 1, 1, 1, 1, 1, 0]]


#         # center point of affectnet
#         # [{'neutral': (-0.019574175715601782, -0.0627188977615683)},
#         # {'happy': (0.07028056723210271, 0.6643268644924916)},
#         # {'sad': (-0.2570043582489509, -0.6370063573392657)},
#         # {'surprise': (0.6893279744258441, 0.1804439995542943)},
#         # {'fear': (0.7661802158670421, -0.12574468561147706)},
#         # {'disgust': (0.4573438234998662, -0.693718996983959)},
#         # {'anger': (0.5667776418527468, -0.45273569245960493)},
#         # {'contempt': (0.5828403601946657, -0.5145737456826646)},
#         # {'None': (-0.16301792557996833, 0.1225122491637435)},
#         # {'Uncertain': (-2.0, -2.0)}, {'Non-Face': (-2.0, -2.0)}]


#         # fer2013_cls_num = ['anger', 'disgust', 'fear', 'happy',
#         #                            'sad', 'surprised', 'normal']
#         # anger disgust
#         # fear suprised
#         # normal
#         # happy
#         # sad

#         if self.datatype == 'AffectNet':  # affectnet
#             # dist = [[0 for _ in range(8)] for _ in range(8)]
#             dist = [[0.   , 0.733, 0.621, 0.749, 0.788, 0.791, 0.704, 0.753],
#                     [0.733, 0.   , 1.342, 0.786, 1.053, 1.412, 1.222, 1.286],
#                     [0.621, 1.342, 0.   , 1.251, 1.144, 0.717, 0.844, 0.849],
#                     [0.749, 0.786, 1.251, 0.   , 0.316, 0.904, 0.645, 0.703],
#                     [0.788, 1.053, 1.144, 0.316, 0.   , 0.647, 0.383, 0.43 ],
#                     [0.791, 1.412, 0.717, 0.904, 0.647, 0.   , 0.265, 0.219],
#                     [0.704, 1.222, 0.844, 0.645, 0.383, 0.265, 0.   , 0.064],
#                     [0.753, 1.286, 0.849, 0.703, 0.43 , 0.219, 0.064, 0.   ]]

#         elif self.datatype == 'FER2013': # fer2013
#             cls_num = ['anger', 'disgust', 'fear', 'happy',
#                                'sad', 'surprised', 'normal']
#             # dist = [[0, 1, 3, 3, 3, 3, 2],
#             #         [1, 0, 3, 3, 3, 3, 2],
#             #         [3, 3, 0, 3, 3, 1, 2],
#             #         [3, 3, 3, 0, 3, 3, 2],
#             #         [3, 3, 3, 3, 0, 3, 2],
#             #         [3, 3, 1, 3, 3, 0, 2],
#             #         [2, 2, 2, 2, 2, 2, 0]]
#             dist = [[0.   , 0.265, 0.383, 1.222, 0.844, 0.645, 0.704],
#                     [0.265, 0.   , 0.647, 1.412, 0.717, 0.904, 0.791],
#                     [0.383, 0.647, 0.   , 1.053, 1.144, 0.316, 0.788],
#                     [1.222, 1.412, 1.053, 0.   , 1.342, 0.786, 0.733],
#                     [0.844, 0.717, 1.144, 1.342, 0.   , 1.251, 0.621],
#                     [0.645, 0.904, 0.316, 0.786, 1.251, 0.   , 0.749],
#                     [0.704, 0.791, 0.788, 0.733, 0.621, 0.749, 0.   ]]
#         elif self.datatype == 'RAF_DB':
#             cls_num = ['surprised', 'fear', 'disgust', 'happy',
#                        'sad', 'anger', 'normal']

#             dist = [[0.   , 0.316, 0.904, 0.786, 1.251, 0.645, 0.749],
#                  [0.316, 0.   , 0.647, 1.053, 1.144, 0.383, 0.788],
#                  [0.904, 0.647, 0.   , 1.412, 0.717, 0.265, 0.791],
#                  [0.786, 1.053, 1.412, 0.   , 1.342, 1.222, 0.733],
#                  [1.251, 1.144, 0.717, 1.342, 0.   , 0.844, 0.621],
#                  [0.645, 0.383, 0.265, 1.222, 0.844, 0.   , 0.704],
#                  [0.749, 0.788, 0.791, 0.733, 0.621, 0.704, 0.   ],]
#         else:
#             print('unsupport dataset')
#             raise NotImplementedError

#         # dist = [[0, 1, 2, 3, 4, 5, 6],
#         #         [1, 0, 1, 2, 3, 4, 5],
#         #         [2, 1, 0, 1, 2, 3, 4],
#         #         [3, 2, 1, 0, 1, 2, 3],
#         #         [4, 3, 2, 1, 0, 1, 2],
#         #         [5, 4, 3, 2, 1, 0, 1],
#         #         [6, 5, 4, 3, 2, 1, 0]]

#         # dist = [[0,1.0,1.0,1.0], [1.0,0,1.0,1.0], [1.0,1.0,0,1.0], [1 .0,1.0,1.0,0]]
#         self.dist = torch.from_numpy(np.array(dist, np.float))
#         # self.dist = pow(self.dist, 2)
#         self.dist = norm_array_by_row(self.dist)  # norm after pow
#         if self.logger:
#             self.logger.write('alpha:', self.alpha)
#             self.logger.write(self.dist)
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         log_probs = self.logsoftmax(inputs)
#         probs = self.softmax(inputs)
#         targets_one_hot = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)  # to one hot

#         if self.use_gpu: targets_one_hot = targets_one_hot.cuda()
#         if self.use_label_smooth:
#             targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes   # label_smooth   [e/n e/n 1-e + e/n e/n e/n]
#             ce_loss = (- targets_smooth * log_probs).mean(0).sum()     # cross entropy
#         else:
#             ce_loss = (- targets_one_hot * log_probs).mean(0).sum()     # cross entropy

#         # wasserstein
#         def wasser_criterion(pred, gt_cls):
#             '''
#             gt_cls   not one hot, not cuda
#             '''
#             gt_cls = gt_cls.detach().cpu().long()  # index_select need int64
#             import pdb
#             gt_cls = gt_cls.squeeze()  # b,1 -> b
#             wasserstein_weight = self.dist.index_select(dim=0, index=gt_cls)  # (num_bg/fg) * 5

#             wasserstein_weight = wasserstein_weight.cuda()
#             wasserstein_weight.requires_grad = False
#             # print('wasserstein_weight:\n', wasserstein_weight, '\n')

#             # pred = torch.mul(wasserstein_weight, pred).reshape(-1)
#             loss = torch.mul(wasserstein_weight, pred)
#             loss = torch.sum(loss)
#             return loss

#         wloss = wasser_criterion(probs, targets)

#         loss = ce_loss + float(self.alpha) * wloss
#         # print("loss:",loss)
#         # print("ce_loss:",ce_loss)
#         # print("wloss:",wloss)
#         return loss


# class CrossEntropyLabelSmooth(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.

#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.
#                 ce = -(gt==1) * log pred
#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """

#     def __init__(self,
#                  num_classes,
#                  use_label_smooth = False,
#                  epsilon=0.1,
#                  use_gpu=True,
#                  logger=None):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.use_label_smooth = use_label_smooth
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#         self.logger = logger
#         # self.logger.write('alpha:', self.alpha)

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
#         if self.use_gpu: targets = targets.cuda()
#         if self.use_label_smooth:
#             targets_smooth = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#             ce_loss = (- targets_smooth * log_probs).mean(0).sum()     # cross entropy
#         else:
#             ce_loss = (- targets * log_probs).mean(0).sum()     # cross entropy
#         return ce_loss, ce_loss, None

# if __name__ == "__main__":

#     fer2013_cls_num = ['anger', 'disgust', 'fear', 'happy',
#                        'sad', 'surprised', 'normal']
#     affectnet_cls_num = ['normal', 'happy', 'sad', 'surprised',
#                          'fear', 'disgust', 'anger', 'contempt']
#     rafdb_cls_num = ['surprised', 'fear', 'disgust', 'happy',
#                'sad', 'anger', 'normal']
#     affectnet_cls_num_and_center = {'normal': (-0.019574175715601782, -0.0627188977615683),
#                                     'happy': (0.07028056723210271, 0.6643268644924916),
#                                     'sad': (-0.2570043582489509, -0.6370063573392657),
#                                     'surprised': (0.6893279744258441, 0.1804439995542943),
#                                     'fear': (0.7661802158670421, -0.12574468561147706),
#                                     'disgust': (0.4573438234998662, -0.693718996983959),
#                                     'anger': (0.5667776418527468, -0.45273569245960493),
#                                     'contempt': (0.5828403601946657, -0.5145737456826646)}

#     def CalcEulerDistance(p1, p2):
#         return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


#     rafdb_distance = np.zeros(shape=(7, 7))
#     for i in range(len(rafdb_cls_num)):
#         for j in range(len(rafdb_cls_num)):
#             p1 = affectnet_cls_num_and_center[rafdb_cls_num[i]]
#             p2 = affectnet_cls_num_and_center[rafdb_cls_num[j]]
#             rafdb_distance[i, j] = CalcEulerDistance(p1, p2)
#     np.set_printoptions(precision=3, suppress=True)
#     print(rafdb_distance)

#     # fer_distance = np.zeros(shape=(7,7))
#     # for i in range(len(fer2013_cls_num)):
#     #     for j in range(len(fer2013_cls_num)):
#     #         p1 = affectnet_cls_num_and_center[fer2013_cls_num[i]]
#     #         p2 = affectnet_cls_num_and_center[fer2013_cls_num[j]]
#     #         fer_distance[i,j] = CalcEulerDistance(p1, p2)
#     # np.set_printoptions(precision=3, suppress=True)
#     # print(fer_distance)

#     # affect_distance = np.zeros(shape=(8, 8))
#     # for i in range(len(affectnet_cls_num)):
#     #     for j in range(len(affectnet_cls_num)):
#     #         p1 = affectnet_cls_num_and_center[affectnet_cls_num[i]]
#     #         p2 = affectnet_cls_num_and_center[affectnet_cls_num[j]]
#     #         affect_distance[i, j] = CalcEulerDistance(p1, p2)
#     # np.set_printoptions(precision=3, suppress=True)
#     # print(affect_distance)
#     # pdb.set_trace()

#     # 0 anger 生气； 1 disgust 厌恶； 2 fear 恐惧； 3 happy 开心；
#         # 4 sad 伤心；5 surprised 惊讶； 6 normal 中性
#         # dist = [[0, 1, 1, 1, 1, 1, 1],
#         #         [1, 0, 1, 1, 1, 1, 1],
#         #         [1, 1, 0, 1, 1, 1, 1],
#         #         [1, 1, 1, 0, 1, 1, 1],
#         #         [1, 1, 1, 1, 0, 1, 1],
#         #         [1, 1, 1, 1, 1, 0, 1],
#         #         [1, 1, 1, 1, 1, 1, 0]]


#         # center point of affectnet
#         # [{'neutral': (-0.019574175715601782, -0.0627188977615683)},
#         # {'happy': (0.07028056723210271, 0.6643268644924916)},
#         # {'sad': (-0.2570043582489509, -0.6370063573392657)},
#         # {'surprise': (0.6893279744258441, 0.1804439995542943)},
#         # {'fear': (0.7661802158670421, -0.12574468561147706)},
#         # {'disgust': (0.4573438234998662, -0.693718996983959)},
#         # {'anger': (0.5667776418527468, -0.45273569245960493)},
#         # {'contempt': (0.5828403601946657, -0.5145737456826646)},
#         # {'None': (-0.16301792557996833, 0.1225122491637435)},
#         # {'Uncertain': (-2.0, -2.0)},
#         # {'Non-Face': (-2.0, -2.0)}]
