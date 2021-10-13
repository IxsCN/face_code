import torchvision.models as models
import torch.nn as nn
import os
from fer_strong_baseline.models.model_utils import load_pretrained, initialize_weight_goog
import torch

class Res18Feature(nn.Module):
    def __init__(self, pretrained='imagenet_pretrained', num_classes=7, drop_rate=0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        # imagenet pretrained model, should use!!!
        resnet = models.resnet18(True)
        # self.tail = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        # self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1
        self.features_cbr = nn.Sequential(*list(resnet.children())[:4])  # after avgpool 512x1
        self.features_stage1 = nn.Sequential(*list(resnet.children())[4])
        self.features_stage2 = nn.Sequential(*list(resnet.children())[5])
        self.features_stage3 = nn.Sequential(*list(resnet.children())[6])
        self.features_stage4 = nn.Sequential(*list(resnet.children())[7])
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))


        # self.tail_1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn_1 = nn.BatchNorm2d(1, affine=True)
        # self.relu_1 = nn.ReLU()
        # self.tail_2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn_2 = nn.BatchNorm2d(1, affine=True)
        # self.relu_2 = nn.ReLU()


        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())
        if pretrained == '':
            for m in self.modules():
                initialize_weight_goog(m)

        load_dict = {}
        if os.path.exists(pretrained):
            state_dict = torch.load(pretrained)['state_dict']

            ## cbr
            # conv 1
            load_dict.update({'features_cbr.0.weight': state_dict['module.conv1.weight']})
            # bn 1
            load_dict.update({'features_cbr.1.weight': state_dict['module.bn1.weight']})
            load_dict.update({'features_cbr.1.bias': state_dict['module.bn1.bias']})
            load_dict.update({'features_cbr.1.running_mean': state_dict['module.bn1.running_mean']})
            load_dict.update({'features_cbr.1.running_var': state_dict['module.bn1.running_var']})

            # features s1  conv bn conv bn 
            for i in range(1, 5):
                for j in range(2):
                    load_dict.update({'features_stage{}.{}.conv1.weight'.format(i,j):  state_dict['module.layer{}.{}.conv1.weight'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn1.weight'.format(i,j):  state_dict['module.layer{}.{}.bn1.weight'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn1.bias'.format(i,j):  state_dict['module.layer{}.{}.bn1.bias'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn1.running_mean'.format(i,j):  state_dict['module.layer{}.{}.bn1.running_mean'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn1.running_var'.format(i,j):  state_dict['module.layer{}.{}.bn1.running_var'.format(i,j)]})
                    
                    load_dict.update({'features_stage{}.{}.conv2.weight'.format(i,j):  state_dict['module.layer{}.{}.conv2.weight'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn2.weight'.format(i,j):  state_dict['module.layer{}.{}.bn2.weight'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn2.bias'.format(i,j):  state_dict['module.layer{}.{}.bn2.bias'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn2.running_mean'.format(i,j):  state_dict['module.layer{}.{}.bn2.running_mean'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.bn2.running_var'.format(i,j):  state_dict['module.layer{}.{}.bn2.running_var'.format(i,j)]})
                if i in range(2,5):
                    j = 0
                    load_dict.update({'features_stage{}.{}.downsample.0.weight'.format(i,j):  state_dict['module.layer{}.{}.downsample.0.weight'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.downsample.1.weight'.format(i,j):  state_dict['module.layer{}.{}.downsample.1.weight'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.downsample.1.bias'.format(i,j):  state_dict['module.layer{}.{}.downsample.1.bias'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.downsample.1.running_mean'.format(i,j):  state_dict['module.layer{}.{}.downsample.1.running_mean'.format(i,j)]})
                    load_dict.update({'features_stage{}.{}.downsample.1.running_var'.format(i,j):  state_dict['module.layer{}.{}.downsample.1.running_var'.format(i,j)]})
                    
            # self.state_dict()['fc.weight'] = state_dict['module.fc.weight']
            # self.state_dict()['fc.bias'] = state_dict['module.fc.bias']
            load_dict.update({"fc.weight":torch.zeros_like(self.state_dict()["fc.weight"])})
            load_dict.update({"fc.bias":torch.zeros_like(self.state_dict()["fc.bias"])})
            load_dict.update({"alpha.0.weight":torch.zeros_like(self.state_dict()["alpha.0.weight"])})
            load_dict.update({"alpha.0.bias":torch.zeros_like(self.state_dict()["alpha.0.bias"])})

            self.load_state_dict(load_dict)
            # self.state_dict()['features_cbr.0.weight']
            '''
            [
            'module.feature.weight', 
            'module.feature.bias', 
            '''

            '''
            'features_cbr.1.num_batches_tracked', 
            'features_stage1.0.bn1.num_batches_tracked', 
            'features_stage1.0.bn2.num_batches_tracked', 
            'features_stage1.1.bn1.num_batches_tracked', 
            'features_stage2.0.bn1.num_batches_tracked', 
            'features_stage2.0.bn2.num_batches_tracked', 
            'features_stage2.0.downsample.1.num_batches_tracked', 
            'features_stage2.1.bn2.num_batches_tracked',
            'features_stage3.0.bn1.num_batches_tracked', 
            'features_stage3.0.bn2.num_batches_tracked', 
            'features_stage3.0.downsample.1.num_batches_tracked', 
            'features_stage3.1.bn1.num_batches_tracked', 
            'features_stage3.1.bn2.num_batches_tracked', 
            'features_stage4.0.bn1.num_batches_tracked',
            'features_stage4.0.bn2.num_batches_tracked',
            'features_stage4.0.downsample.1.num_batches_tracked',
            'features_stage4.1.bn1.num_batches_tracked', 
            'features_stage4.1.bn2.num_batches_tracked',

            'alpha.0.weight', 'alpha.0.bias'])
            '''

            # import pdb
            # pdb.set_trace()
            # self.load_state_dict(state_dict)
        # if os.path.exists(pretrained):
            # self.features = load_pretrained(self.features, pretrained)

    def forward(self, x):
        # x = self.tail(x)
        # x = self.features(x)

        x1 = self.features_cbr(x)
        x1 = self.features_stage1(x1)
        
        import numpy as np
        import torch
        gaussian_mask = np.ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        gaussian_mask = torch.from_numpy(gaussian_mask).float()
        gaussian_mask = gaussian_mask.cuda()
        # x2 = self.tail_1(gaussian_mask)
        # x2 = self.bn_1(x2)
        # x2 = self.relu_1(x2)
        # x2 = self.tail_2(x2)
        # x2 = self.bn_2(x2)
        # x2 = self.relu_2(x2)

        # fusion 
        # x = x1 * (1 + 0 * x2)
        x = x1
        
        x = self.features_stage2(x)
        x = self.features_stage3(x)
        x = self.features_stage4(x)
        x = self.pooling(x)


        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out



