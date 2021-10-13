import torchvision.models as models
import torch.nn as nn
import os
from fer_strong_baseline.models.model_utils import load_pretrained, initialize_weight_goog
import pdb
import torch

'''
第一层conv改为4通道
'''
class Res18Feature4ch(nn.Module):
    def __init__(self, pretrained='imagenet_pretrained', num_classes=7, drop_rate=0):
        super(Res18Feature4ch, self).__init__()
        self.drop_rate = drop_rate
        
        # self.tail_1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn_1 = nn.BatchNorm2d(1, affine=True)
        # self.relu_1 = nn.ReLU()
        # self.tail_2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn_2 = nn.BatchNorm2d(1, affine=True)
        # self.relu_2 = nn.ReLU()
        
        # imagenet pretrained model, should use!!!
        resnet = models.resnet18(True)
        # resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # resnet.bn1 = nn.BatchNorm2d(4)

        # self.feature = nn.Sequential(*list(resnet.children())[:-1]) # before avgpool

        # conv, batchnorm, relu
        self.features_cbr = nn.Sequential(*list(resnet.children())[:4])  # after avgpool 512x1

        self.features_stage1 = nn.Sequential(*list(resnet.children())[4])
        self.features_stage2 = nn.Sequential(*list(resnet.children())[5])
        self.features_stage3 = nn.Sequential(*list(resnet.children())[6])
        self.features_stage4 = nn.Sequential(*list(resnet.children())[7])
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())
        
        self.sigmoid = nn.Sigmoid()
        if pretrained == '':
            for m in self.modules():
                initialize_weight_goog(m)
        # if os.path.exists(pretrained):
        #     self.features = load_pretrained(self.features, pretrained)

    def forward(self, x):
        bgr = x[:,:3,:,:]
        # gaussian_mask = x[:,3,:56,:56]
        gaussian_mask = x[:,3,:,:]
        gaussian_mask = torch.unsqueeze(gaussian_mask, 1)
        # x = self.tail(x)

        # x2 = self.tail_1(gaussian_mask)
        # x2 = self.bn_1(x2)
        # x2 = self.relu_1(x2)

        # x2 = self.tail_2(x2)
        # x2 = self.bn_2(x2)
        # x2 = self.relu_2(x2)

        # torch.Size([1, 1, 56, 56])
        # x2 = self.sigmoid(x2)
                                                
        # torch.Size([1, 64, 56, 56])
        x1 = self.features_cbr(bgr)

        # torch.Size([1, 64, 56, 56])
        x1 = self.features_stage1(x1)

       
        # add attention 
        try:
            # x = x1 * (1 + 0 * x2)
            x = x1
            # x = x1 * gaussian_mask
        except Exception as e:
            pdb.set_trace()
        # x = x1

        # torch.Size([1, 128, 28, 28])
        x = self.features_stage2(x)

        # torch.Size([1, 256, 14, 14])
        x = self.features_stage3(x)

        # torch.Size([1, 512, 7, 7])
        x = self.features_stage4(x)

        x = self.pooling(x)

        # x = self.feature(bgr)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)

        try:
            attention_weights = self.alpha(x)
        except Exception as e:
            pdb.set_trace()
            

        out = attention_weights * self.fc(x)
        return attention_weights, out



if __name__ == "__main__":
    import numpy as np
    import torch
    model = Res18Feature4ch()
    print(model)
    
    input = np.zeros((1, 4, 224, 224))
    input = torch.from_numpy(input).float()
    x = model(input)

    print(x)


