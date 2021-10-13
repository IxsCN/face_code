import math
import torch
import torch.nn as nn
import pdb

def load_pretrained(model, pretrained_model_weight):
    print("Loading pretrained weights...", pretrained_model_weight)
    pretrained = torch.load(pretrained_model_weight)
    try:
        pretrained_state_dict = pretrained['state_dict']
    except Exception as e:
        # pdb.set_trace()
        pretrained_state_dict = pretrained
    model_state_dict = model.state_dict()

    loaded_keys = 0
    total_keys = 0
    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
            pass
        else:
            model_state_dict[key] = pretrained_state_dict[key]
            total_keys += 1
            if key in model_state_dict:
                loaded_keys += 1

        # pretrain model {a:1}
        # model_s {b:0}

        # model_s {b:0, a:1}


    print("Loaded params num:", loaded_keys)
    print("Total params num:", total_keys)

    model.load_state_dict(model_state_dict, strict=False)
    # model.load_state_dict(model_state_dict)
    return model

def initialize_weight_goog(m, n=''):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    # if isinstance(m, CondConv2d):
        # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # init_weight_fn = get_condconv_initializer(
            # lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        # init_weight_fn(m.weight)
        # if m.bias is not None:
            # m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()
