from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .TransformerModel import TransformerModel
from .SAT import SAT

def setup(opt):
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Transformer
    elif opt.caption_model == 'transformer':
        model = TransformerModel(opt)
    # Semi-autogressive Transformer
    elif opt.caption_model == 'sat':
        model = SAT(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from

        transformer_state_dict = torch.load(os.path.join(opt.start_from, 'model-best.pth'))
        model_dict = model.state_dict()
        petrained_dict = {k: v for k, v in transformer_state_dict.items() if not k.startswith('model.generator_length')}
        model_dict.update(petrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))

    return model
