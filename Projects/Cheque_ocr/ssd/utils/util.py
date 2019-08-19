from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division


import torch.nn as nn

from ssd.layers import CondensingLinear, CondensingConv


count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def convert_model(model):
    for m in model._modules:
        child = model._modules[m]
        if is_leaf(child):
            if isinstance(child, nn.Linear):
                model._modules[m] = CondensingLinear(child, 0.5)
                del(child)
        elif is_pruned(child):
            model._modules[m] = CondensingConv(child)
            del(child)
        else:
            convert_model(child)