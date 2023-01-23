import os
import cifar10.model_loader
from MS_Architecture import MazeSolvingNN_DT
import torch

def load(dataset, model_name, model_file, data_parallel=False, num_iter=None):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'mazes_dt':
        if num_iter == None:
            net = MazeSolvingNN_DT(num_iter=20)
            net.load_state_dict(torch.load(model_file))
        else:
            init_net = MazeSolvingNN_DT(num_iter=20)
            init_net.load_state_dict(torch.load(model_file))
            net = MazeSolvingNN_DT(num_iter=num_iter)
            net.expand_iterations(init_net)

    return net
