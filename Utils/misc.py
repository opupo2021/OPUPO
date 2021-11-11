from __future__ import print_function, absolute_import

import errno
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def initialization(manualSeed=2021):
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)
    return use_cuda


def utility_loss(mat1, mat2):
    return np.mean(np.linalg.norm(mat1 - mat2, axis=1))


def shuffle_dataset(x, y):
    index = np.arange(len(x))
    random.shuffle(index)
    return x[index], y[index]


def tensor_rounding(tensor, n=1):
    return torch.round(tensor * (10 ** n)) / (10 ** n)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # transpose size = (5,100)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res


if __name__ == '__main__':
    a = torch.from_numpy(np.array([[0.423423, 0.4984234, 0.454323], [0.074327, 0.86427, 0.78623]]))
    print(a)
    print(tensor_rounding(a, 1))
    print(tensor_rounding(a, 2))
    print(tensor_rounding(a, 3))
