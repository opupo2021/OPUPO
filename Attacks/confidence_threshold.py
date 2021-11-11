import numpy as np
import torch

from Utils.get_predictions import get_predictions, get_predictions_opupo
from Utils.misc import initialization


kappa, delta, eta = 5, 0.9, 1.0
use_perturbation = True


def max_probabilities(X):
    return torch.sort(X.max(1)[0])[0]


def confidence_threshold_attack(train_X, train_Y, test_X, test_Y):
    length = len(train_Y) // 2
    non_member = train_X[length:]
    sorted_non_member = max_probabilities(non_member)
    test_values = test_X.max(1)[0]
    thresholds = []
    mia_accs = []
    for p in np.linspace(0, 1, 101):
        t = int(length * p) if p < 1 else int(length * p) - 1
        threshold = sorted_non_member[t]
        mia_infer = torch.where(test_values >= threshold, 1, 0)
        mia_acc = (test_Y == mia_infer).sum().item() / len(test_Y)
        thresholds.append(threshold.item())
        mia_accs.append(mia_acc)
    idx = np.argmax(np.array(mia_accs))
    print('idx = %d - The best attack accuracy is %.2f%% with threshold set to %f' % (
    idx, (mia_accs[idx] * 100), thresholds[idx]))


def confidence_threshold_raw(model_name='mnist', num_classes=10, ratio=1.0, use_cuda=True):
    train_X, train_Y, test_X, test_Y, _, _ = get_predictions(model_name, num_classes, ratio, use_cuda)
    print(model_name)
    confidence_threshold_attack(train_X, train_Y, test_X, test_Y)


def confidence_threshold_opupo_unaware(model_name='mnist', num_classes=10, ratio=1.0, use_cuda=True):
    train_X, train_Y, test_X, test_Y, _, _ = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta,
                                                                   use_perturbation=use_perturbation,
                                                                   model_name=model_name, num_classes=num_classes,
                                                                   ratio=ratio, use_cuda=use_cuda)
    test_values = test_X.max(1)[0]
    print(model_name)
    thresholds = {'fashion_resnet18': 0.986489, 'cifar10_resnet18': 0.995106, 'cifar100_resnet18': 0.981086,
                  'tiny_resnet18': 0.047985}
    threshold = thresholds[model_name]
    mia_infer = torch.where(test_values >= threshold, 1, 0)
    mia_acc = (test_Y == mia_infer).sum().item() / len(test_Y)
    print(model_name, ' : ', mia_acc)


def confidence_threshold_opupo_aware(model_name='mnist', num_classes=10, ratio=1.0, use_cuda=True):
    train_X, train_Y, test_X, test_Y, _, _ = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta,
                                                                   use_perturbation=use_perturbation,
                                                                   model_name=model_name, num_classes=num_classes,
                                                                   ratio=ratio, use_cuda=use_cuda)
    print(model_name)
    confidence_threshold_attack(train_X, train_Y, test_X, test_Y)


if __name__ == '__main__':
    use_cuda = initialization(2021)

    # Attack Undefended Models
    # print('*'*66)
    # confidence_threshold_raw(model_name='fashion_resnet18')
    confidence_threshold_raw(model_name='cifar10_resnet18')
    # confidence_threshold_raw(model_name='cifar100_resnet18', num_classes=100)
    # print('*'*66)

    # # Defense-unaware Adversary
    # print('*'*66)
    # confidence_threshold_opupo_unaware(model_name='fashion_resnet18')
    confidence_threshold_opupo_unaware(model_name='cifar10_resnet18')
    # confidence_threshold_opupo_unaware(model_name='cifar100_resnet18', num_classes=100)
    # print('*'*66)
    #
    # # Defense-aware Adversary
    # print('*'*66)
    # confidence_threshold_opupo_aware(model_name='fashion_resnet18')
    confidence_threshold_opupo_aware(model_name='cifar10_resnet18')
    # confidence_threshold_opupo_aware(model_name='cifar100_resnet18', num_classes=100)
    # print('*'*66)
