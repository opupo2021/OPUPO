import numpy as np
from sklearn.svm import OneClassSVM

from Utils.get_predictions import get_predictions, get_predictions_opupo
from Utils.misc import initialization
from Utils.mmd import mmd_rbf
from Utils.model_operation import save_ml_mia_model, load_ml_mia_model


def blind_1class(non_members, targets, labels, target_name='mnist'):
    non_members, targets, labels = non_members.numpy(), targets.numpy(), labels.numpy()
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in range(4):
        attack_model = OneClassSVM(kernel=kernel[i])
        attack_model.fit(non_members)
        preds = attack_model.predict(targets)
        save_ml_mia_model(attack_model, 'onesvm_' + kernel[i] + '_' + target_name, defense='')
        preds[preds == 1] = 0
        preds[preds == -1] = 1
        print(kernel[i], np.mean(preds == labels))
    return np.mean(preds == labels)


def blind_1class_opupo_unaware(targets, labels, target_name='mnist'):
    kernel = {'fashion_resnet18': 'sigmoid', 'cifar10_resnet18': 'sigmoid', 'cifar100_resnet18': 'rbf',
              'tiny_resnet18': 'linear'}
    targets, labels = targets.numpy(), labels.numpy()
    attack_model = load_ml_mia_model('onesvm_' + kernel[target_name] + '_' + target_name, defense='')
    preds = attack_model.predict(targets)
    preds[preds == 1] = 0
    preds[preds == -1] = 1
    return np.mean(preds == labels)


def blind_1class_opupo_aware(non_members, targets, labels, target_name='mnist', defense=''):
    kernel = {'fashion_resnet18': 'sigmoid', 'cifar10_resnet18': 'sigmoid', 'cifar100_resnet18': 'rbf',
              'tiny_resnet18': 'linear'}
    targets, labels = targets.numpy(), labels.numpy()
    attack_model = OneClassSVM(kernel=kernel[target_name])
    attack_model.fit(non_members)
    preds = attack_model.predict(targets)
    save_ml_mia_model(attack_model, 'onesvm_' + kernel[target_name] + '_' + target_name, defense=defense)
    preds[preds == 1] = 0
    preds[preds == -1] = 1
    return np.mean(preds == labels)


def blind_diff(non_members, targets, labels):
    non_members, targets, labels = non_members.numpy(), targets.numpy(), labels.numpy()
    targets = np.concatenate((targets, labels.reshape(-1, 1), np.zeros_like(labels.reshape(-1, 1))), axis=1)
    pred_nonmembers = []
    flag = True
    while flag and targets.shape[0] > 0:
        flag = False
        initial_d = mmd_rbf(non_members, targets[:, :-2])
        for i in range(len(targets)):
            d = mmd_rbf(np.concatenate((non_members, targets[i, :-2].reshape(1, -1)), axis=0),
                        np.delete(targets, i, axis=0)[:, :-2])
            if d >= initial_d:
                pred_nonmembers.append(targets[i])
                targets[i, -1] = 1
                flag = True
        targets = targets[targets[:, -1] == 0]
    pred_nonmembers = np.array(pred_nonmembers)
    return (len(pred_nonmembers[pred_nonmembers[:, -2] == 0]) + len(targets[targets[:, -2] == 1])) / len(labels)


if __name__ == '__main__':
    use_cuda = initialization(2021)
    targets = ('fashion_resnet18', 'cifar10_resnet18', 'cifar100_resnet18')
    nums = {'fashion_resnet18': 10, 'cifar10_resnet18': 10, 'cifar100_resnet18': 100}
    kappa, delta, eta = 5, 0.9, 1.0
    use_perturbation = True

    # Attack Undefended Models
    # Blind Oneclass Attack
    # target = 'fashion_resnet18'
    target = 'cifar10_resnet18'
    # target = 'cifar100_resnet18'
    train_X, train_Y, test_X, test_Y, train_L, test_L = get_predictions(target, nums[target], use_cuda)
    acc = blind_1class(train_X[len(train_X) // 2:], test_X, test_Y, target)
    print(target, ' : ', acc)
    # Blind Diff Attack
    non_members = train_X[-50:]
    target_size = len(test_Y) // 10
    start = (len(test_Y) - target_size) // 2
    acc = blind_diff(non_members, test_X[start:start + target_size], test_Y[start:start + target_size])
    print(target, ' : ', acc)

    # Defense-unaware Adversaries
    train_X, _, _, _, _, _ = get_predictions(target, nums[target], use_cuda)
    _, _, test_X, test_Y, _, _ = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta,
                                                       use_perturbation=use_perturbation, model_name=target,
                                                       num_classes=nums[target])
    # Blind Oneclass Attack
    acc = blind_1class_opupo_unaware(test_X, test_Y, target)
    print(target, ' : ', acc)
    # Blind Diff Attack
    non_members = train_X[-50:]
    target_size = len(test_Y) // 10
    start = (len(test_Y) - target_size) // 2
    acc = blind_diff(non_members, test_X[start:start + target_size], test_Y[start:start + target_size])
    print(target, ' : ', acc)

    # Defense-aware Adversary
    train_X, train_Y, test_X, test_Y, train_L, test_L = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta,
                                                                              use_perturbation=use_perturbation,
                                                                              model_name=target,
                                                                              num_classes=nums[target])
    # Blind Oneclass Attack
    defense = 'opupo_' + '_' + str(kappa) + '_' + str(delta) + '_' + str(eta) + '_' + str(use_perturbation)
    acc = blind_1class_opupo_aware(train_X[len(train_X) // 2:], test_X, test_Y, target, defense)
    print(target, ' : ', acc)
    # Blind Diff Attack
    non_members = train_X[-50:]
    target_size = len(test_Y) // 10
    start = (len(test_Y) - target_size) // 2
    acc = blind_diff(non_members, test_X[start:start + target_size], test_Y[start:start + target_size])
    print(target, ' : ', acc)