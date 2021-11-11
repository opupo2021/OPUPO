import warnings

warnings.filterwarnings(action='ignore')
import torch

from sklearn.naive_bayes import GaussianNB
from Utils.get_predictions import get_predictions, get_predictions_opupo
from Utils.model_operation import save_shadow_models, load_shadow_models
from sklearn.metrics import accuracy_score
import numpy as np
from Utils.misc import initialization


kappa, delta, eta = 5, 0.9, 1.0
use_perturbation = True


def get_shadow_data(target_name='cifar10', num_classes=10, use_cuda=True):
    train_X, train_Y, test_X, test_Y, train_L, test_L = get_predictions(model_name=target_name, num_classes=num_classes,
                                                                        use_cuda=use_cuda)
    shadow_train_total = torch.cat((train_X, train_Y.view(-1, 1), train_L.view(-1, 1)), dim=1)
    shadow_test_total = torch.cat((test_X, test_Y.view(-1, 1), test_L.view(-1, 1)), dim=1)
    shadow_train_sets = []
    shadow_test_sets = []
    for i in range(num_classes):
        shadow_train_sets.append(shadow_train_total[shadow_train_total[:, -1] == i][:, :-1])
        shadow_test_sets.append(shadow_test_total[shadow_test_total[:, -1] == i][:, :-1])
    return shadow_train_sets, shadow_test_sets


def get_shadow_data_opupo(kappa=5, delta=0.9, eta=1, use_perturbation=False, target_name='cifar10', num_classes=10,
                          use_cuda=True):
    train_X, train_Y, test_X, test_Y, train_L, test_L = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta,
                                                                              use_perturbation=use_perturbation,
                                                                              model_name=target_name,
                                                                              num_classes=num_classes,
                                                                              use_cuda=use_cuda)
    shadow_train_total = torch.cat((train_X, train_Y.view(-1, 1), train_L.view(-1, 1)), dim=1)
    shadow_test_total = torch.cat((test_X, test_Y.view(-1, 1), test_L.view(-1, 1)), dim=1)
    shadow_train_sets = []
    shadow_test_sets = []
    for i in range(num_classes):
        shadow_train_sets.append(shadow_train_total[shadow_train_total[:, -1] == i][:, :-1])
        shadow_test_sets.append(shadow_test_total[shadow_test_total[:, -1] == i][:, :-1])
    return shadow_train_sets, shadow_test_sets


def shadow_evaluate(target_name='cifar10', num_classes=10, use_cuda=True):
    shadow_train_sets, shadow_test_sets = get_shadow_data(target_name, num_classes, use_cuda)
    train_accs = []
    test_accs = []
    correct_num = 0
    total_num = 0
    for i in range(num_classes):
        train_x, train_y = shadow_train_sets[i][:, :-1], shadow_train_sets[i][:, -1]
        test_x, test_y = shadow_test_sets[i][:, :-1], shadow_test_sets[i][:, -1]
        attack_model = load_shadow_models(target_name + str(i), '')
        train_preds = attack_model.predict(train_x)
        test_preds = attack_model.predict(test_x)
        train_accs.append(accuracy_score(train_preds, train_y))
        test_accs.append(accuracy_score(test_preds, test_y))
        correct_num += np.sum(test_preds == test_y.numpy())
        total_num += test_y.size(0)
    print(train_accs)
    print(test_accs)
    print(correct_num / total_num)


def shadow_nb(shadow_train_sets, shadow_test_sets, target_name='cifar10', num_classes=10, use_cuda=True):
    # shadow_train_sets, shadow_test_sets = get_shadow_data(target_name, num_classes, use_cuda)
    var_smoothing = [1e-9, 1e-7, 1e-5, 1e-3, 0.1, 1]
    for j in range(6):
        print(var_smoothing[j])
        correct_num = 0
        total_num = 0
        for i in range(num_classes):
            train_x, train_y = shadow_train_sets[i][:, :-1], shadow_train_sets[i][:, -1]
            test_x, test_y = shadow_test_sets[i][:, :-1], shadow_test_sets[i][:, -1]
            attack_model = GaussianNB(var_smoothing=var_smoothing[j]).fit(train_x, train_y)
            save_shadow_models(attack_model, target_name + str(i), '')
            test_preds = attack_model.predict(test_x)
            correct_num += np.sum(test_preds == test_y.numpy())
            total_num += test_y.size(0)
        print(target_name, ' NB: ', correct_num / total_num)


def shadow_train_undefend(model_name='cifar10_resnet18', num_classes=10):
    print(model_name)
    shadow_train_sets, shadow_test_sets = get_shadow_data(model_name, num_classes)
    correct_num = 0
    total_num = 0
    for i in range(num_classes):
        train_x, train_y = shadow_train_sets[i][:, :-1], shadow_train_sets[i][:, -1]
        test_x, test_y = shadow_test_sets[i][:, :-1], shadow_test_sets[i][:, -1]
        attack_model = GaussianNB(var_smoothing=1e-5).fit(train_x, train_y)
        save_shadow_models(attack_model, model_name + str(i), '')
        test_preds = attack_model.predict(test_x)
        correct_num += np.sum(test_preds == test_y.numpy())
        total_num += test_y.size(0)
    print(model_name, correct_num / total_num)


def shadow_opupo_unaware(model_name='cifar10_resnet18', num_classes=10):
    shadow_train_sets, shadow_test_sets = get_shadow_data_opupo(kappa=kappa, delta=delta, eta=eta,
                                                                use_perturbation=use_perturbation,
                                                                target_name=model_name, num_classes=num_classes)
    correct_num = 0
    total_num = 0
    for i in range(num_classes):
        # train_x, train_y = shadow_train_sets[i][:, :-1], shadow_train_sets[i][:, -1]
        test_x, test_y = shadow_test_sets[i][:, :-1], shadow_test_sets[i][:, -1]
        attack_model = load_shadow_models(model_name=model_name + str(i), defense='')
        test_preds = attack_model.predict(test_x)
        correct_num += np.sum(test_preds == test_y.numpy())
        total_num += test_y.size(0)
    print(model_name, correct_num / total_num)


def shadow_opupo_aware(model_name='cifar10_resnet18', num_classes=10, defense=''):
    print(model_name)
    shadow_train_sets, shadow_test_sets = get_shadow_data_opupo(kappa=kappa, delta=delta, eta=eta,
                                                                use_perturbation=use_perturbation,
                                                                target_name=model_name, num_classes=num_classes)
    correct_num = 0
    total_num = 0
    for i in range(num_classes):
        train_x, train_y = shadow_train_sets[i][:, :-1], shadow_train_sets[i][:, -1]
        test_x, test_y = shadow_test_sets[i][:, :-1], shadow_test_sets[i][:, -1]
        attack_model = GaussianNB(var_smoothing=1e-5).fit(train_x, train_y)
        save_shadow_models(attack_model, model_name + str(i), defense)
        test_preds = attack_model.predict(test_x)
        correct_num += np.sum(test_preds == test_y.numpy())
        total_num += test_y.size(0)
    print(model_name, correct_num / total_num)


def shadow_attack(shadow_train_sets, shadow_test_sets, model_name='mnist'):
    nums = {'fashion_resnet18': 10, 'cifar10_resnet18': 10, 'cifar100_resnet18': 100}
    print(model_name)
    correct_num = 0
    total_num = 0
    for i in range(nums[model_name]):
        train_x, train_y = shadow_train_sets[i][:, :-1], shadow_train_sets[i][:, -1]
        test_x, test_y = shadow_test_sets[i][:, :-1], shadow_test_sets[i][:, -1]
        attack_model = GaussianNB(var_smoothing=1e-5).fit(train_x, train_y)
        test_preds = attack_model.predict(test_x)
        correct_num += np.sum(test_preds == test_y.numpy())
        total_num += test_y.size(0)
    print(model_name, correct_num / total_num)


if __name__ == '__main__':
    use_cuda = initialization(2021)
    nums = {'fashion_resnet18': 10, 'cifar10_resnet18': 10, 'cifar100_resnet18': 100}
    # Attack Undefended Models
    # target = 'fashion_resnet18'
    target = 'cifar10_resnet18'
    # target = 'cifar100_resnet18'
    shadow_train_undefend(target, nums[target])

    # Defense-unaware adversaries
    shadow_opupo_unaware(target, nums[target])

    # Defense-aware adversary
    defense = 'opupo_' + '_' + str(kappa) + '_' + str(delta) + '_' + str(eta) + '_' + str(use_perturbation)
    shadow_opupo_aware(target, nums[target], defense)