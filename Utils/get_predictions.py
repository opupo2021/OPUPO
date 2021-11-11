import torch
import torch.nn as nn

from Defenses.opupo import opupo2
from Utils.data_split import cifar10_train_val_test_split, cifar100_train_val_test_split, fashion_train_val_test_split
from Utils.misc import initialization
from Utils.model_operation import load_best_target_nn

num_samples = 500
times = 10


def get_predictions(model_name='mnist', num_classes=10, ratio=1.0, use_cuda=True):
    model, _, _ = load_best_target_nn(model_name, num_classes, use_cuda)
    n_samples = int(num_samples * ratio)
    if model_name.startswith('fashion'):
        _, train_loader, _, val_loader, _, test_loader = fashion_train_val_test_split(n_samples, n_samples, n_samples)
    if model_name.startswith('cifar10_'):
        _, train_loader, _, val_loader, _, test_loader = cifar10_train_val_test_split(n_samples, n_samples, n_samples)
    elif model_name.startswith('cifar100_'):
        _, train_loader, _, val_loader, _, test_loader = cifar100_train_val_test_split(n_samples, n_samples, n_samples)

    train_ins, test_ins, train_outs, test_outs = [], [], [], []
    train_in_labels, test_in_labels, train_out_labels, test_out_labels = [], [], [], []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda() if use_cuda else inputs
        if batch_idx < times:
            train_ins.append(model(inputs).cpu().detach())
            train_in_labels.append(targets.cpu().detach())
        elif batch_idx < times * 2:
            test_ins.append(model(inputs).cpu().detach())
            test_in_labels.append(targets.cpu().detach())
        else:
            break
    # start = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda() if use_cuda else inputs
        train_outs.append(model(inputs).cpu().detach())
        train_out_labels.append(targets.cpu().detach())
    # end = time.time()
    # print('Original = ', (end-start)/(len(train_out_labels)*len(train_out_labels[0]))*1000)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda() if use_cuda else inputs
        test_outs.append(model(inputs).cpu().detach())
        test_out_labels.append(targets.cpu().detach())
    train_in = torch.cat(train_ins, dim=0)
    train_out = torch.cat(train_outs, dim=0)
    test_in = torch.cat(test_ins, dim=0)
    test_out = torch.cat(test_outs, dim=0)
    train_in_label = torch.cat(train_in_labels, dim=0)
    train_out_label = torch.cat(train_out_labels, dim=0)
    test_in_label = torch.cat(test_in_labels, dim=0)
    test_out_label = torch.cat(test_out_labels, dim=0)
    train_X, test_X = torch.cat((train_in, train_out)), torch.cat((test_in, test_out))
    train_Y, test_Y = torch.cat((torch.ones(n_samples * times), torch.zeros(n_samples * times))), torch.cat(
        (torch.ones(n_samples * times), torch.zeros(n_samples * times)))

    return nn.Softmax(dim=1)(train_X), train_Y, nn.Softmax(dim=1)(test_X), test_Y, torch.cat(
        (train_in_label, train_out_label)), torch.cat((test_in_label, test_out_label))


def get_predictions_opupo(kappa=5, delta=0.5, eta=1, use_perturbation=False, model_name='mnist', num_classes=10,
                          ratio=1.0, use_cuda=True):
    train_X, train_Y, test_X, test_Y, train_L, test_L = get_predictions(model_name, num_classes, ratio, use_cuda)
    # start = time.time()
    train_X = opupo2(train_X, kappa, delta, eta, use_perturbation=use_perturbation)
    # end = time.time()
    # print('Obfuscation = ', (end-start)/train_X.size(0)*1000)
    test_X = opupo2(test_X, kappa, delta, eta, use_perturbation=use_perturbation)
    return train_X, train_Y, test_X, test_Y, train_L, test_L


def get_logits(model_name='mnist', num_classes=10, ratio=1.0, use_cuda=True):
    model, _, _ = load_best_target_nn(model_name, num_classes, use_cuda)
    n_samples = int(num_samples * ratio)
    if model_name.startswith('fashion'):
        _, train_loader, _, val_loader, _, test_loader = fashion_train_val_test_split(n_samples, n_samples, n_samples)
    if model_name.startswith('cifar10_'):
        _, train_loader, _, val_loader, _, test_loader = cifar10_train_val_test_split(n_samples, n_samples, n_samples)
    elif model_name.startswith('cifar100_'):
        _, train_loader, _, val_loader, _, test_loader = cifar100_train_val_test_split(n_samples, n_samples, n_samples)

    train_ins, test_ins, train_outs, test_outs = [], [], [], []
    train_in_labels, test_in_labels, train_out_labels, test_out_labels = [], [], [], []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda() if use_cuda else inputs
        if batch_idx < times:
            train_ins.append(model(inputs).cpu().detach())
            train_in_labels.append(targets.cpu().detach())
        elif batch_idx < times * 2:
            test_ins.append(model(inputs).cpu().detach())
            test_in_labels.append(targets.cpu().detach())
        else:
            break
    # start = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda() if use_cuda else inputs
        train_outs.append(model(inputs).cpu().detach())
        train_out_labels.append(targets.cpu().detach())
    # end = time.time()
    # print('Original = ', (end-start)/(len(train_out_labels)*len(train_out_labels[0]))*1000)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda() if use_cuda else inputs
        test_outs.append(model(inputs).cpu().detach())
        test_out_labels.append(targets.cpu().detach())
    train_in = torch.cat(train_ins, dim=0)
    train_out = torch.cat(train_outs, dim=0)
    test_in = torch.cat(test_ins, dim=0)
    test_out = torch.cat(test_outs, dim=0)
    train_in_label = torch.cat(train_in_labels, dim=0)
    train_out_label = torch.cat(train_out_labels, dim=0)
    test_in_label = torch.cat(test_in_labels, dim=0)
    test_out_label = torch.cat(test_out_labels, dim=0)
    train_X, test_X = torch.cat((train_in, train_out)), torch.cat((test_in, test_out))
    train_Y, test_Y = torch.cat((torch.ones(n_samples * times), torch.zeros(n_samples * times))), torch.cat(
        (torch.ones(n_samples * times), torch.zeros(n_samples * times)))
    return train_X, train_Y, test_X, test_Y, torch.cat((train_in_label, train_out_label)), torch.cat(
        (test_in_label, test_out_label))


if __name__ == '__main__':
    use_cuda = initialization(2021)
    delta = 0.5
    eta = 1
    use_perturbation = False
    model_name = 'location'
    num_classes = 30
    train_X, train_Y, test_X, test_Y, train_L, test_L = get_predictions_opupo(delta, eta, use_perturbation, model_name,
                                                                              num_classes)
    test_in = test_X[:len(test_X) // 2]
    test_out = test_X[len(test_X) // 2:]
