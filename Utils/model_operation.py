import os
import pickle
import shutil
import time

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from Networks.resnet import ResNet18, ResNet34, ResNet50, ResNetFashion
from Utils.misc import AverageMeter, accuracy
from Utils.misc import mkdir_p


def create_nn(model_name='cifar10_resnet18', num_classes=10, use_cuda=True):
    if model_name.startswith('fashion'):
        model = ResNetFashion()
    elif model_name.startswith('cifar'):
        if model_name.endswith('18'):
            model = ResNet18(num_classes=num_classes)
        elif model_name.endswith('34'):
            model = ResNet34(num_classes=num_classes)
        elif model_name.endswith('50'):
            model = ResNet50(num_classes=num_classes)
        elif model_name.endswith('101'):
            model = ResNet34(num_classes=num_classes)
        elif model_name.endswith('152'):
            model = ResNet50(num_classes=num_classes)
        else:
            print('Invalid name! The neural network cannot be created.')
            return
    elif model_name.startswith('tiny'):
        if model_name.endswith('18'):
            model = resnet18()
        elif model_name.endswith('34'):
            model = resnet34()
        elif model_name.endswith('50'):
            model = resnet50()
        elif model_name.endswith('101'):
            model = resnet101()
        elif model_name.endswith('152'):
            model = resnet152()
        else:
            print('Invalid name! The neural network cannot be created.')
            return
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc.out_features = num_classes
    else:
        print('Invalid name! The neural network cannot be created.')
        return
    if use_cuda:
        model = model.cuda()
    return model


def adjust_learning_rate(optimizer, epoch):
    lr = 0.01
    if epoch >= 50 and epoch < 100:
        lr = 0.001
    elif epoch > 100:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch(trainloader, model, criterion, optimizer, use_cuda=True):
    model.train()  # switch to train mode
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs, targets = torch.autograd.Variable(inputs).float(), torch.as_tensor(torch.autograd.Variable(targets), dtype=torch.long)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)  # inputs shape = [-1, 3, 32, 32]
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # plot progress
        if (batch_idx + 1) % 100 == 0:
            print('({batch}/{size}) | Loss: {loss:.4f} | top1: {top1: .2%} | top5: {top5: .2%}'.format(
                batch=batch_idx + 1, size=len(trainloader), loss=losses.avg, top1=top1.avg, top5=top5.avg))
    return (losses.avg, top1.avg, top5.avg)


def test(testloader, model, criterion, use_cuda=True):
    model.eval()  # switch to evaluate mode
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # inputs, targets = torch.autograd.Variable(inputs).float(), torch.as_tensor(torch.autograd.Variable(targets), dtype=torch.long)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # plot progress
        if (batch_idx + 1) % 100 == 0:
            print('({batch}/{size}) | Loss: {loss:.4f} | top1: {top1: .2%} | top5: {top5: .2%}'.format(
                batch=batch_idx + 1, size=len(testloader), loss=losses.avg, top1=top1.avg, top5=top5.avg))
    return (losses.avg, top1.avg, top5.avg)


def evaluate(test_loader, criterion, model_name='cifar10', num_classes=10, use_cuda=True):
    model, train_acc, test_acc = load_best_target_nn(model_name, num_classes, use_cuda)
    start = time.time()
    test_loss, test_acc1, test_acc5 = test(test_loader, model, criterion, use_cuda)
    end = time.time()
    print('The prediction last ', end - start, ' seconds')
    print("Train acc = {:.2%}, Test acc = {:.2%}".format(train_acc, test_acc))
    print("Test loss = {:.4f}, Test top1 = {:.2%}, Test top5 = {:.2%}".format(test_loss, test_acc1, test_acc5))


def save_nn(state, test_acc, best_acc, checkpoint='mnist', filename='checkpoint.pth.tar'):
    model_path = '../Models/' + checkpoint
    if not os.path.isdir(model_path):
        mkdir_p(model_path)
    filepath = os.path.join(model_path, filename)
    torch.save(state, filepath)
    if test_acc > best_acc:
        shutil.copyfile(filepath, os.path.join(model_path, 'model_best.pth.tar'))
        best_acc = test_acc
    return best_acc


def load_best_target_nn(model_name='cifar10_resnet18', num_classes=10, use_cuda=True):
    model = create_nn(model_name=model_name, num_classes=num_classes, use_cuda=use_cuda)
    model_state = torch.load('../Models/' + model_name + '/model_best.pth.tar')
    state_dict, train_acc, test_acc = model_state['state_dict'], model_state['train_acc'], model_state['test_acc']
    model.load_state_dict(state_dict)
    return model, train_acc, test_acc


def save_ml_mia_model(model, model_name='mlp_mnist', defense=''):
    model_path = '../Models/mia_ml/' + defense
    if not os.path.isdir(model_path):
        mkdir_p(model_path)
    with open(os.path.join(model_path, model_name + '.pkl'), 'wb') as wf:
        pickle.dump(model, wf)


def load_ml_mia_model(model_name='mlp_mnist', defense=''):
    model_path = '../Models/mia_ml/' + defense
    with open(os.path.join(model_path, model_name + '.pkl'), 'rb') as rf:
        model = pickle.load(rf)
    return model


def save_shadow_models(model, model_name='mnist_1', defense=''):
    model_path = '../Models/shadow/' + defense
    if not os.path.isdir(model_path):
        mkdir_p(model_path)
    with open(os.path.join(model_path, model_name + '.pkl'), 'wb') as wf:
        pickle.dump(model, wf)


def load_shadow_models(model_name='mnist_1', defense=''):
    model_path = '../Models/shadow/' + defense
    with open(os.path.join(model_path, model_name + '.pkl'), 'rb') as rf:
        model = pickle.load(rf)
    return model
