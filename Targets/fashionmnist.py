import torch.nn as nn
import torch.optim as optim

from Utils.data_split import fashion_train_val_test_split
from Utils.misc import initialization
from Utils.model_operation import save_nn, train_epoch, test, create_nn, evaluate


def fashion_train(train_loader, test_loader, use_cuda=True):
    model_name = 'fashion_resnet18'
    num_classes = 10
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    model = create_nn(model_name, num_classes, use_cuda)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    epochs = 50

    best_acc = 0.0
    for epoch in range(epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, lr))
        train_loss, train_acc1, train_acc5 = train_epoch(train_loader, model, criterion, optimizer, use_cuda)
        test_loss, test_acc1, test_acc5 = test(test_loader, model, criterion, use_cuda)
        print(
            '({e}/{size}) train_loss: {trainl:.4f} | test_loss: {testl:.4f} | train_top1: {train1:.2%} | train_top5: {train5: .2%} | test_top1: {test1: .2%} | test_top5: {test5: .2%}'.format(
                e=epoch + 1, size=epochs, trainl=train_loss, testl=test_loss, train1=train_acc1, train5=train_acc5,
                test1=test_acc1, test5=test_acc5))
        # save model
        best_acc = save_nn(
            {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'train_acc': train_acc1, 'test_acc': test_acc1,
             'optimizer': optimizer.state_dict(), }, test_acc1, best_acc, checkpoint=model_name,
            filename='epoch%d' % epoch)
    print('Best acc:', best_acc)


if __name__ == '__main__':
    use_cuda = initialization(2021)
    _, train_loader, _, val_loader, _, test_loader = fashion_train_val_test_split(train_batch=128, val_batch=128,
                                                                                  test_batch=128)
    fashion_train(train_loader, test_loader, use_cuda)
    evaluate(test_loader, nn.CrossEntropyLoss(), 'fashion_resnet18', 10, use_cuda)
