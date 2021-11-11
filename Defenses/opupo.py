import numpy as np
import torch

from Utils.misc import initialization


def process_onehot_predictions(outputs):
    case = outputs[np.any(outputs >= 1, axis=1)]
    indexes = np.arange(len(case))
    max_indexes = np.argmax(case, axis=1)
    new_max_confs = np.random.uniform(0.98, 0.99, max_indexes.shape)
    rest_confs = 1 - new_max_confs
    case = np.random.rand(case.size).reshape(case.shape)
    case[indexes, max_indexes] = 0
    case /= np.sum(case, axis=1).repeat(case.shape[1]).reshape(case.shape)
    case *= rest_confs.repeat(case.shape[1]).reshape(case.shape)
    case[indexes, max_indexes] = new_max_confs
    return case


def perturbation(outputs, indexes, max_indexes, second_max_indexes, bottom, top, deltas):
    new_max_confs = outputs[indexes, max_indexes]
    new_second_max_confs = outputs[indexes, second_max_indexes]
    outputs[indexes, max_indexes] = 0
    outputs[indexes, second_max_indexes] = 0
    third_max_indexes = np.argmax(outputs, axis=1)
    new_third_max_confs = outputs[indexes, third_max_indexes]
    if outputs.shape[1] == 2:
        eps = np.random.uniform(
            np.maximum((bottom - deltas) * 0.5, (new_second_max_confs - new_max_confs) * 0.5),
            (top - deltas) * 0.5)
    elif outputs.shape[1] > 2:
        eps = np.random.uniform(
            np.maximum((bottom - deltas) * 0.5, (new_second_max_confs - new_max_confs) * 0.5),
            np.minimum((top - deltas) * 0.5, new_second_max_confs - new_third_max_confs))
    outputs[indexes, max_indexes] = new_max_confs + eps
    outputs[indexes, second_max_indexes] = new_second_max_confs - eps
    return outputs


def opupo(outputs, delta, eta, use_perturbation=True):
    outputs = outputs.numpy() if torch.is_tensor(outputs) else outputs
    outputs[np.any(outputs >= 1, axis=1)] = process_onehot_predictions(outputs)
    indexes = np.arange(len(outputs))
    max_indexes = np.argmax(outputs, axis=1)
    max_confs = outputs[indexes, max_indexes]
    deltas = np.array([delta for i in indexes])
    outputs[indexes, max_indexes] = 0
    second_max_indexes = np.argmax(outputs, axis=1)
    second_max_confs = outputs[indexes, second_max_indexes]
    ori_d = max_confs - second_max_confs
    bottom = ori_d * (1 - eta)
    top = ori_d * (1 - eta) + eta
    deltas = np.minimum(deltas, top)
    deltas = np.maximum(deltas, bottom)
    denominator = 1 - max_confs + second_max_confs
    outputs *= (1 - deltas).repeat(outputs.shape[1]).reshape(outputs.shape)
    outputs /= denominator.repeat(outputs.shape[1]).reshape(outputs.shape)
    outputs[indexes, max_indexes] = (deltas * (1 - max_confs) + second_max_confs) / denominator
    if use_perturbation:
        outputs = perturbation(outputs, indexes, max_indexes, second_max_indexes, bottom, top, deltas)
    outputs = torch.FloatTensor(outputs)
    return outputs


def opupo2(outputs, kappa=5, delta=0.5, eta=1, use_perturbation=False):
    outputs = outputs.numpy() if torch.is_tensor(outputs) else outputs
    ret = outputs
    if kappa < outputs.shape[1]:
        sorted_index = np.argsort(outputs)
        sorted_index = sorted_index[:, ::-1]
        indexes = np.arange(outputs.shape[0]).reshape(-1, 1)
        outputs_ordered = outputs[indexes, sorted_index[:, :kappa]]
        outputs_not_ordered = outputs[indexes, sorted_index[:, kappa:]]
        ordered_sum = np.sum(outputs_ordered, axis=1).reshape(-1, 1)
        not_ordered_sum = np.sum(outputs_not_ordered, axis=1).reshape(-1, 1)
        outputs_ordered += (outputs_ordered / ordered_sum * not_ordered_sum)

        outputs_not_ordered = np.random.randn(outputs_not_ordered.shape[0], outputs_not_ordered.shape[1])

        redist = outputs_ordered[indexes, -1] / 2
        outputs_ordered *= (1 - redist)
        not_ordered_sum = np.sum(outputs_not_ordered, axis=1).reshape(-1, 1)
        outputs_not_ordered = outputs_not_ordered / not_ordered_sum * redist
        ret[indexes, sorted_index[:, :kappa]] = outputs_ordered
        ret[indexes, sorted_index[:, kappa:]] = outputs_not_ordered
    return opupo(ret, delta, eta, use_perturbation)


if __name__ == '__main__':
    use_cuda = initialization(2021)
    a = np.array([[0.9, 0.05, 0.03, 0.02], [0.15, 0.3, 0.5, 0.05], [0.2, 0.6, 0.05, 0.15]])
    b = np.array([[0.9, 0.05, 0.025, 0.025], [0.5, 0.3, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    c = np.array([[0.925, 0.075, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    print(opupo2(a, kappa=2))
    print(opupo2(b, kappa=2))
    print(opupo2(c, kappa=2))
    print(np.sum(opupo2(a, kappa=2), axis=1))
    print(np.sum(opupo2(b, kappa=2), axis=1))
    print(np.sum(opupo2(c, kappa=2), axis=1))
