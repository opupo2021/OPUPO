import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from Utils.misc import initialization
from Utils.model_operation import save_ml_mia_model, load_ml_mia_model
from Utils.get_predictions import get_predictions, get_predictions_opupo

kappa, delta, eta = 5, 0.9, 1.0
use_perturbation = True

def mlp_attack(train_x, train_y, test_x, test_y, target_name = 'mnist', defense = ''):
    mlp = MLPClassifier(random_state=2021).fit(train_x, train_y)
    acc = accuracy_score(test_y, mlp.predict(test_x))
    print(f'{acc : .2%}')
    save_ml_mia_model(mlp, 'mlp_' + target_name, defense)
    return acc

def ml_attacks_with_default_params(train_x, train_y, test_x, test_y, target_name = 'mnist', defense = ''):
    print('*'*30, target_name, '*'*30)
    rf = RandomForestClassifier(random_state=2021).fit(train_x, train_y)
    acc = accuracy_score(test_y, rf.predict(test_x))
    print(f'{acc : .2%}')
    save_ml_mia_model(rf, 'rf_' + target_name, defense)
    mlp = MLPClassifier(random_state=2021).fit(train_x, train_y)
    acc = accuracy_score(test_y, mlp.predict(test_x))
    print(f'{acc : .2%}')
    save_ml_mia_model(mlp, 'mlp_' + target_name, defense)


def ml_unaware_attack_opupo(target_name = 'mnist', num_classes = 10):
    print(target_name)
    train_X, train_Y, test_X, test_Y, _, _ = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta, use_perturbation=True, model_name=target_name, num_classes=num_classes)
    rf = load_ml_mia_model('rf_'+target_name, defense='')
    pred = rf.predict(test_X)
    acc = accuracy_score(test_Y, pred)
    print('rf : ',acc)
    mlp = load_ml_mia_model('mlp_'+target_name, defense='')
    pred = mlp.predict(test_X)
    acc = accuracy_score(test_Y, pred)
    print('mlp : ',acc)


def ml_opupo_aware(train_x, train_y, test_x, test_y, target_name = 'mnist', defense = ''):
    print('*'*30, target_name, '*'*30)
    rf = RandomForestClassifier(random_state=2021).fit(train_x, train_y)
    acc = accuracy_score(test_y, rf.predict(test_x))
    print(f'{acc : .2%}')
    save_ml_mia_model(rf, 'rf_' + target_name, defense)
    mlp = MLPClassifier(random_state=2021).fit(train_x, train_y)
    acc = accuracy_score(test_y, mlp.predict(test_x))
    print(f'{acc : .2%}')
    save_ml_mia_model(mlp, 'mlp_' + target_name, defense)

def ml_attack(train_x, train_y, test_x, test_y, ml = 'rf'):
    if ml == 'rf':
        attack_model = RandomForestClassifier(random_state=2021)
    elif ml == 'mlp':
        attack_model = MLPClassifier(random_state=2021)
    else:
        print('Invalid machine learning algorithm!')

    attack_model.fit(train_x, train_y)
    pred = attack_model.predict(test_x)
    attack_acc = accuracy_score(test_y, pred)
    print(f'{attack_acc : .2%}')

if __name__=='__main__':
    use_cuda = initialization(2021)
    # Attack undefended models
    # train_x, train_y, test_x, test_y, _, _ = get_predictions('fashion_resnet18')
    # ml_attacks_with_default_params(train_x, train_y, test_x, test_y, 'fashion_resnet18')
    train_x, train_y, test_x, test_y, _, _ = get_predictions('cifar10_resnet18')
    ml_attacks_with_default_params(train_x, train_y, test_x, test_y, 'cifar10_resnet18')
    # train_x, train_y, test_x, test_y, _, _ = get_predictions('cifar100_resnet18', 100)
    # ml_attacks_with_default_params(train_x, train_y, test_x, test_y, 'cifar100_resnet18')

    # Defense-unaware Adversaries
    # ml_unaware_attack_opupo('fashion_resnet18', 10)
    ml_unaware_attack_opupo('cifar10_resnet18', 10)
    # ml_unaware_attack_opupo('cifar100_resnet18', 100)

    # Defense-aware Adversary
    defense = 'opupo_' + '_' + str(kappa) + '_' + str(delta) + '_' + str(eta) + '_' + str(use_perturbation)
    # train_x, train_y, test_x, test_y, _, _ = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta, use_perturbation=use_perturbation, model_name='fashion_resnet18', num_classes=10)
    # ml_opupo_aware(train_x, train_y, test_x, test_y, 'fashion_resnet18', defense)
    train_x, train_y, test_x, test_y, _, _ = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta, use_perturbation=use_perturbation, model_name='cifar10_resnet18', num_classes=10)
    ml_opupo_aware(train_x, train_y, test_x, test_y, 'cifar10_resnet18', defense)
    # train_x, train_y, test_x, test_y, _, _ = get_predictions_opupo(kappa=kappa, delta=delta, eta=eta, use_perturbation=use_perturbation, model_name='cifar100_resnet18', num_classes=100)
    # ml_opupo_aware(train_x, train_y, test_x, test_y, 'cifar100_resnet18', defense)
