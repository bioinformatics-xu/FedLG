import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import pandas as pd
import matplotlib.pyplot as plt

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from datasets import kaggle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    #parser.add_argument('--dataset', type=str, default='generated', help='dataset used for training')
    parser.add_argument('--dataset', type=str, default='coronary', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='alg')
    parser.add_argument('--batch-size', type=int, default=35, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=3,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='FedLG',help='fl algorithms: fedavg/fedprox/fednova/FedLG')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/coronary", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.01, help='the mu parameter for fedLG')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    elif args.datadir == './data/kaggle':
                        input_size = 11
                        output_size = 2
                        hidden_sizes = [16, 8]
                    elif args.datadir == './data/heart':
                        input_size = 16
                        output_size = 2
                        hidden_sizes = [16, 8]
                    elif args.datadir == './data/coronary01':
                        input_size = 36
                        output_size = 2
                        hidden_sizes = [16, 8]
                    elif args.datadir == './data/coronary1':
                        input_size = 42
                        output_size = 2
                        hidden_sizes = [16, 8]
                    elif args.datadir == './data/coronary2':
                        input_size = 42
                        output_size = 2
                        hidden_sizes = [16, 8]
                    elif args.datadir == './data/coronary10':
                        input_size = 36
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type

#fedavg全局模型

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                                device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                                device=device)

    # 创建表格
    df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
    print("fedCN Confusion_Matrix：")
    print(df)

    # 计算Specificity
    spe = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Specificity(特异性)：", spe)
    # 计算Sensitivity
    sen = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    print("Sensitivity（敏感性）：", sen)
    # 计算PPV
    ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    print("PPV（精确度）：", ppv)
    # 计算NPV
    npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    print("NPV（阴性预测值）：", npv)

    logger.info('>> Train accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info('>> Test auc: %f' % test_auc)
    logger.info('>> Test f1: %f' % test_f1)


    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # 创建表格
    df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
    print("fedCN Confusion_Matrix：")
    print(df)

    # 计算Specificity
    spe = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Specificity(特异性)：", spe)
    # 计算Sensitivity
    sen = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    print("Sensitivity（敏感性）：", sen)
    # 计算PPV
    ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    print("PPV（精确度）：", ppv)
    # 计算NPV
    npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    print("NPV（阴性预测值）：", npv)

    logger.info('>> Train accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info('>> Test auc: %f' % test_auc)
    logger.info('>> Test f1: %f' % test_f1)


    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())

#for fedprox
       #global_weight_collector = list(global_net.to(device).parameters())
           # fed_nova_reg = 0.0
           # for param_index, param in enumerate(net.parameters()):
              #  fed_nova_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
           # loss += fed_prox_reg
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                                device=device)

    # 创建表格
    df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
    print("fedCN Confusion_Matrix：")
    print(df)

    # 计算Specificity
    spe = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Specificity(特异性)：", spe)
    # 计算Sensitivity
    sen = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    print("Sensitivity（敏感性）：", sen)
    # 计算PPV
    ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    print("PPV（精确度）：", ppv)
    # 计算NPV
    npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    print("NPV（阴性预测值）：", npv)

    logger.info('>> Train accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info('>> Test auc: %f' % test_auc)
    logger.info('>> Test f1: %f' % test_f1)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def train_net_FedLG(net_id, net, global_net , train_dataloader, test_dataloader, epochs, lr, mu , args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))


    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                                device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    tau = 0
#
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                # for fedprox

                fed_nova_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_nova_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                loss += fed_nova_reg

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
    train_acc, train_auc ,train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, test_auc ,test_f1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # 创建表格
    df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
    print("fedCN Confusion_Matrix：")
    print(df)

    # 计算Specificity
    spe = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Specificity(特异性)：", spe)
    # 计算Sensitivity
    sen = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    print("Sensitivity（敏感性）：", sen)
    # 计算PPV
    ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    print("PPV（精确度）：", ppv)
    # 计算NPV
    npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    print("NPV（阴性预测值）：", npv)

    logger.info('>> Train accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info('>> Test auc: %f' % test_auc)
    logger.info('>> Test f1: %f' % test_f1)
    logger.info(' ** Training complete **')

    return train_acc, test_acc, a_i, norm_grad

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)

#feavg本地训练模型
def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list



def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc


    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list

def local_train_net_FedLG(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_FedLG(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.mu , args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc


    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list


#训练数据集
def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    #for handler in logging.root.handlers[:]:
    #    logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    #确定数据集
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))
    #数据集的处理
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)



    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        logger.info("Initializing nets")
        # 创建一个空列表来存储AUC值
        auc_results = []
        acc_results = []
        f1_results = []
        # 创建一个空列表来存储ppv，npv值
        ppv_results = []
        npv_results = []
        spe_results = []
        sen_results = []

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc, auc ,f1 = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_auc ,test_f1 = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            # 创建表格
            df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
            print("fedavg Confusion_Matrix：")
            print(df)

            # 计算Specificity
            spe = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            print("Specificity(特异性)：", spe)
            # 计算Sensitivity
            sen = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
            print("Sensitivity（敏感性）：", sen)
            # 计算PPV
            ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
            print("PPV（精确度）：", ppv)
            # 计算NPV
            npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
            print("NPV（阴性预测值）：", npv)

            # 将pvv值添加到列表中
            ppv_results.append(ppv)
            npv_results.append(npv)
            spe_results.append(spe)
            sen_results.append(sen)

            # 将AUC值添加到列表中
            acc_results.append(test_acc)
            auc_results.append(test_auc)
            f1_results.append(test_f1)

            print(auc_results)
            # 创建一个x轴上的范围，从1到auc值的个数

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test auc: %f' % test_auc)
            logger.info('>> Global Model Test f1: %f' % test_f1)

            # 创建一个DataFrame对象
        df = pd.DataFrame({'acc_results': acc_results, 'auc_results': auc_results, 'f1_results': f1_results})
        # 将DataFrame写入Excel文件
        df.to_excel('fedavg.xlsx', index=False)

        # 创建一个DataFrame对象
        df = pd.DataFrame({' ppv_results': ppv_results, 'npv_results': npv_results, 'spe_results': spe_results,
                           'sen_results': sen_results})
        # 将DataFrame写入Excel文件
        df.to_excel('fedavg_ppv.xlsx', index=False)


    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        # 创建一个空列表来存储AUC值
        auc_results = []
        acc_results = []
        f1_results = []
        # 创建一个空列表来存储ppv，npv值
        ppv_results = []
        npv_results = []
        spe_results = []
        sen_results = []

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))


            global_model.to(device)
            train_acc, auc ,f1 = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_auc ,test_f1 = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            # 创建表格
            df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
            print("fedprox Confusion_Matrix：")
            print(df)

            # 计算Specificity
            spe = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            print("Specificity(特异性)：", spe)
            # 计算Sensitivity
            sen = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
            print("Sensitivity（敏感性）：", sen)
            # 计算PPV
            ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
            print("PPV（精确度）：", ppv)
            # 计算NPV
            npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
            print("NPV（阴性预测值）：", npv)

            # 将pvv值添加到列表中
            ppv_results.append(ppv)
            npv_results.append(npv)
            spe_results.append(spe)
            sen_results.append(sen)

            # 将AUC值添加到列表中
            acc_results.append(test_acc)
            auc_results.append(test_auc)
            f1_results.append(test_f1)

            print(auc_results)
            # 创建一个x轴上的范围，从1到auc值的个数

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test auc: %f' % test_auc)
            logger.info('>> Global Model Test f1: %f' % test_f1)

            # 创建一个DataFrame对象
        df = pd.DataFrame({'acc_results': acc_results, 'auc_results': auc_results, 'f1_results': f1_results})
        # 将DataFrame写入Excel文件
        df.to_excel('fedprox.xlsx', index=False)

        # 创建一个DataFrame对象
        df = pd.DataFrame({' ppv_results': ppv_results, 'npv_results': npv_results, 'spe_results': spe_results,
                           'sen_results': sen_results})
        # 将DataFrame写入Excel文件
        df.to_excel('fedprox_ppv.xlsx', index=False)




    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        # 创建一个空列表来存储AUC值
        auc_results = []
        acc_results = []
        f1_results = []
        # 创建一个空列表来存储ppv，npv值
        ppv_results = []
        npv_results = []
        spe_results = []
        sen_results = []

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            total_n = sum(n_list)
            #print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    #if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    #else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n


            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                #print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc, train_auc, train_f1 = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(global_model, test_dl_global,
                                                                        get_confusion_matrix=True, device=device)
            # 创建表格
            df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
            print("fednova Confusion_Matrix：")
            print(df)

            # 计算Specificity
            spe = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            print("Specificity(特异性)：", spe)
            # 计算Sensitivity
            sen = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
            print("Sensitivity（敏感性）：", sen)
            # 计算PPV
            ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
            print("PPV（精确度）：", ppv)
            # 计算NPV
            npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
            print("NPV（阴性预测值）：", npv)

            # 将pvv值添加到列表中
            ppv_results.append(ppv)
            npv_results.append(npv)
            spe_results.append(spe)
            sen_results.append(sen)

            # 将AUC值添加到列表中
            acc_results.append(test_acc)
            auc_results.append(test_auc)
            f1_results.append(test_f1)

            print(auc_results)
            # 创建一个x轴上的范围，从1到auc值的个数

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test auc: %f' % test_auc)
            logger.info('>> Global Model Test f1: %f' % test_f1)


        x = list(range(1, len(auc_results) + 1))

        # 设置颜色的 RGBA 值
        color1=(0,0,1,0.3)
        color2 = (0, 1, 0, 0.3)
        color3 = (1, 0, 0, 0.3)

        # 绘制折线图并设置颜色透明度
        plt.plot(x, acc_results, label='Accuracy', linestyle='-',color=color1, alpha=0.4,  lw=1)
        plt.plot(x, auc_results, label='AUC',linestyle='-', color=color2, alpha=0.4,  lw=1)
        plt.plot(x, f1_results, label='f1', linestyle='-',color=color3, alpha=0.4,  lw=1)

        # 设置背景网格线样式
        plt.grid(color='gray', linestyle='--', linewidth=0.3)
        # 设置图例
        plt.legend()
        # 设置x轴和y轴标签
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('fednova Model Score')
        # 显示图形
        plt.show()

        # 创建一个DataFrame对象
        df = pd.DataFrame({'acc_results': acc_results, 'auc_results': auc_results, 'f1_results': f1_results})
        # 将DataFrame写入Excel文件
        df.to_excel('fednova.xlsx', index=False)

        # 创建一个DataFrame对象
        df = pd.DataFrame({' ppv_results': ppv_results, 'npv_results': npv_results, 'spe_results': spe_results,
                           'sen_results': sen_results})
        # 将DataFrame写入Excel文件
        df.to_excel('fednova_ppv.xlsx', index=False)


    elif args.alg == 'FedLG':
        logger.info("Initializing nets")
        # 创建一个空列表来存储AUC值
        auc_results = []
        acc_results=[]
        f1_results=[]
        # 创建一个空列表来存储ppv，npv值
        ppv_results = []
        npv_results = []
        spe_results = []
        sen_results = []

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())

        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0

        for key in d_total_round:
            d_total_round[key] = 0
        data_sum = 0

        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []

        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)
        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_FedLG(nets, selected, global_model, args, net_dataidx_map,
                                                                test_dl=test_dl_global, device=device)
            total_n = sum(n_list)
            # print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())

            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    # if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    # else:

                    d_total_round[key] += d_para[key] * n_list[i] / total_n
            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)
            # update global model

            coeff = 0.0

            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i] / total_n
            updated_model = global_model.state_dict()

            for key in updated_model:
                # print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    # print(updated_model[key].type())
                    # print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            global_model.to(device)

            train_acc, train_auc, train_f1 = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_auc, test_f1 = compute_accuracy(global_model, test_dl_global,get_confusion_matrix=True, device=device)


            # 创建表格
            df = pd.DataFrame(conf_matrix, index=['真实负例', '真实正例'], columns=['预测负例', '预测正例'])
            print("Confusion_Matrix：")
            print(df)

            #计算Specificity
            spe = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
            print("Specificity(特异性)：", spe)
            #计算Sensitivity
            sen = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
            print("Sensitivity（敏感性）：", sen)
            # 计算PPV
            ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
            print("PPV（精确度）：", ppv)
            # 计算NPV
            npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
            print("NPV（阴性预测值）：", npv)

            # 将pvv值添加到列表中
            ppv_results.append(ppv)
            npv_results.append(npv)
            spe_results.append(spe)
            sen_results.append(sen)

            # 将AUC值添加到列表中
            acc_results.append(test_acc)
            auc_results.append(test_auc)
            f1_results.append(test_f1)

            print(auc_results)
            # 创建一个x轴上的范围，从1到auc值的个数

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test auc: %f' % test_auc)
            logger.info('>> Global Model Test f1: %f' % test_f1)

        # 创建一个DataFrame对象
        df = pd.DataFrame({'acc_results': acc_results, 'auc_results': auc_results, 'f1_results': f1_results})
        # 将DataFrame写入Excel文件
        df.to_excel('FedLG.xlsx', index=False)

        # 创建一个DataFrame对象
        df = pd.DataFrame({' ppv_results': ppv_results, 'npv_results': npv_results,'spe_results': spe_results,'sen_results': sen_results})
        # 将DataFrame写入Excel文件
        df.to_excel('FedLG_ppv.xlsx', index=False)




    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs

        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)

   
