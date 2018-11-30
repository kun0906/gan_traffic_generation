from collections import Counter

from sklearn.metrics import accuracy_score, confusion_matrix

from model import SGAN, RSGAN, RaSGAN, RaLSGAN
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
import torchvision_sunner.transforms as sunnertransforms
import numpy as np
import argparse
import torch
import math
import cv2
import os

# Hyper-parameters
from utilties.TrafficDataset import TrafficDataset, split_train_test
from utilties.data_preprocess import normalize_data

img_size = 64


def parse():
    """
        Parse the argument

        --type      :   str object, the type of the GAN you want to use
                        This program only accept SGAN, RGAN, RaGAN and RaLSGAN currently
        --epoch     :   int object
        --det       :   The folder you want to store result in
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='rasgan')
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--det', type=str, default='./det')
    args = parser.parse_args()
    model_type = args.type.lower()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'sgan':
        net = SGAN(device=device).to(device)
    elif model_type == 'rsgan':
        net = RSGAN(device=device).to(device)
    elif model_type == 'rasgan':
        net = RaSGAN(device=device).to(device)
    elif model_type == 'ralsgan':
        net = RaLSGAN(device=device).to(device)
    else:
        raise Exception("Sorry, the program doesn't support " + model_type + " GAN...")
    return args, net


def visialize(fake_img, path=None):
    """
        Visualize the render result

        Arg:    fake_img    - The generated image
                path        - The path you want to store into, default is None
    """
    img = make_grid(fake_img)
    img = sunnertransforms.asImg(img.unsqueeze(0))
    if path is None:
        cv2.imshow('show', img[0])
        cv2.waitKey(10)
    else:
        cv2.imwrite(path, img[0])


def get_loader_iterators_contents(train_loader):
    X = []
    y = []
    for step, (b_x, b_y) in enumerate(train_loader):
        X.extend(b_x.data.tolist())
        y.extend(b_y.data.tolist())

    return X, y


def run_main(input_file, num_features):
    # # input_file = '../data/data_split_train_v2_711/train_%dpkt_images_merged.csv' % i
    # input_file = '../data/attack_normal_data/benign_data.csv'
    print(input_file)
    dataset = TrafficDataset(input_file, transform=None, normalization_flg=True)

    train_sampler, test_sampler = split_train_test(dataset, split_percent=0.7, shuffle=True)
    cntr = Counter(dataset.y)
    print('dataset: ', len(dataset), ' y:', sorted(cntr.items()))
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)  # use all dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=4, sampler=train_sampler)
    X, y = get_loader_iterators_contents(train_loader)
    cntr = Counter(y)
    print('train_loader: ', len(train_loader.sampler), ' y:', sorted(cntr.items()))
    global test_loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                              sampler=test_sampler)
    X, y = get_loader_iterators_contents(test_loader)
    cntr = Counter(y)
    print('test_loader: ', len(test_loader.sampler), ' y:', sorted(cntr.items()))

    # model = WGAN(num_classes, num_features, batch_size=batch_size).to(device)
    # model.run_train(train_loader)
    # show_results(model.results, i)

    # Train
    bar = tqdm(range(args.epoch))
    for i in bar:
        for j, (img, label) in enumerate(train_loader):
            img = img.view(img.size()[0], 1, img.size()[1], 1).float()
            net.optimize(img)
            info = net.getInfo()
            if j % 50 == 0:
                bar.set_description('loss_D: ' + str(info['loss_D']) + '    loss_G: ' + str(info['loss_G']))
                loss_D_list.append(math.log(info['loss_D']))
                loss_G_list.append(math.log(info['loss_G']))
                bar.refresh()
                gen_img = net.fake_img
        #         visialize(gen_img)
        # visialize(gen_img, os.path.join(args.det, str(i) + '.png'))

    # Plot the loss curve
    plt.plot(range(len(loss_D_list)), loss_D_list, '-', label='D loss')
    plt.plot(range(len(loss_G_list)), loss_G_list, '-', label='G loss')
    plt.legend()
    plt.title('The loss curve (log scale)')
    plt.savefig(os.path.join(args.det, 'loss.png'))

    # model.run_test(test_loader)

    # Save the model checkpoint
    # torch.save(model.state_dict(), 'wgan_model_%d.ckpt' % i)

    model = net
    return model, test_loader


def two_stages_online_evaluation(benign_model, attack_model, benign_test_loader, attack_test_loader):
    """

    :param benign_model: 0
    :param attack_model: 1
    :param input_file: mix 'benign_data' and 'attack_data'
    :return:
    """
    data = []
    y_s = []
    for i, (img, label) in enumerate(benign_test_loader):
        data.extend(img)
        y_s.extend(label.data.tolist())
    for i, (img, label) in enumerate(attack_test_loader):
        data.extend(img)
        y_s.extend(label.data.tolist())

    unknow_cnt = 0
    y_preds = []

    for i in range(len(data)):
        x = data[i].view([1, 1, 41, 1]).float()  # (nSamples, nChannels, x_Height, x_Width)
        y = y_s[i]
        threshold1 = benign_model.net_d(x)
        if (threshold1 > 0.1) and (threshold1 < 0.9):
            # if threshold1 < 0.3 :
            y_pred = '1'
            print('benign_data', y, y_pred, threshold1.data.tolist())  # attack =0, benign = 1
        else:
            threshold2 = attack_model.net_d(x)
            if (threshold2 > 0.1) and (threshold2 < 0.9):
                # if threshold2 < 0.3:
                y_pred = '0'
                print('attack_data', y, y_pred, threshold2.data.tolist())
            else:
                y_pred = '-10'
                print('unknow_data', y, y_pred, threshold1.data.tolist(), threshold2.data.tolist())
                unknow_cnt += 1
        y_preds.append(y_pred)

    cm = confusion_matrix(y_s, y_preds, labels=['0', '1', 'unknow'])
    sk_accuracy = accuracy_score(y_s, y_preds) * len(data)
    cntr = Counter(y_s)
    print('y_s =', sorted(cntr.items()))
    cntr = Counter(y_preds)
    print('y_preds =', sorted(cntr.items()))
    print(cm, ', acc =', sk_accuracy / len(data))
    print('unknow_cnt', unknow_cnt)


if __name__ == '__main__':
    # Parse parameter and create model
    args, net = parse()  # get default parameters
    loss_D_list = []
    loss_G_list = []

    torch.manual_seed(1)
    # Hyper parameters
    global num_epochs, num_classes, batch_size
    batch_size = args.batch_size

    # # Create folder, loader
    if not os.path.exists(args.det):
        os.mkdir(args.det)
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('image/data', train = True, download = True, transform=transforms.Compose([
    #                 transforms.Scale(img_size),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #         ])),
    #     batch_size = args.batch_size, shuffle = True
    # )

    cross_validation_flg = False
    input_file = '../../data/attack_normal_data/benign_data.csv'
    benign_model, benign_test_loader = run_main(input_file, num_features=41)

    input_file = '../../data/attack_normal_data/attack_data.csv'
    attack_model, attack_test_loader = run_main(input_file, num_features=41)

    two_stages_online_evaluation(benign_model, attack_model, benign_test_loader, attack_test_loader)
