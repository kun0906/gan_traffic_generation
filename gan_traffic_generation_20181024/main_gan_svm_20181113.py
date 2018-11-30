# -*- coding: utf-8 -*-
"""
    use naive_gan to generate more data.
"""
import argparse
import os
import time
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle

from achieve_train_test_set import mix_two_files, pca_show, open_file
from dcgan import DCGAN
from main_rf_20181024 import split_mix_data
from naive_gan import NaiveGAN, TrafficDataset
from utilities.common_funcs import load_model, dump_model, normalizate_data
from utilities.csv_dataloader import mix_normal_attack_and_label, save_data, save_numpy_data
from utilities.plot import show_figures, show_figures_2

np.set_printoptions(suppress=True)  # Suppresses the use of scientific notation for small numbers in numpy array


def run_gan_main(input_f, name='normal', generated_num=10000, output_dir='log', epochs=10, show_flg=True, gan_type ='naive_gan',
                 time_str = '', **kwargs):
    # step 1 achieve train set
    train = TrafficDataset(input_f, transform=None, normalization_flg=False)
    print('\'%s\' train size : (%d, %d) used for training \'%s\' naive_gan.' % (
        name, len(train.X), len(train.X[0]), name))
    # step 2.1 initialize gan
    if gan_type =='dcgan':
        gan_m = DCGAN(num_epochs=epochs, num_features=len(train.X[0]), batch_size=64, show_flg=show_flg,
                         output_dir=output_dir, GAN_name=name, time_str=time_str)
    else:  # default gan_type
        gan_m = NaiveGAN(num_epochs=epochs, num_features=len(train.X[0]), batch_size=64, show_flg=show_flg,
                     output_dir=output_dir, GAN_name=name, time_str=time_str)
    # step 2.2 train gan model
    print('\nTraining begins ...')
    gan_m.train(train)
    print('Train finished.')

    # step 3.1 dump model
    model_file = dump_model(gan_m, os.path.join(output_dir, 'gan_%s_model.p' % name))

    # step 3.2 load model
    gan_m = load_model(model_file)

    # step 4 generated more data
    print('generated_num is', generated_num)
    gen_data = gan_m.generate_data(generated_num)
    output_f = save_data(np.asarray(gen_data).tolist(),
                         output_f=os.path.join(output_dir, 'gan_%s_model' % name + '_generated_%s_samples.csv' % str(
                             generated_num)))

    return output_f, gan_m.gan_loss_file, gan_m.gan_decision_file


def svm_evalution(train_set, val_set, test_set, name='svm'):
    # def svm_evalution(X_train, y_train,X_test, y_test, name='svm'):
    acc_dict = {'svm': {'train_set': [], 'val_set': [], 'test_set': []}, 'RF': {'train_set': [], 'test_set': []},
                'mlp': {'train_set': [], 'test_set': []}}
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set
    train_size_lst = []
    svm_m = SVC(gamma='auto')
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)

    svm_m.fit(X_train, y_train)
    y_preds = svm_m.predict(X_train)
    # y_preds = (rf_m.predict(X_test) == -1) * 1
    # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
    cm = confusion_matrix(y_train, y_preds)
    print(name + ', confusion matrix on train_set:\n', cm)
    acc = 100.0 * sum(y_train == y_preds) / len(y_train)
    print(' Acc: %.2f%% \n' % (acc))
    acc_dict['svm']['train_set'].append(acc)

    X_normal = []  # predict correctly samples : TP
    X_attack = []  # predict correctly samples : TN
    y_normal = []
    y_attack = []
    for i in range(len(y_train)):
        if y_preds[i] == y_train[i]:  # only save predict correctly samples for training GAN
            if y_train[i] == 0:
                X_normal.append(X_train[i])
                y_normal.append(y_train[i])
            if y_train[i] == 1:
                X_attack.append(X_train[i])
                y_attack.append(y_train[i])

    y_preds = svm_m.predict(X_val)
    # y_preds = (rf_m.predict(X_test) == -1) * 1
    # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
    cm = confusion_matrix(y_val, y_preds)
    print(name + ', confusion matrix on val_set:\n', cm)
    acc = 100.0 * sum(y_val == y_preds) / len(y_val)
    print(' Acc: %.2f%% \n' % (acc))
    acc_dict['svm']['val_set'].append(acc)

    y_preds = svm_m.predict(X_test)
    # y_preds = (rf_m.predict(X_test) == -1) * 1
    # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
    cm = confusion_matrix(y_test, y_preds)
    print(name + ', confusion matrix on test_set:\n', cm)
    acc = 100.0 * sum(y_test == y_preds) / len(y_test)
    print(' Acc: %.2f%% \n' % (acc))
    acc_dict['svm']['test_set'].append(acc)

    return (X_normal, y_normal), (X_attack, y_attack)


def dimension_reduction(X, y, n_components=5):
    # # # # PCA dimension reduction
    pca_m = PCA(n_components=n_components)
    # # normal_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X = np.asarray(X, dtype=float)
    X_reduced = pca_m.fit_transform(
        X)  # in this function, it will do x-mean(x,axis=0), so there is no need to do x-mean(x, axis=0) before.
    print('pca_m:', pca_m.explained_variance_ratio_)
    print('pca sum:', sum(pca_m.explained_variance_ratio_))
    X = X_reduced
    y = np.asarray(y, dtype=int)
    # _ = save_numpy_data((X, y), output_f=os.path.join(output_dir, 'original_norm_mix_data.txt'))
    # reduced_normal_f, reduced_attack_f = split_mix_data(save_data(X, y, output_f=output_f + 'reduced-dims.csv'))

    return X, y


def mix_data(normal_file, attack_file, label_dict, output_dir):
    def open_file(input_file):
        X = []
        with open(input_file, 'r') as in_hdl:
            for line in in_hdl:
                line_arr = line.strip().split(',')
                X.append(line_arr)

        return X

    X_normal = open_file(normal_file)
    y_normal = [label_dict['normal']] * len(X_normal)

    X_attack = open_file(attack_file)
    y_attack = [label_dict['attack']] * len(X_attack)

    min_val = min(len(y_normal), len(y_attack))

    X = []
    y = []
    X_normal = X_normal[:min_val]
    y_normal = y_normal[:min_val]
    X_attack = X_attack[:min_val]
    y_attack = y_attack[:min_val]

    X.extend(X_normal)
    y.extend(y_normal)
    X.extend(X_attack)
    y.extend(y_attack)

    return X, y


def split_train_val_test_data(mix_data, train_val_test_ratio=[0.7, 0.1, 0.2]):
    X, y = mix_data
    # step 1.2. split train (70%) and test (30%) on original mixed data
    orig_X_train, X_test, orig_y_train, y_test = train_test_split(np.asarray(X, dtype=float),
                                                                  np.asarray(y, dtype=int),
                                                                  test_size=train_val_test_ratio[-1], random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(orig_X_train, orig_y_train,
                                                      test_size=train_val_test_ratio[-2],
                                                      random_state=1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_mix_data(data, label_dict={'normal': '0', 'attack': '1'}):
    X_normal = []
    y_normal = []
    X_attack = []
    y_attack = []
    for idx, (X, y) in enumerate(zip(data[0], data[1])):
        if str(y) == label_dict['normal']:
            X_normal.append(X)
            y_normal.append(y)
        elif str(y) == label_dict['attack']:
            X_attack.append(X)
            y_attack.append(y)
        else:
            print(f'others={idx}')

    return X_normal, y_normal, X_attack, y_attack


def main(normal_f='', attack_f='', gan_type= 'dcgan', epochs=10, label_dict={'normal': '0', 'attack': '1'}, output_dir='./log',
         select_train_size=0.01, show_flg=False, random_state=1, tp_tn_train_flg=True, time_str =time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) ):
    """

    :param normal_f:
    :param attack_f:
    :param output_dir:
    :param gan_type= 'naive_gan' or 'dcgan'
    :param tp_tn_train_flg: train normal and attack gan only the svm predict correctly values (tp and tn) from case2 train set.
    :return:
    """

    # step 1 achieve raw mix normal and attack data, and make (attack data size == normal data size).
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    orig_mix_data = mix_data(normal_f, attack_f, label_dict, output_dir)
    X, y = orig_mix_data
    print(f"original data shape: {np.asarray(X, dtype=float).shape, np.asarray(y,dtype=int).shape} and y_label:{Counter(y)}")
    if show_flg:
        pca_show(X, y, name='pca n_components=2 on all data')

    # step 1.2 features reduction
    orig_mix_data = dimension_reduction(X, y, n_components=3)
    print()

    # step 1.3 split train, val and test
    print('case1...')
    # 70% train, 10% val and 20% test
    train_70, val_10, test_20 = split_train_val_test_data(orig_mix_data, train_val_test_ratio=[0.7, 0.1, 0.2])
    case1_data = {'train_set': train_70, 'val_set': val_10, 'test_set': test_20}
    norm_X_train_70, min_val, max_val, range_val = normalizate_data(case1_data['train_set'][0])
    norm_X_val_10 = (case1_data['val_set'][0] - min_val) / range_val
    print(f'after normalization, val_set range is {np.max(norm_X_val_10, axis=0)- np.min(norm_X_val_10, axis=0)}')
    norm_X_test_20 = (case1_data['test_set'][0] - min_val) / range_val
    print(f'after normalization, test_set range is {np.max(norm_X_test_20, axis=0)- np.min(norm_X_test_20, axis=0)}')
    case1_norm_data = {'train_set': (norm_X_train_70, train_70[1]),
                       'val_set': (norm_X_val_10, case1_data['val_set'][1]),
                       'test_set': (norm_X_test_20, case1_data['test_set'][1])}
    print(f"case1 train_set:{case1_norm_data['train_set'][0].shape, case1_norm_data['train_set'][1].shape}, "
          f"val_set:{case1_norm_data['val_set'][0].shape, case1_norm_data['val_set'][1].shape}, "
          f"test_set:{case1_norm_data['test_set'][0].shape, case1_norm_data['test_set'][1].shape}")
    print(f"case1 y_train:{Counter(case1_norm_data['train_set'][1])}, "
          f"y_val:{Counter(case1_norm_data['val_set'][1])}, "
          f"y_test:{Counter(case1_norm_data['test_set'][1])}\n")

    case1_train_f = save_numpy_data(case1_norm_data['train_set'],
                                    output_f=os.path.join(output_dir, 'case1_train_file.txt'))
    case1_val_f = save_numpy_data(case1_norm_data['val_set'],
                                  output_f=os.path.join(output_dir, 'case1_val_file.txt'))
    case1_test_f = save_numpy_data(case1_norm_data['test_set'],
                                   output_f=os.path.join(output_dir, 'case1_test_file.txt'))

    print('case2...')
    ## 70%*0.3 train, 10% val and 20% test
    X_train, y_train = train_70
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=select_train_size, random_state=random_state)
    case2_data = {'train_set': (X_train, y_train), 'val_set': val_10, 'test_set': test_20}
    norm_X_train_70_03, min_val, max_val, range_val = normalizate_data(
        np.asarray(case2_data['train_set'][0], dtype=float))  # min, max, range maybe change on different train.
    norm_X_val_10 = (case2_data['val_set'][0] - min_val) / range_val
    print(f'after normalization, val_set range is {np.max(norm_X_val_10, axis=0)- np.min(norm_X_val_10, axis=0)}')
    norm_X_test_20 = (case2_data['test_set'][0] - min_val) / range_val
    print(f'after normalization, test_set range is {np.max(norm_X_test_20, axis=0)- np.min(norm_X_test_20, axis=0)}')
    case2_norm_data = {'train_set': (norm_X_train_70_03, y_train),
                       'val_set': (norm_X_val_10, case2_data['val_set'][1]),
                       'test_set': (norm_X_test_20, case2_data['test_set'][1])}
    print(f"case2 train_set:{case2_norm_data['train_set'][0].shape, case2_norm_data['train_set'][1].shape}, "
          f"val_set:{case2_norm_data['val_set'][0].shape, case2_norm_data['val_set'][1].shape}, "
          f"test_set:{case2_norm_data['test_set'][0].shape, case2_norm_data['test_set'][1].shape}")
    print(f"case2 y_train:{Counter(case2_norm_data['train_set'][1])}, "
          f"y_val:{Counter(case2_norm_data['val_set'][1])}, "
          f"y_test:{Counter(case2_norm_data['test_set'][1])}\n")

    case2_train_f = save_numpy_data(case2_norm_data['train_set'],
                                    output_f=os.path.join(output_dir, 'case2_train_file.txt'))
    # case2_val and case2_test are the same with case1 and case3
    save_numpy_data(case2_norm_data['val_set'],
                    output_f=os.path.join(output_dir,
                                          'case2_val_file.txt'))  # case2_val and case2_test are the same with case1 and case3, however, the range_val will be different.
    save_numpy_data(case2_norm_data['test_set'],
                    output_f=os.path.join(output_dir, 'case2_test_file.txt'))
    X_normal, y_normal, X_attack, y_attack = split_mix_data(case2_norm_data['train_set'], label_dict)
    save_numpy_data((X_normal, y_normal),
                    output_f=os.path.join(output_dir, 'case2_normal_train_file.txt'))
    save_numpy_data((X_attack, y_attack),
                    output_f=os.path.join(output_dir, 'case2_attack_train_file.txt'))

    # step2 train and evaluate on traditional machine learning
    print('case1 train svm on 70% train-set, 30% test-set')
    # svm_evalution(case1_norm_data['train_set'], case1_norm_data['val_set'], case1_norm_data['test_set'],
    #               name='svm on 70% train set')
    print(f'case2 train svm on 70%*{select_train_size*100}% train-set, 30% test-set')
    tp_normal, tn_attack = svm_evalution(case2_norm_data['train_set'], case2_norm_data['val_set'],
                                         case2_norm_data['test_set'],
                                         name=f'svm on 70%*{select_train_size*100}% train set')
    if tp_tn_train_flg:
        print('normal and attack gan will be trained on different train set size according to the confusion matrix of svm.')
        X_normal, y_normal = tp_normal  # for training GAN, only save svm predict correctly normal values (True Positive)
        X_attack, y_attack = tn_attack  # for training GAN, only save svm predict correctly attack values (True Negative)
    else:
        print('normal and attack gan will be trained on equal train set size.')
    tp_normal_train_f = save_numpy_data((X_normal, y_normal),
                                        output_f=os.path.join(output_dir, 'case2_tp_normal_train_file.txt'))
    tn_attack_train_f = save_numpy_data((X_attack, y_attack),
                                        output_f=os.path.join(output_dir, 'case2_tn_attack_train_file.txt'))

    # step 3. build different gan for normal and attack data separately, just use train_size = 0.3.
    num = 10000
    print(f'\nnormal_gan on train set:{np.asarray(X_normal, dtype=float).shape, np.asarray(y_normal,dtype=int).shape}')
    #   3.1 noraml_gan
    new_gen_normal_f, normal_gan_loss_file, normal_gan_decision_file = run_gan_main(input_f=tp_normal_train_f,
                                                                                    name='normal', generated_num=num,
                                                                                    output_dir=output_dir,
                                                                                    epochs=epochs, show_flg=show_flg,
                                                                                    gan_type=gan_type, time_str=time_str)
    print(f'\nattack_gan on train set:{np.asarray(X_attack, dtype=float).shape, np.asarray(y_attack,dtype=int).shape}')
    new_gen_attack_f, attack_gan_loss_file, attack_gan_decision_file = run_gan_main(input_f=tn_attack_train_f,
                                                                                    name='attack', generated_num=num,
                                                                                    output_dir=output_dir,
                                                                                    epochs=epochs, show_flg=show_flg,
                                                                                    gan_type=gan_type, time_str=time_str)
    #   3.3 merge the generated data (normal and attack data)
    (_, _), generate_train_f = mix_normal_attack_and_label(new_gen_normal_f, new_gen_attack_f, label_dict=label_dict,
                                                           start_feat_idx=[0, '-'],
                                                           output_f=os.path.join(output_dir,
                                                                                 'case3_generated_%d_mix_data.csv' % num))
    # step 4. mix original train data and new generated data
    new_train_f = mix_two_files(case2_train_f, generate_train_f,
                                output_f=os.path.join(output_dir, 'case3_new_train_set.csv'))
    print('\'new train set\' (includes original train set and generated train) is in \'%s\'\n.' % new_train_f)

    # step 5. train and test on ML on the new data
    print('case3...')
    X_new_train, y_new_train = open_file(
        new_train_f)  # GAN use sigmoid, so there does not need to do normalization again.
    # case3_data = {'train_set':(X_new_train, y_new_train),
    #                    'val_set': (val_10[0], val_10[1]),
    #                    'test_set': (test_20[0], test_20[1])}
    # # case3 use the same range_val as case 2 on train set because both of them have the same original train set(70%*0.3)
    # norm_X_train_new, min_val, max_val, range_val = normalizate_data(case3_data['train_set'][0])
    # norm_X_val_10 = (case3_data['val_set'][0] - min_val) / range_val
    # print(f'after normalization, val_set range is {np.max(norm_X_val_10, axis=0)- np.min(norm_X_val_10, axis=0)}')
    # norm_X_test_20 = (case3_data['test_set'][0] - min_val) / range_val
    # print(f'after normalization, test_set range is {np.max(norm_X_test_20, axis=0)- np.min(norm_X_test_20, axis=0)}')
    # case3_norm_data = {'train_set': (norm_X_train_new, y_new_train),
    #               'val_set': (norm_X_val_10, case3_data['val_set'][1]),
    #               'test_set': (norm_X_test_20, case3_data['test_set'][1])}
    # np.random.shuffle(zip(X_new_train, y_new_train))  # cannot successful
    X_new_train, y_new_train = shuffle(X_new_train, y_new_train, random_state=random_state)

    case3_norm_data = {'train_set': (X_new_train, y_new_train),
                       'val_set': case2_norm_data['val_set'],
                       'test_set': case2_norm_data['test_set']}
    print(f"case3 train_set:{case3_norm_data['train_set'][0].shape, case3_norm_data['train_set'][1].shape}, "
          f"val_set:{case3_norm_data['val_set'][0].shape, case3_norm_data['val_set'][1].shape}, "
          f"test_set:{case3_norm_data['test_set'][0].shape, case3_norm_data['test_set'][1].shape}")
    print(f"case3 y_train:{Counter(case3_norm_data['train_set'][1])}, "
          f"y_val:{Counter(case3_norm_data['val_set'][1])}, "
          f"y_test:{Counter(case3_norm_data['test_set'][1])}\n")
    print('case3 train svm on new train and evaluate on 30% test')
    svm_evalution(case3_norm_data['train_set'], case3_norm_data['val_set'], case3_norm_data['test_set'],
                  name='svm on new train set')

    return case1_train_f, case2_train_f, new_train_f, case1_val_f, case1_test_f


def parse_params():
    parser = argparse.ArgumentParser(prog='GAN')
    parser.add_argument('-i', '--input_files_dict', type=str, dest='input_files_dict',
                        help='{\'normal_files\': [normal_file,...], \'attack_files\': [attack_file_1, attack_file_2,...]}',
                        default='../Data/normal_demo.txt', required=True)  # '-i' short name, '--input_dir' full name
    parser.add_argument('-e', '--epochs', dest='epochs', help="epochs", default='100')
    parser.add_argument('-o', '--output_dir', dest='out_dir', help="the output information of this scripts",
                        default='../log')
    args = vars(parser.parse_args())

    return args


def plot_data(gan_loss_file, gan_decision_file, name='gan'):
    X, _ = open_file(gan_loss_file, has_y_flg=False)
    show_figures(X[:, 0], X[:, 1], name)
    X, _ = open_file(input_f=gan_decision_file, has_y_flg=False)
    show_figures_2(X, name)


if __name__ == '__main__':
    show_flg = 0
    if show_flg:
        print('display the existing results.')
        for name in ['normal', 'attack']:
            gan_loss_file = 'log/%s_D_loss+G_loss.txt' % name
            gan_decision_file = 'log/%s_D_decision.txt' % name
            plot_data(gan_loss_file, gan_decision_file, name=name)
        print('Warning: please backup these results files firstly when you restart this application.')
        exit()  # just show the results, if you want restart, please save these result files as backup firstly. if not, these files will be overwrite.

    # input_files_dict={'normal_files': 'data/normal_demo.txt', 'attack_files': 'data/attack_demo.txt'}
    args = parse_params()
    input_files_dict = eval(args['input_files_dict'])
    epochs = args['epochs']
    out_dir = args['out_dir']
    print('args:%s\n' % args)
    # ocsvm_main(input_files_dict, kernel=kernel, out_dir='../log')
    noraml_f = input_files_dict['normal_files']
    attack_f = input_files_dict['attack_files']

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'*** time_str:{time_str}')
    main(normal_f=noraml_f, attack_f=attack_f, epochs=int(epochs), output_dir='./log',
         label_dict={'normal': '0', 'attack': '1'}, show_flg=show_flg, time_str=time_str)
