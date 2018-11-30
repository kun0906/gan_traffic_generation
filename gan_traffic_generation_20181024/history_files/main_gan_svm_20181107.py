# -*- coding: utf-8 -*-
"""
    use naive_gan to generate more data.
"""
import argparse
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from achieve_train_test_set import achieve_train_test_set, mix_two_files, pca_show, open_file, split_normal_attack_data
from main_rf_20181024 import split_mix_data
from naive_gan import NaiveGAN, TrafficDataset
from utilities.csv_dataloader import mix_normal_attack_and_label, save_data, save_numpy_data
from utilities.common_funcs import load_model, dump_model, normalizate_data

np.set_printoptions(suppress=True)  # Suppresses the use of scientific notation for small numbers in numpy array


def run_gan_main(input_f, name='normal', generated_num=10000, output_dir='log', epochs=10, show_flg=True, **kwargs):
    # step 1 achieve train set
    train_set = TrafficDataset(input_f, transform=None, normalization_flg=False)
    print('\'%s\' train_set size : (%d, %d) used for training \'%s\' naive_gan.' % (
        name, len(train_set.X), len(train_set.X[0]), name))
    # step 2.1 initialize gan
    gan_m = NaiveGAN(num_epochs=epochs, num_features=len(train_set.X[0]), batch_size=64, show_flg=show_flg, output_dir=output_dir, GAN_name=name)

    # step 2.2 train gan model
    print('\nTraining begins ...')
    gan_m.train(train_set)
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


def test_SVM(X_train, y_train,X_test, y_test, name='svm'):
    acc_dict = {'svm': {'train': [], 'test': []}, 'RF': {'train': [], 'test': []}, 'mlp': {'train': [], 'test': []}}
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
    print(name + ' svm train confusion matrix:\n', cm)
    acc = 100.0 * sum(y_train == y_preds) / len(y_train)
    print(' Acc: %.2f%% \n' % (acc))
    acc_dict['svm']['train'].append(acc)

    X_normal=[]  # predict correctly samples : TP
    X_attack=[]  # predict correctly samples : TN
    y_normal=[]
    y_attack=[]
    for i in range(len(y_train)):
        if y_preds[i] == y_train[i]:  # only save predict correctly samples for training GAN
            if y_train[i] == 0:
                X_normal.append(X_train[i])
                y_normal.append(y_train[i])
            if y_train[i] == 1:
                X_attack.append(X_train[i])
                y_attack.append(y_train[i])

    y_preds = svm_m.predict(X_test)
    # y_preds = (rf_m.predict(X_test) == -1) * 1
    # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
    cm = confusion_matrix(y_test, y_preds)
    print(name + ' svm test confusion matrix:\n', cm)
    acc = 100.0 * sum(y_test == y_preds) / len(y_test)
    print(' Acc: %.2f%% \n' % (acc))
    acc_dict['svm']['test'].append(acc)

    return X_normal, y_normal, X_attack, y_attack


def dimension_reduction(X,y, n_components=5):
    # # # # PCA dimension reduction
    pca_m = PCA(n_components=n_components)
    # # normal_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X = np.asarray(X, dtype=float)
    X_reduced = pca_m.fit_transform(X)  # in this function, it will do x-mean(x,axis=0), so there is no need to do x-mean(x, axis=0) before.
    print('pca_m:', pca_m.explained_variance_ratio_)
    print('pca sum:', sum(pca_m.explained_variance_ratio_))
    X = X_reduced
    y = np.asarray(y, dtype=int)
    # _ = save_numpy_data((X, y), output_f=os.path.join(output_dir, 'original_normalized_mix_data.txt'))
    # reduced_normal_f, reduced_attack_f = split_mix_data(save_data(X, y, output_f=output_f + 'reduced-dims.csv'))

    return X, y


def main(normal_f='', attack_f='', epochs=10, label_dict={'normal': '0', 'attack': '1'}, output_dir='./log', select_train_size=0.03, show_flg=False):
    """

    :param normal_f:
    :param attack_f:
    :param output_dir:
    :return:
    """

    # step 1 load data  (attack data size == normal data size)
    # step 1.1. mix normal (label = 0)and attack (label=1) data, then normalize the mixed data [0,1]
    (X, y), output_f = mix_normal_attack_and_label(normal_f, attack_f, label_dict=label_dict,
                                                   start_feat_idx=[0,'-'],
                                                   output_f=os.path.join(output_dir, 'original_mix_data.csv'))
    if show_flg:
        pca_show(X, y, name='pca n_components=2 on all data')
    # X= np.asarray(X, dtype=float)
    X = normalizate_data(np.asarray(X, dtype=float))
    X,y = dimension_reduction(X, y, n_components=5)
    if show_flg:
        pca_show(X, y, name = 'pca n_components=2 on dimension reduction data')
    X = normalizate_data(np.asarray(X, dtype=float))

    y = np.asarray(y, dtype=int)
    _ = save_numpy_data((X, y), output_f=os.path.join(output_dir, 'original_normalized_mix_data.txt'))
    # t_sne_show(X, y)

    # step 1.2. split train (70%) and test (30%) on original mixed data
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X, dtype=float),
                                                        np.asarray(y, dtype=int), test_size=0.3, random_state=1)
    train_set = (X_train, y_train)
    if show_flg:
        pca_show(X_train, y_train, name = 'pca n_components=2, 70% train_set')
    test_set = (X_test, y_test)
    if show_flg:
        pca_show(X_test, y_test, name = 'pca n_components=2, 30% test_set')
    original_train_f = save_numpy_data(train_set,
                                       output_f=os.path.join(output_dir, 'original_normalized_train_set.csv'))
    original_test_f = save_numpy_data(test_set,
                                      output_f=os.path.join(output_dir,
                                                            'original_normalized_test_set.csv'))  # for evalution
    print('test svm on 70% train-set, 30% test-set')
    test_SVM(X_train, y_train, X_test, y_test, name = 'svm on 70% train set')


    # step 4. separate normal and attack data in original train set
    _, X_selected, _, y_selected = train_test_split(X_train,y_train, test_size=select_train_size, random_state=1)
    original_select_train_f=save_numpy_data((X_selected, y_selected), output_f=os.path.join(output_dir, 'original_selected_mix_data.csv'))

    print('test svm on %.2f%% train-set, 30%% test-set'%(select_train_size*100))
    X_predict_normal,y_predict_normal, X_predict_attack, y_predict_attack=\
        test_SVM(X_selected,y_selected, X_test, y_test, name='svm on %.2f%% train set'%(select_train_size*100))
    normal_train_set_f=save_numpy_data((X_predict_normal, y_predict_normal),output_f=os.path.join(output_dir, 'normal_data_for_GAN.csv'))
    attack_train_set_f=save_numpy_data((X_predict_attack, y_predict_attack), output_f= os.path.join(output_dir, 'attack_data_for_GAN.csv'))

    # step 5. build different gan for normal and attack data separately, just use train_size = 0.3.
    num = 10000
    #   5.1 noraml_gan
    new_gen_normal_f,normal_gan_loss_file, normal_gan_decision_file = run_gan_main(input_f=normal_train_set_f, name='normal', generated_num=num, output_dir=output_dir,
                                    epochs=epochs,show_flg=show_flg)
    #   5.2 attack_gan
    new_gen_attack_f, attack_gan_loss_file, attack_gan_decision_file = run_gan_main(input_f=attack_train_set_f, name='attack', generated_num=num, output_dir=output_dir,
                                    epochs=epochs,show_flg=show_flg)
    #   5.3 merge the generated data (normal and attack data)
    (_, _), generate_train_f = mix_normal_attack_and_label(new_gen_normal_f, new_gen_attack_f, label_dict=label_dict,
                                                           start_feat_idx=[0, '-'],
                                                           output_f=os.path.join(output_dir,
                                                                                 'generated_%d_mix_data.csv' % num))
    # step 6. mix original train data and new generated data
    new_train_set_f = mix_two_files(original_select_train_f, generate_train_f,
                                    output_f=os.path.join(output_dir, '0_new_train_set.csv'))

    print('\'new train set\' (includes original train set and generated train_set) is in \'%s\'.' % new_train_set_f)

    print('test svm on new train-set, 30% test-set')
    X_new_train,y_new_train = open_file(new_train_set_f)
    test_SVM(X_new_train, y_new_train, X_test, y_test, name = 'svm on new train set')

    return original_train_f, original_select_train_f, new_train_set_f, original_test_f



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


if __name__ == '__main__':
    # input_files_dict={'normal_files': 'data/normal_demo.txt', 'attack_files': 'data/attack_demo.txt'}
    args = parse_params()
    input_files_dict = eval(args['input_files_dict'])
    epochs = args['epochs']
    out_dir = args['out_dir']
    print('args:%s\n' % args)
    # ocsvm_main(input_files_dict, kernel=kernel, out_dir='../log')
    noraml_f = input_files_dict['normal_files']
    attack_f = input_files_dict['attack_files']
    main(normal_f=noraml_f, attack_f=attack_f, epochs=int(epochs), output_dir='./log',
         label_dict={'normal': '0', 'attack': '1'})
