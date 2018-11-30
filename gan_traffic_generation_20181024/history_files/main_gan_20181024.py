# -*- coding: utf-8 -*-
"""
    use naive_GAN to generate more data.
"""
import argparse
import os

import numpy as np

from achieve_train_test_set import achieve_train_test_set, mix_two_files
from naive_gan import TrafficDataset, NaiveGAN
from utilities.csv_dataloader import mix_normal_attack_and_label, save_data
from utilities.common_funcs import load_model, dump_model

np.set_printoptions(suppress=True)  # Suppresses the use of scientific notation for small numbers in numpy array


def run_gan_main(input_f, name='normal', generated_num=10000, output_dir='log', epochs=10, show_flg=True, **kwargs):
    # step 1 achieve train set
    train_set = TrafficDataset(input_f, transform=None, normalization_flg=False)
    print('\'%s\' train_set size : (%d, %d) used for training \'%s\' naive_GAN.' % (
        name, len(train_set.X), len(train_set.X[0]), name))
    # step 2.1 initialize gan
    gan_m = NaiveGAN(num_epochs=epochs, num_features=len(train_set.X[0]), batch_size=64, show_flg=True)

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

    return output_f


def main(normal_f='', attack_f='', epochs=10, label_dict={'normal': '0', 'attack': '1'}, output_dir='./log'):
    """

    :param normal_f:
    :param attack_f:
    :param output_dir:
    :return:
    """

    # step 1 get data
    original_train_f, original_test_f, original_select_train_f, (
    normal_train_set_f, attack_train_set_f) = achieve_train_test_set(
        normal_f=input_files_dict['normal_files'],
        attack_f=input_files_dict['attack_files'],
        label_dict={'normal': '0', 'attack': '1'},
        select_train_size=0.01, output_dir='./log', start_feat_idx=[0,'-'])

    # step 5. build different gan for normal and attack data separately, just use train_size = 0.3.
    num = 10000
    #   5.1 noraml_gan
    new_gen_normal_f = run_gan_main(input_f=normal_train_set_f, name='normal', generated_num=num, output_dir=output_dir,
                                    epochs=epochs)
    #   5.2 attack_gan
    new_gen_attack_f = run_gan_main(input_f=attack_train_set_f, name='attack', generated_num=num, output_dir=output_dir,
                                    epochs=epochs)
    #   5.3 merge the generated data (normal and attack data)
    (_, _), generate_train_f = mix_normal_attack_and_label(new_gen_normal_f, new_gen_attack_f, label_dict=label_dict,
                                                           start_feat_idx=[0, '-'],
                                                           output_f=os.path.join(output_dir,
                                                                                 'generated_%d_mix_data.csv' % num))
    # step 6. mix original train data and new generated data
    new_train_set_f = mix_two_files(original_select_train_f, generate_train_f,
                                    output_f=os.path.join(output_dir, '0_new_train_set.csv'))

    print('\'new train set\' (includes original train set and generated train_set) is in \'%s\'.' % new_train_set_f)


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
