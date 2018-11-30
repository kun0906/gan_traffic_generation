import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from utilities.common_funcs import normalizate_data
from utilities.csv_dataloader import save_numpy_data, mix_normal_attack_and_label, save_data


def split_normal_attack_data(input_f='', select_train_size=0.3, label_dict={'normal': '0', 'attack': '1'},
                             output_dir='./log'):
    data_dict = {'normal_data': [], 'attack_data': []}
    with open(input_f, 'r') as in_f:
        line = in_f.readline()
        while line:
            line_arr = line.strip().split(',')
            if line_arr[-1] == label_dict['normal']:  # or line_arr[-1] == '0.0' or line_arr[-1] == '0':
                data_dict['normal_data'].append(line_arr)
            elif line_arr[-1] == label_dict['attack']:  # or line_arr[-1] == '1.0' or line_arr[-1] == '1':
                data_dict['attack_data'].append(line_arr)
            else:
                pass
            line = in_f.readline()

    print('normal_train_set size : (%d, %d)' % (
        len(data_dict['normal_data']), len(data_dict['normal_data'][0]) - 1))  # include label
    print('attack_train_set size : (%d, %d)' % (len(data_dict['attack_data']), len(data_dict['attack_data'][0]) - 1))

    print('select size = ', select_train_size)
    save_size = min(int(len(data_dict['normal_data']) * select_train_size),
                    int(len(data_dict['attack_data']) * select_train_size))
    data_dict['normal_data'] = data_dict['normal_data'][:save_size]
    data_dict['attack_data'] = data_dict['attack_data'][:save_size]

    select_train_set = []
    select_train_set.extend(data_dict['normal_data'])
    select_train_set.extend(data_dict['attack_data'])
    original_select_train_f = save_data(select_train_set, os.path.join(output_dir, 'original_select_train_set.csv'))
    normal_input_f = save_data(data_dict['normal_data'],
                               os.path.join(output_dir, 'original_normal_normalized_train_set.csv'))
    attack_input_f = save_data(data_dict['attack_data'],
                               os.path.join(output_dir, 'original_attack_normalized_train_set.csv'))

    print('after selected, normal_train_set size : (%d, %d)' % (
        len(data_dict['normal_data']), len(data_dict['normal_data'][0]) - 1))  # include label
    print('after selected, attack_train_set size : (%d, %d)\n' % (
        len(data_dict['attack_data']), len(data_dict['attack_data'][0]) - 1))

    return original_select_train_f, normal_input_f, attack_input_f, (data_dict['normal_data'], data_dict['attack_data'])


def mix_two_files(original_train_f, generate_train_f, output_f):
    with open(output_f, 'w') as out_f:
        with open(original_train_f, 'r') as in_f:
            line = in_f.readline()
            while line:
                out_f.write(line)
                line = in_f.readline()

        with open(generate_train_f, 'r') as in_f:
            line = in_f.readline()
            while line:
                out_f.write(line)
                line = in_f.readline()

    return output_f


def open_file(input_f, has_y_flg=True):
    X = []
    y = []
    with open(input_f, 'r') as in_h:
        for line in in_h:
            line_arr_tmp = line.strip().split(',')
            if has_y_flg:
                X.append(line_arr_tmp[:-1])
                y.append(line_arr_tmp[-1])
            else:
                X.append(line_arr_tmp)
                y.append('0')

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    return X, y


def pca_show(X, y, name='pca'):
    pca_results = PCA(n_components=2).fit_transform(X)

    plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5, c=y, marker='x')
    plt.xlabel('x-pca')
    plt.ylabel('y-pca')
    plt.title(name)
    plt.show()


def t_sne_show(X, y):
    """
        T-Distributed Stochastic Neighbouring Entities (t-SNE
    :param input_f:
    :return:
    """

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, c=y, marker='x')
    plt.xlabel('x-tsne')
    plt.ylabel('y-tsne')
    plt.show()
    # df_tsne['x-tsne'] = tsne_results[:, 0]
    # df_tsne['y-tsne'] = tsne_results[:, 1]
    #
    # chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
    #         + geom_point(size=70, alpha=0.1) \
    #         + ggtitle("tSNE dimensions colored by digit")
    # chart
    #


def achieve_train_test_set(normal_f, attack_f, label_dict={'normal': '0', 'attack': '1'}, select_train_size=0.3,
                           output_dir='./log', start_feat_idx=['-', '-']):
    # step 1. mix normal (label = 0)and attack (label=1) data, then normalize the mixed data [0,1]
    (X, y), output_f = mix_normal_attack_and_label(normal_f, attack_f, label_dict=label_dict,
                                                   start_feat_idx=start_feat_idx,
                                                   output_f=os.path.join(output_dir, 'original_mix_data.csv'))
    # X= np.asarray(X, dtype=float)
    X = normalizate_data(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=int)
    _ = save_numpy_data((X, y), output_f=os.path.join(output_dir, 'original_normalized_mix_data.txt'))
    pca_show(X, y)
    # t_sne_show(X, y)
    # step 2. split train (70%) and test (30%) on original mixed data
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X, dtype=float),
                                                        np.asarray(y, dtype=int), test_size=0.3, random_state=1)

    # step 3. save train and test to file.
    train_set = (X_train, y_train)
    test_set = (X_test, y_test)
    original_train_f = save_numpy_data(train_set,
                                       output_f=os.path.join(output_dir, 'original_normalized_train_set.csv'))
    original_test_f = save_numpy_data(test_set,
                                      output_f=os.path.join(output_dir,
                                                            'original_normalized_test_set.csv'))  # for evalution
    X, y = open_file(original_train_f)
    pca_show(X, y)
    X, y = open_file(original_test_f)
    pca_show(X, y)
    # t_sne_show(original_train_f)
    # step 4. separate normal and attack data in original train set
    original_select_train_f, normal_train_set_f, attack_train_set_f, (
        normal_data, attack_data) = split_normal_attack_data(
        input_f=original_train_f, select_train_size=select_train_size)

    return original_train_f, original_test_f, original_select_train_f, (normal_train_set_f, attack_train_set_f)


if __name__ == '__main__':
    normal_f = ''
    attack_f = ''
    achieve_train_test_set()
