# -*- coding: utf-8 -*-
"""
    use SVM and RF to train and evaluate.
"""
import os
import random

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from achieve_train_test_set import achieve_train_test_set, pca_show
from utilities.csv_dataloader import mix_normal_attack_and_label
from utilities.common_funcs import normalizate_data, normalizate_data_with_u_std

np.set_printoptions(suppress=True)  # Suppresses the use of scientific notation for small numbers in numpy array


def load_data(input_f='', start_feat_idx=0, shuffle_flg=False):
    X = []
    y = []
    with open(input_f, 'r') as in_f:
        line = in_f.readline()
        while line:
            line_arr = line.strip('\n').split(',')
            X.append(line_arr[start_feat_idx:-1])
            y.append(line_arr[-1])

            line = in_f.readline()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    return X, y


def train_svm(train_f):
    X, y = load_data(train_f, start_feat_idx=0, shuffle_flg=True)

    # step 2.1 initialize svm
    svm_m = SVC(gamma='auto')
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)
    svm_m.fit(X, y)

    return svm_m


def train_rf(train_f):
    X, y = load_data(train_f, start_feat_idx=0, shuffle_flg=True)

    # step 2.1 initialize OC-SVM
    rf_m = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=0)
    rf_m.fit(X, y)

    return rf_m


def evaluate(model, test_f, name='test'):
    X, y = load_data(test_f, start_feat_idx=0, shuffle_flg=True)
    y_preds = model.predict(X)
    # y_preds = (model.predict(X) == -1) * 1
    # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0

    # Step 2. achieve the evluation standards.
    cm = confusion_matrix(y, y_preds)
    print(name + ' confusion matrix:\n', cm)
    acc = 100.0 * sum(y == y_preds) / len(y)
    print(' Acc: %.2f%% \n' % (acc))

    return acc, cm


def run_svm_main(original_train_f, original_test_f, original_select_train_f, new_train_f, out_dir='log', **kwargs):
    print('--svm train and evalute')
    orig_svm_m = train_svm(original_train_f)
    evaluate(orig_svm_m, original_train_f, name='original_train_f')
    evaluate(orig_svm_m, original_test_f, name='original_test_f')

    orig_select_svm_m = train_svm(original_select_train_f)
    evaluate(orig_select_svm_m, original_select_train_f, name='original_select_train_f')
    evaluate(orig_select_svm_m, original_test_f, name='original_test_f')

    new_svm_m = train_svm(new_train_f)
    evaluate(new_svm_m, new_train_f, name='new_train_f')
    evaluate(new_svm_m, original_test_f, name='original_test_f')


def run_rf_main(original_train_f, original_test_f, original_select_train_f, new_train_f, out_dir='log', **kwargs):
    print('--rf train and evalute')
    orig_rf_m = train_rf(original_train_f)
    evaluate(orig_rf_m, original_train_f, name='original_train_f')
    evaluate(orig_rf_m, original_test_f, name='original_test_f')

    orig_select_rf_m = train_rf(original_select_train_f)
    evaluate(orig_select_rf_m, original_select_train_f, name='original_select_train_f')
    evaluate(orig_select_rf_m, original_test_f, name='original_test_f')

    new_rf_m = train_rf(new_train_f)
    evaluate(new_rf_m, new_train_f, name='new_train_f')
    evaluate(new_rf_m, original_test_f, name='original_test_f')


def split_mix_data(input_f='mix_normal_attack', n_components=20):
    X_normal=[]
    X_attack=[]
    with open(input_f,'r') as in_f:
        line = in_f.readline()
        while line:
            if line.startswith('"Private"'):
                line = in_f.readline()
                continue
            line_arr= line.split(',')
            if line_arr[-1].strip('\n') =='0' or line_arr[-1].strip('\n')=='BENIGN': #line_arr[0] == '"Yes"' or
                X_normal.append(line_arr[:-1])
            elif line_arr[-1].strip('\n') =='1' or line_arr[-1].strip('\n')=='DDoS':   # line_arr[0] =='"No"' or
                X_attack.append(line_arr[:-1])
            else:
                print('line:',line)
                pass
            line = in_f.readline()

    normal_f = os.path.join(os.path.split(input_f)[0], 'normal_demo.csv')
    attack_f = os.path.join(os.path.split(input_f)[0], 'attack_demo.csv')

    save_num = min(len(X_normal), len(X_attack))
    random.shuffle(X_normal)
    random.shuffle(X_attack)

    # # PCA dimension reduction
    # normal_pca = PCA(n_components)
    # # normal_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    # X_normal= np.asarray(X_normal, dtype=float)
    # X_normal_reduced=normal_pca.fit_transform(normalizate_data_with_u_std(X_normal,u_std_dict={'u': np.mean(X_normal,axis=0), 'std': np.std(X_normal,axis=0)}))
    # print('normal_pca:',normal_pca.explained_variance_ratio_)
    # print('normal_pca sum:', sum(normal_pca.explained_variance_ratio_))
    # X_normal=np.asarray(X_normal_reduced,dtype=str)
    #
    # attack_pca= PCA(n_components)
    # # attack_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    # X_attack = np.asarray(X_attack,dtype=float)
    # X_attack_reduced=attack_pca.fit_transform(normalizate_data_with_u_std(X_attack, u_std_dict={'u':  np.mean(X_attack,axis=0), 'std':  np.std(X_attack,axis=0)}))
    # print('attack_pca:',attack_pca.explained_variance_ratio_)
    # print('attack_pca sum:', sum(attack_pca.explained_variance_ratio_))
    # X_attack=np.asarray(X_attack_reduced,dtype=str)

    with open(normal_f, 'w') as out_f:
        for x in X_normal[:save_num+1]:
            line_str = ','.join(x) +'\n'
            out_f.write(line_str)

    with open(attack_f, 'w') as out_f:
        for x in X_attack[:save_num+1]:
            line_str = ','.join(x) +'\n'
            out_f.write(line_str)

    return normal_f, attack_f


def parse_UNB_CSV(input_f):

    output_f = input_f+'_new.csv'
    with open(output_f,'w') as out_h:
        with open(input_f, 'r') as in_h:
            err_line_cnt = 0
            for i, line in enumerate(in_h):
                # if line.startswith('')
                if 'Infinity' in line:
                    # print(line)
                    err_line_cnt +=1
                    continue
                line_arr = line.split(',')
                line = line_arr[2] + ',' + line_arr[4]+',' + line_arr[5]+','+','.join(line_arr[7:-1]) +'\n'
                out_h.write(line)
    print('err_line_cnt:',err_line_cnt)
    return output_f

def save_data(X, y, output_f):
    with open(output_f, 'w') as out_h:
        for i, line in enumerate(zip(X,y)):
            line_str = list(map(lambda x:str(x),X[i]))
            line_str = ','.join(line_str) + ','+str(y[i])
            out_h.write(line_str+'\n')

    return output_f

def demo_test(output_dir='log', name='demo'):
    # normal_f, attack_f = split_mix_data('/home/kun/PycharmProjects/gan_traffic_generation_20181024/data/new_binary_dataset.csv')
    input_f = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    normal_f, attack_f = split_mix_data(parse_UNB_CSV(input_f))
    # normal_f, attack_f = split_mix_data(
    #     '/home/kun/PycharmProjects/gan_traffic_generation_20181024/data/pima-indians-diabetes.data.csv')
    print('normal_f:',normal_f)
    print('attack_f:',attack_f)
    # exit()

    # input_files_dict = {'normal_files': 'data/benign_data.csv', 'attack_files': 'data/attack_data.csv'}
    # input_files_dict = {'normal_files': 'data/normal_demo.txt', 'attack_files': 'data/attack_demo.txt'}
    # normal_f = input_files_dict['normal_files']
    # attack_f = input_files_dict['attack_files']
    # step 1. mix normal (label = 0)and attack (label=1) data, then normalize the mixed data [0,1]
    (X, y), output_f = mix_normal_attack_and_label(normal_f, attack_f, label_dict={'normal': '0', 'attack': '1'},
                                                   start_feat_idx=[0, '-'],
                                                   output_f=os.path.join(output_dir, 'original_mix_data.csv'))
    # # # # PCA dimension reduction
    pca_m = PCA(n_components=5)
    # # normal_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X= np.asarray(X, dtype=float)
    X,_,_,_= normalizate_data(X)
    X_reduced=pca_m.fit_transform(X)  # in this function, it will do x-mean(x,axis=0), so there is no need to do x-mean(x, axis=0) before.
    print('pca_m:',pca_m.explained_variance_ratio_)
    print('pca sum:', sum(pca_m.explained_variance_ratio_))
    X= X_reduced
    y = np.asarray(y, dtype=int)
    # _ = save_numpy_data((X, y), output_f=os.path.join(output_dir, 'original_normalized_mix_data.txt'))
    pca_show(X, y)
    normal_f, attack_f = split_mix_data(save_data(X,y, output_f=input_f+'reduced-dims.csv'))
    # X = np.asarray(X, dtype=float)
    X,_,_,_ = normalizate_data(np.asarray(X, dtype=float))

    acc_dict={'svm':{'train':[],'test':[]},'RF':{'train':[],'test':[]},'mlp':{'train':[],'test':[]}}
    train_size_lst=[]
    print('X[0]:',X[0,:])
    X_train_org, X_test, y_train_org, y_test = train_test_split(np.asarray(X, dtype=float),
                                                        np.asarray(y, dtype=int),
                                                        test_size=0.3, random_state=1)
    for i, train_size_tmp in enumerate(np.logspace(np.log10(0.01), np.log10(0.7), num=10, base=10)):
        # step 2. split train (70%) and test (30%) on original mixed data
        print('i=%d, train_size=%f'%(i,train_size_tmp))
        train_size_lst.append(train_size_tmp)
        X_train, _, y_train, _ = train_test_split(X_train_org,y_train_org, train_size=train_size_tmp, random_state=1)

        # step 3. save train and test to file.
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)

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

        y_preds = svm_m.predict(X_test)
        # y_preds = (rf_m.predict(X_test) == -1) * 1
        # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
        cm = confusion_matrix(y_test, y_preds)
        print(name + ' svm test confusion matrix:\n', cm)
        acc = 100.0 * sum(y_test == y_preds) / len(y_test)
        print(' Acc: %.2f%% \n' % (acc))
        acc_dict['svm']['test'].append(acc)


        rf_m = RandomForestClassifier(n_estimators=3, max_depth=3, min_samples_leaf=5, random_state=0)  #n_estimators=10, max_depth=None,
        rf_m.fit(X_train, y_train)
        # print(rf_m.t)

        y_preds = rf_m.predict(X_train)
        # y_preds = (rf_m.predict(X_test) == -1) * 1
        # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
        cm = confusion_matrix(y_train, y_preds)
        print(name + ' RF confusion matrix:\n', cm)
        acc = 100.0 * sum(y_train == y_preds) / len(y_train)
        print(' Acc: %.2f%% \n' % (acc))
        acc_dict['RF']['train'].append(acc)

        y_preds = rf_m.predict(X_test)
        # y_preds = (rf_m.predict(X_test) == -1) * 1
        # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
        cm = confusion_matrix(y_test, y_preds)
        print(name + ' RF confusion matrix:\n', cm)
        acc = 100.0 * sum(y_test == y_preds) / len(y_test)
        print(' Acc: %.2f%% \n' % (acc))
        acc_dict['RF']['test'].append(acc)

        mlp_m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)  #n_estimators=10, max_depth=None,
        mlp_m.fit(X_train, y_train)
        # print(rf_m.t)

        y_preds = mlp_m.predict(X_train)
        # y_preds = (rf_m.predict(X_test) == -1) * 1
        # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
        cm = confusion_matrix(y_train, y_preds)
        print(name + ' mlp confusion matrix:\n', cm)
        acc = 100.0 * sum(y_train == y_preds) / len(y_train)
        print(' Acc: %.2f%% \n' % (acc))
        acc_dict['mlp']['train'].append(acc)

        y_preds = mlp_m.predict(X_test)
        # y_preds = (rf_m.predict(X_test) == -1) * 1
        # y_pred = (self.ocsvm.predict(X) == -1) * 1  # -1=>True, 0=>False, then True=>1, False=>0
        cm = confusion_matrix(y_test, y_preds)
        print(name + ' mlp confusion matrix:\n', cm)
        acc = 100.0 * sum(y_test == y_preds) / len(y_test)
        print(' Acc: %.2f%% \n' % (acc))
        acc_dict['mlp']['test'].append(acc)


    return train_size_lst, acc_dict

def show_plot(train_size_lst, acc_dict=[], x_label='train_size', y_label='acc',title='', fig_label=''):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_size_lst, acc_dict['train'],  '-*g',scalex=True, alpha=0.5, label='train')
    plt.plot(train_size_lst, acc_dict['test'], '-+r', scalex=True, alpha=0.5, label='test')
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    flg_demo = 1
    if flg_demo:
        train_size_lst,acc_dict = demo_test()
        print('train_size_lst:',train_size_lst)
        show_plot(train_size_lst, acc_dict['svm'],x_label='train_size', y_label='acc', title='svm',fig_label='svm')
        show_plot(train_size_lst, acc_dict['RF'],x_label='train_size', y_label='acc', title='RF')
        show_plot(train_size_lst, acc_dict['mlp'], x_label='train_size', y_label='acc', title='mlp')
        exit()

    # original_train_f, (normal_train_set_f, attack_train_set_f) = achieve_train_test_set(normal_f, attack_f, label_dict,
    #                                                                                     output_dir)

    original_train_f = '/home/kun/PycharmProjects/gan_traffic_generation_20181024/log/original_normalized_train_set.csv'
    original_test_f = '/home/kun/PycharmProjects/gan_traffic_generation_20181024/log/original_normalized_test_set.csv'
    original_select_train_f = '/home/kun/PycharmProjects/gan_traffic_generation_20181024/log/original_select_train_set.csv'
    new_train_f = '/home/kun/PycharmProjects/gan_traffic_generation_20181024/log/0_new_train_set.csv'

    input_files_dict = {'normal_files': 'data/normal_demo.txt', 'attack_files': 'data/attack_demo.txt'}
    # input_files_dict = {'normal_files': 'data/benign_data.csv', 'attack_files': 'data/attack_data.csv'}
    if os.path.exists(original_train_f) and os.path.exists(original_test_f) and os.path.exists(
            original_select_train_f) and os.path.exists(new_train_f):
        print('all the files are exists.')
        pass
    else:
        original_train_f, original_test_f, original_select_train_f, (
        normal_train_set_f, attack_train_set_f) = achieve_train_test_set(normal_f=input_files_dict['normal_files'],
                                                                         attack_f=input_files_dict['attack_files'],
                                                                         label_dict={'normal': '0', 'attack': '1'},
                                                                         select_train_size=0.3, output_dir='./log')

    run_svm_main(original_train_f, original_test_f, original_select_train_f, new_train_f, out_dir='./log')
    run_rf_main(original_train_f, original_test_f, original_select_train_f, new_train_f, out_dir='./log')
