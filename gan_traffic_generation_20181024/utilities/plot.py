
import numpy as np

def open_file(input_f):

    X=[]
    with open(input_f,'r') as in_h:
        for line in in_h:
            X.append(line.strip().split(','))

    X = np.asarray(X, dtype= float
                   )
    return X

def show_figures(D_loss, G_loss,name):
    import matplotlib.pyplot as plt
    plt.figure()

    plt.plot(D_loss, 'r', alpha=0.5, label='D_loss of real and fake sample')
    plt.plot(G_loss, 'g', alpha=0.5, label='D_loss of G generated fake sample')
    plt.legend(loc='upper right')
    plt.title(name+ ': D\'s loss of real and fake sample.')
    plt.show()


def show_figures_2(decision_data, name):
    import matplotlib.pyplot as plt
    plt.figure()
    new_decision_data = np.copy(decision_data)
    plt.plot(new_decision_data[:, 0], 'r', alpha=0.5, label='D_predict_value_of_real_sample')
    plt.plot(new_decision_data[:, 1], 'b', alpha=0.5, label='D_predict_value_of_fake_sample')
    # plt.plot(new_decision_data[:, 2], 'g', alpha=0.5, label='D_G_fake')
    plt.legend(loc='upper right')
    plt.title(name+': the D\'s predict value of real and fake sample.')
    plt.show()


if __name__ == '__main__':

    for name in ['normal','attack']:
        X = open_file(input_f='../log/%s_D_loss+G_loss.txt'%name)
        show_figures(X[:,0], X[:,1],name)
        X = open_file(input_f='../log/%s_D_decision.txt'%name)
        show_figures_2(X,name)