# -*- coding: utf-8 -*-

"""
 @ showcase

 created on 20180615
"""

__author__ = 'Learn-Live'


import numpy as np


def show_figures(D_loss, G_loss):
    import matplotlib.pyplot as plt
    plt.figure()

    plt.plot(D_loss, 'r', alpha=0.5, label='Wgan_distance(D_loss-G_loss)')
    # plt.plot(G_loss, 'g', alpha=0.5, label='G_loss')
    plt.legend(loc='upper right')
    plt.show()


def show_figures_2(decision_data):
    import matplotlib.pyplot as plt
    plt.figure()
    new_decision_data = np.copy(decision_data)
    plt.plot(new_decision_data[:, 0], 'r', alpha=0.5, label='D_real')
    # plt.plot(new_decision_data[:,1], 'b', alpha =0.5,label='D_fake')
    plt.plot(new_decision_data[:, 2], 'g', alpha=0.5, label='D_G_fake')
    plt.legend(loc='upper right')
    plt.show()
