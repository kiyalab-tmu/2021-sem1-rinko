from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save(data, f_name, xlabel, ylabel, figs_path, default_path='./exports/'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(list(range(len(data))), data, label=f_name)
    ax1.set_xlabel(xlabel)
    ax1.legend()
    ax1.set_ylabel(ylabel)
    ax1.set_title(ylabel + ' / epoch')
    fig1.savefig(default_path + figs_path + f_name  + '_graph.jpg')

def Fig_train_valid(train, valid, ylabel, figs_path, default_path='./exports/', xlabel='epoch'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(list(range(len(train))), train, label='train ' + ylabel)
    ax1.plot(list(range(len(valid))), valid, label='valid ' + ylabel)
    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(ylabel + ' / epoch')
    fig1.savefig(default_path + figs_path + ylabel  + '_graph.jpg')

# 関数名を confusion_matrix にすると、 cm = confusion_matrix() でsklearnの方ではなく、ここの関数を呼び出しやがる。気を付けて。
def Fig_confusion_matrix(y_pred, y_true, labels, figs_path, default_path='./exports/', f_name='confusion_matrix'):
    plt.figure(figsize=(12, 9))
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float)
    cm /= np.sum(cm, axis=1)
    cm = pd.DataFrame(data=cm * 100, index=labels, columns=labels)
    seaborn.heatmap(cm, annot=True, cmap='Blues')
    plt.savefig(default_path + figs_path + f_name  + '_graph.jpg')