import matplotlib.pyplot as plt

def save(data, f_name, xlabel, ylabel, figs_path, default_path='./exports/'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(list(range(len(data))), data, label=f_name)
    ax1.set_xlabel(xlabel)
    ax1.legend()
    ax1.set_ylabel(ylabel)
    ax1.set_title(ylabel + ' / epoch')
    fig1.savefig(default_path + figs_path + f_name  + '_graph.jpg')